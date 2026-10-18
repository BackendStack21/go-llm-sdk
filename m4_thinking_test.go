package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

// anthropicWireBody sends req through a real ChatClient to an httptest server
// and returns the decoded Anthropic request body that hit the wire.
func anthropicWireBody(t *testing.T, req *ChatRequest, model string) map[string]any {
	t.Helper()
	var mu sync.Mutex
	var raw []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		raw = b
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer srv.Close()
	cc := newTestClient(t, ProviderConfig{ID: "anthropic", Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k"}, srv)
	cc.model = model
	if _, err := cc.Call(context.Background(), req); err != nil {
		t.Fatalf("Call: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("wire body %q: %v", raw, err)
	}
	return m
}

func userMsgs() []Message { return []Message{{Role: RoleUser, Content: "x"}} }

// AC-1: presets map to adaptive thinking + output_config.effort on the wire.
func TestM4_AdaptivePresetsOnWire(t *testing.T) {
	cases := []struct {
		level      string
		wantThink  bool
		wantEffort string
	}{
		{"enabled", true, ""},
		{"low", true, "low"},
		{"medium", true, "medium"},
		{"high", true, "high"},
		{"max", true, "max"},
		{"", false, ""},
		{"disabled", false, ""},
	}
	for _, tc := range cases {
		t.Run("level="+tc.level, func(t *testing.T) {
			m := anthropicWireBody(t, &ChatRequest{Messages: userMsgs(), Thinking: tc.level}, "claude-any")
			th, has := m["thinking"].(map[string]any)
			if has != tc.wantThink {
				t.Fatalf("thinking present = %v, want %v (%v)", has, tc.wantThink, m["thinking"])
			}
			if has {
				if len(th) != 1 || th["type"] != "adaptive" {
					t.Errorf("thinking = %v, want exactly {type:adaptive}", th)
				}
			}
			oc, hasOC := m["output_config"].(map[string]any)
			if tc.wantEffort == "" {
				if hasOC {
					t.Errorf("output_config = %v, want absent", oc)
				}
			} else if oc["effort"] != tc.wantEffort {
				t.Errorf("effort = %v, want %q", oc["effort"], tc.wantEffort)
			}
		})
	}
}

// AC-2: an explicit ThinkingBudget keeps the legacy path, with no effort.
func TestM4_ExplicitBudgetKeepsLegacyPath(t *testing.T) {
	for _, level := range []string{"enabled", "low", "high", "max"} {
		m := anthropicWireBody(t, &ChatRequest{Messages: userMsgs(), Thinking: level, ThinkingBudget: 4000}, "claude-any")
		th := m["thinking"].(map[string]any)
		if th["type"] != "enabled" || th["budget_tokens"] != float64(4000) {
			t.Errorf("%s: thinking = %v, want enabled/4000", level, th)
		}
		if m["max_tokens"].(float64) <= 4000 {
			t.Errorf("%s: max_tokens %v must exceed budget", level, m["max_tokens"])
		}
		if _, ok := m["output_config"]; ok {
			t.Errorf("%s: output_config must be absent on the budget path: %v", level, m["output_config"])
		}
	}
	// ThinkingBudget with thinking off is ignored.
	for _, level := range []string{"", "disabled"} {
		m := anthropicWireBody(t, &ChatRequest{Messages: userMsgs(), Thinking: level, ThinkingBudget: 4000}, "claude-any")
		if _, ok := m["thinking"]; ok {
			t.Errorf("level %q: thinking must be absent, got %v", level, m["thinking"])
		}
	}
	// Existing ConfigErrors still fire.
	_, err := buildAnthropicRequest(&ChatRequest{Messages: userMsgs(), Thinking: "high", ThinkingBudget: 4000, MaxTokens: 4000}, "claude", false)
	var ce *ConfigError
	if !errors.As(err, &ce) {
		t.Errorf("budget >= MaxTokens: err = %v, want ConfigError", err)
	}
}

// AC-3: adaptive default max_tokens 16384, explicit as-is, sampling removed.
func TestM4_AdaptiveMaxTokensAndSampling(t *testing.T) {
	cases := []struct {
		name    string
		req     ChatRequest
		wantMax float64
	}{
		{"adaptive default", ChatRequest{Thinking: "medium"}, 16384},
		{"adaptive explicit small", ChatRequest{Thinking: "high", MaxTokens: 512}, 512},
		{"adaptive explicit large", ChatRequest{Thinking: "max", MaxTokens: 64000}, 64000},
		{"no thinking default", ChatRequest{}, 8192},
		{"disabled default", ChatRequest{Thinking: "disabled"}, 8192},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.req.Messages = userMsgs()
			m := anthropicWireBody(t, &tc.req, "claude-any")
			if m["max_tokens"] != tc.wantMax {
				t.Errorf("max_tokens = %v, want %v", m["max_tokens"], tc.wantMax)
			}
		})
	}
	for _, req := range []ChatRequest{{Thinking: "medium"}, {Thinking: "medium", ThinkingBudget: 2048}} {
		req.Messages, req.Temperature, req.TopP = userMsgs(), 0.4, 0.6
		m := anthropicWireBody(t, &req, "claude-any")
		if _, ok := m["temperature"]; ok {
			t.Errorf("budget=%d: temperature must be absent with thinking", req.ThinkingBudget)
		}
		if _, ok := m["top_p"]; ok {
			t.Errorf("budget=%d: top_p must be absent with thinking", req.ThinkingBudget)
		}
	}
	m := anthropicWireBody(t, &ChatRequest{Messages: userMsgs(), Temperature: 0.4, TopP: 0.6}, "claude-any")
	if m["temperature"] != 0.4 || m["top_p"] != 0.6 {
		t.Errorf("sampling must survive without thinking: %v %v", m["temperature"], m["top_p"])
	}
}

// AC-6: forced tool choice passes through on adaptive thinking; the rejection
// fires only on the legacy budget path; the result never depends on the model name.
func TestM4_ForcedToolChoiceAdaptiveVsBudget(t *testing.T) {
	tools := []ToolDef{{Name: "f", Parameters: json.RawMessage(`{"type":"object"}`)}}
	choices := []struct {
		name     string
		tc       *ToolChoice
		wantType string
	}{
		{"required", &ToolChoice{Mode: ToolChoiceRequired}, "any"},
		{"named", ToolChoiceNamed("f"), "tool"},
	}
	for _, model := range []string{"claude-opus-4-8", "claude-haiku-4-5", "totally-unknown-model"} {
		for _, c := range choices {
			t.Run(model+"/"+c.name, func(t *testing.T) {
				m := anthropicWireBody(t, &ChatRequest{Messages: userMsgs(), Tools: tools, ToolChoice: c.tc, Thinking: "high"}, model)
				tcm, _ := m["tool_choice"].(map[string]any)
				if tcm["type"] != c.wantType {
					t.Errorf("tool_choice = %v, want type %s", tcm, c.wantType)
				}
				if th := m["thinking"].(map[string]any); th["type"] != "adaptive" {
					t.Errorf("thinking = %v, want adaptive", th)
				}
				_, err := buildAnthropicRequest(&ChatRequest{Messages: userMsgs(), Tools: tools, ToolChoice: c.tc, Thinking: "high", ThinkingBudget: 2000}, model, false)
				var ce *ConfigError
				if !errors.As(err, &ce) {
					t.Errorf("budget path: err = %v, want ConfigError", err)
				}
			})
		}
	}
	// auto + budget thinking stays accepted.
	if _, err := buildAnthropicRequest(&ChatRequest{Messages: userMsgs(), Tools: tools, ToolChoice: &ToolChoice{Mode: ToolChoiceAuto}, Thinking: "low", ThinkingBudget: 2000}, "claude", false); err != nil {
		t.Errorf("auto + budget thinking must pass: %v", err)
	}
}
