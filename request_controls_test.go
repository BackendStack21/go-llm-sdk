package llm

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

func decodeObject(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode request: %v\nbody: %s", err, body)
	}
	return got
}

func TestRequestControlsMapAcrossFormats(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{{Role: RoleUser, Content: "hello"}},
		TopP:     -1, // canonical explicit zero
		Stop:     []string{"END", "DONE"},
	}

	t.Run("openai", func(t *testing.T) {
		wire := buildOpenAIRequest(
			ProviderConfig{ID: "openai", Format: FormatOpenAI},
			req,
			"gpt-4o",
			false,
			false,
		)
		body, err := json.Marshal(wire)
		if err != nil {
			t.Fatal(err)
		}
		got := decodeObject(t, body)
		assertSamplingControls(t, got, "top_p", "stop")
	})

	t.Run("gemini", func(t *testing.T) {
		body, err := buildGeminiRequest(req, "gemini-2.5-pro", false)
		if err != nil {
			t.Fatal(err)
		}
		got := decodeObject(t, body)
		cfg := got["generationConfig"].(map[string]any)
		assertSamplingControls(t, cfg, "topP", "stopSequences")
	})

	t.Run("anthropic", func(t *testing.T) {
		body, err := buildAnthropicRequest(req, "claude-sonnet-4", false)
		if err != nil {
			t.Fatal(err)
		}
		got := decodeObject(t, body)
		assertSamplingControls(t, got, "top_p", "stop_sequences")
	})
}

func assertSamplingControls(t *testing.T, got map[string]any, topPKey, stopKey string) {
	t.Helper()
	if got[topPKey] != float64(0) {
		t.Errorf("%s = %v, want explicit 0", topPKey, got[topPKey])
	}
	stop, ok := got[stopKey].([]any)
	if !ok || len(stop) != 2 || stop[0] != "END" || stop[1] != "DONE" {
		t.Errorf("%s = %#v, want [END DONE]", stopKey, got[stopKey])
	}
}

func TestRequestControlsOmitDefaultsAcrossFormats(t *testing.T) {
	req := &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hello"}}}
	tests := []struct {
		name    string
		build   func(t *testing.T) map[string]any
		topPKey string
		stopKey string
	}{
		{
			name: "openai",
			build: func(t *testing.T) map[string]any {
				body, err := json.Marshal(buildOpenAIRequest(ProviderConfig{Format: FormatOpenAI}, req, "gpt-4o", false, false))
				if err != nil {
					t.Fatal(err)
				}
				return decodeObject(t, body)
			},
			topPKey: "top_p",
			stopKey: "stop",
		},
		{
			name: "gemini",
			build: func(t *testing.T) map[string]any {
				body, err := buildGeminiRequest(req, "gemini-2.5-pro", false)
				if err != nil {
					t.Fatal(err)
				}
				got := decodeObject(t, body)
				cfg, _ := got["generationConfig"].(map[string]any)
				return cfg
			},
			topPKey: "topP",
			stopKey: "stopSequences",
		},
		{
			name: "anthropic",
			build: func(t *testing.T) map[string]any {
				body, err := buildAnthropicRequest(req, "claude-sonnet-4", false)
				if err != nil {
					t.Fatal(err)
				}
				return decodeObject(t, body)
			},
			topPKey: "top_p",
			stopKey: "stop_sequences",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.build(t)
			if _, ok := got[tt.topPKey]; ok {
				t.Errorf("%s must be omitted at its default", tt.topPKey)
			}
			if _, ok := got[tt.stopKey]; ok {
				t.Errorf("%s must be omitted when empty", tt.stopKey)
			}
		})
	}
}

func TestChatClientRejectsNilRequest(t *testing.T) {
	pc := newProviderClient(
		ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: "http://127.0.0.1:1", APIKey: "k"},
		nil,
		nil,
	)
	client := &ChatClient{pc: pc, model: "gpt-4o"}

	tests := []struct {
		name string
		call func() error
	}{
		{
			name: "buffered",
			call: func() error {
				_, err := client.Call(context.Background(), nil)
				return err
			},
		},
		{
			name: "streaming",
			call: func() error {
				_, err := client.CallStream(context.Background(), nil, func(Delta) error { return nil })
				return err
			},
		},
		{
			name: "nil handler",
			call: func() error {
				_, err := client.CallStream(context.Background(), nil, nil)
				return err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cfgErr *ConfigError
			if err := tt.call(); !errors.As(err, &cfgErr) {
				t.Fatalf("error = %v, want ConfigError", err)
			}
		})
	}
}
