package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func m1StreamServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func m1OpenAIClient(t *testing.T, srv *httptest.Server) *ChatClient {
	t.Helper()
	return newTestClient(t, ProviderConfig{ID: "openai", Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k"}, srv)
}

func m1Req() *ChatRequest {
	return &ChatRequest{Messages: []Message{{Role: RoleUser, Content: "hi"}}}
}

func m1Chunk(content string) string {
	return fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", content)
}

const m1Finish = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"

// Bug 1: a slow consumer callback must not trip the idle watchdog; only
// provider silence counts.
func TestM1_SlowOnDeltaDoesNotTripIdleWatchdog(t *testing.T) {
	setIdleForTest(t, 100*time.Millisecond)
	var body strings.Builder
	for i := 0; i < 8; i++ {
		body.WriteString(m1Chunk("x"))
	}
	body.WriteString(m1Finish + "data: [DONE]\n\n")
	srv := m1StreamServer(t, body.String())
	cc := m1OpenAIClient(t, srv)

	res, err := cc.CallStream(context.Background(), m1Req(), func(Delta) error {
		time.Sleep(150 * time.Millisecond)
		return nil
	})
	if err != nil {
		t.Fatalf("slow consumer must not cause an error, got: %v", err)
	}
	if res == nil || res.Content != "xxxxxxxx" || res.FinishReason != FinishStop {
		t.Fatalf("result = %+v", res)
	}
}

// Bug 2: provider error events and a bare [DONE] must not be silent success.
func TestM1_OpenAIStreamErrorAndBareDone(t *testing.T) {
	cases := []struct {
		name    string
		body    string
		wantMsg string
	}{
		{"error event then DONE", "data: {\"error\":{\"message\":\"upstream overloaded\"}}\n\ndata: [DONE]\n\n", "upstream overloaded"},
		{"bare DONE", "data: [DONE]\n\n", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fastBackoff(t)
			srv := m1StreamServer(t, tc.body)
			cc := m1OpenAIClient(t, srv)
			res, err := cc.CallStream(context.Background(), m1Req(), func(Delta) error { return nil })
			if err == nil {
				t.Fatalf("expected an error, got success: %+v", res)
			}
			if tc.wantMsg != "" && !strings.Contains(err.Error(), tc.wantMsg) {
				t.Fatalf("provider message lost: %v", err)
			}
		})
	}
}

// Bug 2 sibling: Gemini stream error chunk must preserve the provider message.
func TestM1_GeminiStreamErrorChunkPreservesMessage(t *testing.T) {
	fastBackoff(t)
	srv := m1StreamServer(t, "data: {\"error\":{\"code\":503,\"message\":\"upstream overloaded\",\"status\":\"UNAVAILABLE\"}}\n\n")
	cc := newTestClient(t, ProviderConfig{ID: "gemini", Format: FormatGemini, BaseURL: srv.URL, APIKey: "k"}, srv)
	res, err := cc.CallStream(context.Background(), m1Req(), func(Delta) error { return nil })
	if err == nil {
		t.Fatalf("expected an error, got success: %+v", res)
	}
	if !strings.Contains(err.Error(), "upstream overloaded") {
		t.Fatalf("provider message lost: %v", err)
	}
}

// Bug 3: a leading UTF-8 BOM must be stripped and bare-CR line endings
// accepted, per the SSE spec.
func TestM1_SSEBOMAndBareCR(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"BOM", "\xef\xbb\xbf" + m1Chunk("hello") + m1Finish + "data: [DONE]\n\n"},
		{"bare CR", strings.NewReplacer("\n", "\r").Replace(m1Chunk("hello") + m1Finish + "data: [DONE]\n\n")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := m1StreamServer(t, tc.body)
			cc := m1OpenAIClient(t, srv)
			res, err := cc.CallStream(context.Background(), m1Req(), func(Delta) error { return nil })
			if err != nil {
				t.Fatalf("err: %v", err)
			}
			if res.Content != "hello" || res.FinishReason != FinishStop {
				t.Fatalf("first event dropped or stream mis-parsed: %+v", res)
			}
		})
	}
}
