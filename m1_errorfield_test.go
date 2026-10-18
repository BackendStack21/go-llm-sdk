package llm

import (
	"context"
	"strings"
	"testing"
)

// A non-object "error" value on a stream chunk is not a provider error and
// must not fail an otherwise-successful stream; an error object with a
// message still does.
func TestM1_StreamErrorFieldNonObjectIgnored(t *testing.T) {
	formats := []struct {
		name   string
		format Format
		chunk  func(errField string) string // one content chunk carrying the error field
		finish string
		want   string
	}{
		{
			"openai", FormatOpenAI,
			func(e string) string {
				return "data: {\"error\":" + e + ",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"
			},
			m1Finish + "data: [DONE]\n\n", "hello",
		},
		{
			"gemini", FormatGemini,
			func(e string) string {
				return "data: {\"error\":" + e + ",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hello\"}]}}]}\n\n"
			},
			"data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n", "hello",
		},
	}
	values := []struct {
		errField string
		wantErr  bool
	}{
		{`""`, false},
		{`false`, false},
		{`null`, false},
		{`0`, false},
		{`{"message":"upstream overloaded"}`, true},
	}
	for _, f := range formats {
		for _, v := range values {
			t.Run(f.name+"/"+v.errField, func(t *testing.T) {
				fastBackoff(t)
				srv := m1StreamServer(t, f.chunk(v.errField)+f.finish)
				cc := newTestClient(t, ProviderConfig{ID: f.name, Format: f.format, BaseURL: srv.URL, APIKey: "k"}, srv)
				res, err := cc.CallStream(context.Background(), m1Req(), func(Delta) error { return nil })
				if v.wantErr {
					if err == nil || !strings.Contains(err.Error(), "upstream overloaded") {
						t.Fatalf("error object must fail the stream with its message, got res=%+v err=%v", res, err)
					}
					return
				}
				if err != nil {
					t.Fatalf("non-object error value %s failed the stream: %v", v.errField, err)
				}
				if res.Content != f.want {
					t.Fatalf("content = %q, want %q", res.Content, f.want)
				}
			})
		}
	}
}
