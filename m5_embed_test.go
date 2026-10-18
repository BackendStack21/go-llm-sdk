package llm

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// b64Vec encodes v as little-endian float32, base64 (OpenAI's
// encoding_format=base64 wire shape).
func b64Vec(v []float32) string {
	buf := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(f))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

// M5 bug 1: OpenAI-compatible Embed must request encoding_format=base64 (a
// 2048x1536 float-array response overflows the 50MB body cap) and decode
// little-endian float32 base64, while still accepting float arrays from
// providers that ignore the parameter.
func TestM5Embed_OpenAIBase64(t *testing.T) {
	vecs := [][]float32{{0.5, -1.25, 3}, {8, 0.015625, -2}}
	asArray := func(v []float32) string {
		parts := make([]string, len(v))
		for i, f := range v {
			parts[i] = fmt.Sprintf("%v", f)
		}
		return "[" + strings.Join(parts, ",") + "]"
	}
	cases := []struct {
		name string
		body func() string
	}{
		{"base64 response", func() string {
			return fmt.Sprintf(`{"data":[{"index":1,"embedding":%q},{"index":0,"embedding":%q}],"usage":{"prompt_tokens":4}}`,
				b64Vec(vecs[1]), b64Vec(vecs[0]))
		}},
		{"float-array fallback (provider ignores encoding_format)", func() string {
			return fmt.Sprintf(`{"data":[{"index":1,"embedding":%s},{"index":0,"embedding":%s}],"usage":{"prompt_tokens":4}}`,
				asArray(vecs[1]), asArray(vecs[0]))
		}},
		{"mixed per item", func() string {
			return fmt.Sprintf(`{"data":[{"index":0,"embedding":%q},{"index":1,"embedding":%s}],"usage":{"prompt_tokens":4}}`,
				b64Vec(vecs[0]), asArray(vecs[1]))
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(b, &got)
				_, _ = fmt.Fprint(w, tc.body())
			}))
			t.Cleanup(srv.Close)
			sdk := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
			res, err := sdk.Embed(context.Background(), "openai", "emb", EmbedRequest{Inputs: []string{"a", "b"}})
			if got["encoding_format"] != "base64" {
				t.Errorf("wire encoding_format = %v, want base64", got["encoding_format"])
			}
			if err != nil {
				t.Fatalf("Embed: %v", err)
			}
			if len(res.Embeddings) != 2 {
				t.Fatalf("got %d vectors", len(res.Embeddings))
			}
			for i, want := range vecs {
				if len(res.Embeddings[i]) != len(want) {
					t.Fatalf("vector %d len = %d, want %d", i, len(res.Embeddings[i]), len(want))
				}
				for j, f := range want {
					if res.Embeddings[i][j] != float64(f) {
						t.Errorf("vec[%d][%d] = %v, want %v", i, j, res.Embeddings[i][j], f)
					}
				}
			}
			if res.Usage.PromptTokens != 4 {
				t.Errorf("usage = %+v", res.Usage)
			}
		})
	}
}

// A base64 payload whose byte length is not a multiple of 4 is a protocol
// failure, never a silently truncated vector.
func TestM5Embed_OpenAIBase64BadLength(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprintf(w, `{"data":[{"index":0,"embedding":%q}]}`, base64.StdEncoding.EncodeToString([]byte{1, 2, 3, 4, 5}))
	}))
	t.Cleanup(srv.Close)
	sdk := New(WithProvider("openai", WithAPIKey("k"), WithBaseURL(srv.URL)))
	res, err := sdk.Embed(context.Background(), "openai", "emb", EmbedRequest{Inputs: []string{"a"}})
	if err == nil {
		t.Fatalf("want error for 5-byte base64 vector, got %v", res.Embeddings)
	}
	if strings.Contains(err.Error(), "json: cannot unmarshal") {
		t.Errorf("want a base64-length error, got the float-array decode error: %v", err)
	}
}
