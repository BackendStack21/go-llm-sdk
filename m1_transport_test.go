package llm

import (
	"net/http"
	"net/url"
	"testing"
)

// Bug 7: a scheme downgrade on the same host must not be followed (it would
// resend API-key headers in plaintext).
func TestM1_SameHostRedirectRefusesSchemeDowngrade(t *testing.T) {
	mk := func(u string) *http.Request {
		pu, err := url.Parse(u)
		if err != nil {
			t.Fatal(err)
		}
		return &http.Request{URL: pu}
	}
	cases := []struct {
		name     string
		from, to string
		follow   bool
	}{
		{"https to http same host", "https://api.example.com/v1/chat", "http://api.example.com/v1/chat", false},
		{"https to http explicit port", "https://api.example.com/v1/chat", "http://api.example.com:443/v1/chat", false},
		{"https to https same host", "https://api.example.com/a", "https://api.example.com/b", true},
		{"http to https same host", "http://api.example.com/a", "https://api.example.com/b", true},
		{"other host", "https://api.example.com/a", "https://evil.example.net/a", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := sameHostRedirect(mk(tc.to), []*http.Request{mk(tc.from)})
			if tc.follow && err != nil {
				t.Fatalf("redirect should be followed, got %v", err)
			}
			if !tc.follow && err != http.ErrUseLastResponse {
				t.Fatalf("redirect should be refused, got %v", err)
			}
		})
	}
}
