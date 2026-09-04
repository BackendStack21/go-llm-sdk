package llm

import (
	"testing"
)

func TestNewPooledTransport_HonorsHTTPProxy(t *testing.T) {
	tr := newPooledTransport()
	if tr.Proxy == nil {
		t.Fatal("Proxy is nil; want http.ProxyFromEnvironment so HTTP_PROXY/HTTPS_PROXY are honored")
	}
}
