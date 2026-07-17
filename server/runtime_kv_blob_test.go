package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestKvBlobHandlerProxiesRuntime(t *testing.T) {
	digest := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	payload := []byte("slot-kv-bytes")
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/kv/blob/"+digest {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("X-Zerollama-Blob-Token") != "secret" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		w.Header().Set("X-Zerollama-Blob-Digest", digest)
		w.Header().Set("Content-Type", "application/octet-stream")
		_, _ = w.Write(payload)
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{}
	r.GET("/api/kv/blob/:digest", s.KvBlobHandler)

	req := httptest.NewRequest(http.MethodGet, "/api/kv/blob/"+digest, nil)
	req.Header.Set("X-Zerollama-Blob-Token", "secret")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	if got := w.Body.Bytes(); string(got) != string(payload) {
		t.Fatalf("body=%q", got)
	}
	if w.Header().Get("X-Zerollama-Blob-Digest") != digest {
		t.Fatalf("digest header=%q", w.Header().Get("X-Zerollama-Blob-Digest"))
	}
}

func TestKvBlobHandlerRejectsBadDigest(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:9")
	gin.SetMode(gin.TestMode)
	r := gin.New()
	s := &Server{}
	r.GET("/api/kv/blob/:digest", s.KvBlobHandler)
	req := httptest.NewRequest(http.MethodGet, "/api/kv/blob/notadigest", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status=%d", w.Code)
	}
}
