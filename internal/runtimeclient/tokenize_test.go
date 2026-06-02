package runtimeclient

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestTokenize(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/tokenize" {
			t.Fatalf("path=%s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"tokens": []int{1, 2, 3},
			"count":  3,
		})
	}))
	defer srv.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", srv.URL)
	toks, ok, err := Tokenize(context.Background(), "/m.gguf", "hello", true)
	if err != nil || !ok {
		t.Fatalf("err=%v ok=%v", err, ok)
	}
	if len(toks) != 3 || toks[0] != 1 {
		t.Fatalf("tokens=%v", toks)
	}
}

func TestTokenizeUnavailable(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:1")
	_, ok, err := Tokenize(context.Background(), "/m.gguf", "hello", true)
	if ok {
		t.Fatal("expected ok=false")
	}
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err=%v want ErrUnavailable", err)
	}
}
