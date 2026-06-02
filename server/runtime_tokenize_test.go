package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/runtimeworker"
)

func TestMemoizeTokenize(t *testing.T) {
	calls := 0
	inner := func(ctx context.Context, content string) ([]int, error) {
		calls++
		return []int{len(content)}, nil
	}
	tok := memoizeTokenize(inner)
	ctx := context.Background()
	if _, err := tok(ctx, "ab"); err != nil {
		t.Fatal(err)
	}
	if _, err := tok(ctx, "ab"); err != nil {
		t.Fatal(err)
	}
	if _, err := tok(ctx, "xyz"); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("calls=%d want 2 (cache hit on duplicate prompt)", calls)
	}
}

func TestRenderChatUsesRuntimeTokenizeEmbedded(t *testing.T) {
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/tokenize" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"tokens": []int{2, 2, 2},
			"count":  3,
		})
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	runtimeworker.SetBaseURLForTest(rt.URL)
	t.Cleanup(runtimeworker.ClearBaseURLForTest)

	s := &Server{}
	m := &Model{
		Name:      "rt",
		ModelPath: "/data/rt.gguf",
		Template:  testRenderTemplate(t),
	}
	p, mode, _, _, err := s.renderChatPromptPrepared(
		context.Background(),
		m,
		[]api.Message{{Role: "user", Content: "hello"}},
		nil,
		nil,
		8,
		32,
		true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if mode != "tokenize" {
		t.Fatalf("mode=%q want tokenize (embedded runtime URL)", mode)
	}
	if p == "" {
		t.Fatal("expected prompt")
	}
}

func TestRenderChatUsesRuntimeTokenize(t *testing.T) {
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/tokenize" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"tokens": []int{1, 1, 1, 1, 1},
			"count":  5,
		})
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	s := &Server{}
	m := &Model{
		Name:      "rt",
		ModelPath: "/data/rt.gguf",
		Template:  testRenderTemplate(t),
	}
	msgs := []api.Message{
		{Role: "user", Content: "hello"},
	}
	p, mode, dropped, _, err := s.renderChatPromptPrepared(
		context.Background(),
		m,
		msgs,
		nil,
		nil,
		8,
		32,
		true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if mode != "tokenize" {
		t.Fatalf("mode=%q want tokenize", mode)
	}
	if dropped {
		t.Fatalf("unexpected drop on short prompt: %q", p)
	}
}

func TestRenderChatRuntimeTokenizeUnavailableFallsBack(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:1")
	s := &Server{}
	m := &Model{
		Name:      "rt",
		ModelPath: "/data/rt.gguf",
		Template:  testRenderTemplate(t),
	}
	msgs := []api.Message{{Role: "user", Content: "hello"}}
	_, mode, _, _, err := s.renderChatPromptPrepared(
		context.Background(), m, msgs, nil, nil, 8, 32, true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if mode != "heuristic" {
		t.Fatalf("mode=%q want heuristic when runtime tokenize unreachable", mode)
	}
}

func TestRenderChatRuntimeTokenizeHTTPErrorFails(t *testing.T) {
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not found", http.StatusNotFound)
	}))
	defer rt.Close()
	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	s := &Server{}
	m := &Model{
		Name:      "rt",
		ModelPath: "/data/missing.gguf",
		Template:  testRenderTemplate(t),
	}
	msgs := []api.Message{{Role: "user", Content: "hello"}}
	_, _, _, _, err := s.renderChatPromptPrepared(
		context.Background(), m, msgs, nil, nil, 8, 32, true,
	)
	if err == nil {
		t.Fatal("expected error when runtime tokenize returns HTTP 404")
	}
}
