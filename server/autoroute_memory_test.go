package server

import (
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/server/autoroute"
)

func chatSig(t *testing.T, req api.ChatRequest) autoroute.Signals {
	t.Helper()
	c, _ := gin.CreateTestContext(NewRecorder())
	return chatRouteSignals(c, &req)
}

func turn(msgs ...string) []api.Message {
	out := []api.Message{{Role: "system", Content: "you are terse"}}
	for i, m := range msgs {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		out = append(out, api.Message{Role: role, Content: m})
	}
	return out
}

func TestStickyRoutingKeepsConversationModel(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	createRouteTestModel(t, s, "tooly", routeToolsTemplate)
	ctx := t.Context()

	_, first, err := s.applyRouterRewrite(ctx, "auto", chatSig(t, api.ChatRequest{Messages: turn("hello")}))
	if err != nil || first.Sticky {
		t.Fatalf("turn 1: %+v %v", first, err)
	}

	_, second, err := s.applyRouterRewrite(ctx, "auto", chatSig(t, api.ChatRequest{Messages: turn("hello", "hi!", "and now?")}))
	if err != nil || !second.Sticky || second.Reason != "sticky" || second.Candidate != first.Candidate {
		t.Fatalf("turn 2 should stick to %q: %+v %v", first.Candidate, second, err)
	}

	// A different conversation (different first message) routes independently.
	_, other, _ := s.applyRouterRewrite(ctx, "auto", chatSig(t, api.ChatRequest{Messages: turn("something else")}))
	if other.Sticky {
		t.Fatalf("new conversation reused sticky pick: %+v", other)
	}

	// Fresh opts out for one turn.
	_, fresh, _ := s.applyRouterRewrite(ctx, "auto", chatSig(t, api.ChatRequest{
		Messages: turn("hello", "hi!", "again"),
		Options:  map[string]any{"zerollama": map[string]any{"route": "fresh"}},
	}))
	if fresh.Sticky || fresh.Cached {
		t.Fatalf("fresh must re-route: %+v", fresh)
	}
}

func TestStickyBreaksOnHardFilter(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	ctx := t.Context()

	_, first, err := s.applyRouterRewrite(ctx, "auto", chatSig(t, api.ChatRequest{Messages: turn("hello")}))
	if err != nil || !strings.HasPrefix(first.Candidate, "plain") {
		t.Fatalf("turn 1: %+v %v", first, err)
	}
	createRouteTestModel(t, s, "tooly", routeToolsTemplate)

	// Tools appear mid-conversation: plain cannot serve them, so the pick breaks and re-routes.
	_, next, err := s.applyRouterRewrite(ctx, "auto", chatSig(t, api.ChatRequest{
		Messages: turn("hello", "hi!", "weather?"), Tools: []api.Tool{weatherTool},
	}))
	if err != nil || !strings.HasPrefix(next.Candidate, "tooly") || next.Sticky {
		t.Fatalf("turn 2: %+v %v", next, err)
	}
	if !strings.Contains(next.StickyBreak, "tools") {
		t.Fatalf("sticky_break = %q", next.StickyBreak)
	}
	// The conversation now sticks to the new model.
	_, again, _ := s.applyRouterRewrite(ctx, "auto", chatSig(t, api.ChatRequest{
		Messages: turn("hello", "hi!", "weather?", "sunny", "thanks"), Tools: []api.Tool{weatherTool},
	}))
	if !again.Sticky || again.Candidate != next.Candidate {
		t.Fatalf("turn 3 should stick to the re-routed model: %+v", again)
	}
}

func TestDecisionCacheForGenerateAndDryRunsDoNotWrite(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	ctx := t.Context()
	c, _ := gin.CreateTestContext(NewRecorder())
	sig := generateRouteSignals(c, &api.GenerateRequest{Prompt: "once upon a time"})
	if sig.SessionKey != "" {
		t.Fatalf("generate without a cache key has no session: %q", sig.SessionKey)
	}

	// Dry run first: must not populate memory.
	if dec, _, err := s.resolveRouter(ctx, "auto", sig); err != nil || dec.Cached {
		t.Fatalf("dry run: %+v %v", dec, err)
	}
	_, live, _ := s.applyRouterRewrite(ctx, "auto", sig)
	if live.Cached {
		t.Fatalf("dry run wrote the cache: %+v", live)
	}
	_, again, _ := s.applyRouterRewrite(ctx, "auto", sig)
	if !again.Cached || again.Reason != "cached" || again.Candidate != live.Candidate {
		t.Fatalf("identical retry should hit the cache: %+v", again)
	}
	// The dry run now reports what the next request would do.
	if dec, _, _ := s.resolveRouter(ctx, "auto", sig); !dec.Cached {
		t.Fatalf("dry run should read memory: %+v", dec)
	}
}

func TestStickyDisabledByEnv(t *testing.T) {
	t.Setenv("ZEROLLAMA_ROUTER_STICKY_TTL", "0")
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	ctx := t.Context()
	s.applyRouterRewrite(ctx, "auto", chatSig(t, api.ChatRequest{Messages: turn("hello")}))
	_, second, _ := s.applyRouterRewrite(ctx, "auto", chatSig(t, api.ChatRequest{Messages: turn("hello", "hi", "more")}))
	if second.Sticky {
		t.Fatalf("sticky must be off with TTL 0: %+v", second)
	}
}

func TestRouteSessionKey(t *testing.T) {
	a := routeSessionKey(nil, turn("hello"))
	b := routeSessionKey(nil, turn("hello", "hi", "next turn"))
	if a == "" || a != b {
		t.Fatalf("key must be stable across turns: %q vs %q", a, b)
	}
	other := routeSessionKey(nil, []api.Message{{Role: "system", Content: "different"}, {Role: "user", Content: "hello"}})
	if other == a {
		t.Fatal("different system prompt must change the key")
	}
	if got := routeSessionKey(map[string]any{"prompt_cache_key": "thread-7"}, turn("x")); got != "key:thread-7" {
		t.Fatalf("explicit key = %q", got)
	}
	if routeSessionKey(nil, []api.Message{{Role: "system", Content: "only system"}}) != "" {
		t.Fatal("no user message ⇒ no session")
	}
}

func TestRouterAllowsModelAfterConfigEdit(t *testing.T) {
	resetRouterFile(t, `
routers:
  agent:
    classifier: unused
    fallback: qwen3:8b
    embedder: nomic-embed-text
    candidates:
      - {model: coder, labels: [code]}
`)
	for m, want := range map[string]bool{
		"coder": true, "coder:latest": true, "qwen3:8b": true, "nomic-embed-text": true, "gone": false,
	} {
		if got := routerAllowsModel("agent", m); got != want {
			t.Errorf("routerAllowsModel(agent, %q) = %v, want %v", m, got, want)
		}
	}
	if !routerAllowsModel("auto", "anything") {
		t.Fatal("implicit router allows any local model")
	}
}
