package server

import (
	"context"
	"fmt"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestMLXSplitRenderGenStub(t *testing.T) {
	rendered := "<bos><|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
	stable, stub := mlxSplitRenderGenStub(rendered)
	if stub != "<|turn>model\n<|channel>thought\n<channel|>" {
		t.Fatalf("stub=%q", stub)
	}
	if stable != "<bos><|turn>user\nHi<turn|>\n" {
		t.Fatalf("stable=%q", stable)
	}
}

func TestMLXPromptChainTokensForRenderExtendsStablePrefix(t *testing.T) {
	globalMLXPromptChain = mlxPromptChainCache{entries: make(map[string]mlxPromptChainEntry)}
	stable1 := "<bos><|turn>system\nsys<turn|>\n<|turn>user\nHi<turn|>\n"
	msgs1 := []api.Message{{Role: "system", Content: "sys"}, {Role: "user", Content: "Hi"}}
	globalMLXPromptChain.remember("hermes:s1", stable1, []int{1, 2, 3, 4}, msgs1)

	tokenize := func(_ context.Context, s string) ([]int, error) {
		switch s {
		case "<|turn>model\nHello.<turn|>\n<|turn>user\nBye<turn|>\n<|turn>model\n<|channel>thought\n<channel|>":
			return []int{10, 11, 12, 13}, nil
		default:
			t.Fatalf("unexpected suffix %q", s)
			return nil, nil
		}
	}

	stable2 := stable1 + "<|turn>model\nHello.<turn|>\n<|turn>user\nBye<turn|>\n"
	rendered2 := stable2 + "<|turn>model\n<|channel>thought\n<channel|>"
	msgs2 := append(msgs1, api.Message{Role: "assistant", Content: "Hello."}, api.Message{Role: "user", Content: "Bye"})

	got, ok := mlxPromptChainTokensForRender(context.Background(), "hermes:s1", rendered2, msgs2, tokenize)
	if !ok {
		t.Fatal("expected splice ok")
	}
	want := []int{1, 2, 3, 4, 10, 11, 12, 13}
	if len(got) != len(want) {
		t.Fatalf("tokens=%v want %v", got, want)
	}
}

func TestMLXPromptChainTokensForRenderIdenticalReplay(t *testing.T) {
	globalMLXPromptChain = mlxPromptChainCache{entries: make(map[string]mlxPromptChainEntry)}
	stable := "abc\n"
	msgs := []api.Message{{Role: "user", Content: "x"}}
	globalMLXPromptChain.remember("k", stable, []int{5, 6, 7}, msgs)

	rendered := stable + "<|turn>model\n<|channel>thought\n<channel|>"
	got, ok := mlxPromptChainTokensForRender(context.Background(), "k", rendered, msgs, func(_ context.Context, s string) ([]int, error) {
		if s == "<|turn>model\n<|channel>thought\n<channel|>" {
			return []int{99}, nil
		}
		t.Fatalf("unexpected %q", s)
		return nil, nil
	})
	if !ok {
		t.Fatal("expected splice ok")
	}
	if len(got) != 4 || got[3] != 99 {
		t.Fatalf("got %v", got)
	}
}

func TestMLXPromptChainTokensForRenderMissWhenMessagesEdited(t *testing.T) {
	globalMLXPromptChain = mlxPromptChainCache{entries: make(map[string]mlxPromptChainEntry)}
	msgs1 := []api.Message{{Role: "user", Content: "old"}}
	globalMLXPromptChain.remember("k", "stable\n", []int{1, 2}, msgs1)

	msgs2 := []api.Message{{Role: "user", Content: "new"}}
	_, ok := mlxPromptChainTokensForRender(context.Background(), "k", "stable\n<|turn>model\n", msgs2, func(context.Context, string) ([]int, error) {
		t.Fatal("should not tokenize")
		return nil, nil
	})
	if ok {
		t.Fatal("expected miss when message content changed")
	}
}

func TestMLXPromptChainInvalidate(t *testing.T) {
	c := &mlxPromptChainCache{entries: make(map[string]mlxPromptChainEntry)}
	c.remember("k", "stable\n", []int{1, 2, 3}, []api.Message{{Role: "user", Content: "x"}})
	c.invalidate("k")
	if _, ok := c.lookup("k"); ok {
		t.Fatal("expected entry removed after invalidate")
	}
}

func TestMLXPromptChainReconcilePrefersSplice(t *testing.T) {
	spliced := []int{1, 2, 3, 99}
	fresh := []int{9, 8, 7, 6, 5}
	got := mlxPromptChainReconcile("k", spliced, fresh)
	if len(got) != len(spliced) || got[0] != 1 {
		t.Fatalf("got %v", got)
	}
}

func TestMLXPromptChainEviction(t *testing.T) {
	c := &mlxPromptChainCache{entries: make(map[string]mlxPromptChainEntry)}
	for i := range mlxPromptChainMaxEntries + 2 {
		c.remember(fmt.Sprintf("k%d", i), "p", []int{i, i + 1}, []api.Message{{Role: "user", Content: "x"}})
	}
	if len(c.entries) > mlxPromptChainMaxEntries {
		t.Fatalf("entries=%d want <= %d", len(c.entries), mlxPromptChainMaxEntries)
	}
}
