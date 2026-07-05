package server

import (
	"context"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestMLXQoSFromOptionsExplicit(t *testing.T) {
	opts := map[string]any{
		"zerollama": map[string]any{
			"qos_class":      "background",
			"session_group":  "ruby-trivia",
			"session_parent": "parent:1",
			"cache_scope":    "shared",
		},
	}
	q := mlxQoSFromOptions(opts)
	if q.Class != mlxClassBackground || !q.Explicit {
		t.Fatalf("class=%s explicit=%v", q.Class, q.Explicit)
	}
	if q.SessionGroup != "ruby-trivia" || q.ParentKey != "parent:1" {
		t.Fatalf("group=%q parent=%q", q.SessionGroup, q.ParentKey)
	}
	if q.CacheScope != qosCacheScopeShared {
		t.Fatalf("scope=%q", q.CacheScope)
	}
}

func TestMLXQoSFromOptionsPriority(t *testing.T) {
	q := mlxQoSFromOptions(map[string]any{
		"zerollama": map[string]any{"qos_priority": 85},
	})
	if q.Class != mlxClassInteractive {
		t.Fatalf("priority 85 class=%s", q.Class)
	}
}

func TestMLXQoSLegacyClass(t *testing.T) {
	q := mlxQoSFromOptions(map[string]any{"mlx_session_class": "primary"})
	if q.Class != mlxClassInteractive || !q.Explicit {
		t.Fatalf("legacy primary -> %s", q.Class)
	}
}

func TestClassifyExplicitOverridesHeuristic(t *testing.T) {
	class, _ := classifyMLXSession("hermes:agent:main:1", mlxScheduleHints{}, map[string]any{
		"zerollama": map[string]any{"qos_class": "background"},
	})
	if class != mlxClassBackground {
		t.Fatalf("explicit qos should win, got %s", class)
	}
}

func TestInjectWithSessionGroup(t *testing.T) {
	qos := mlxQoS{CacheScope: qosCacheScopeShared, SessionGroup: "my-harness"}
	got := injectMLXSessionKey("digest:m", "", mlxClassBackground, qos)
	want := "bg:digest:m:my-harness"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestMergeZerollamaIntoOptions(t *testing.T) {
	opts := mergeZerollamaIntoOptions(nil, map[string]any{
		"zerollama": map[string]any{"qos_class": "auxiliary"},
	})
	z, ok := opts["zerollama"].(map[string]any)
	if !ok || z["qos_class"] != "auxiliary" {
		t.Fatalf("merge failed: %v", opts)
	}
}

func TestClassifyMLXSession(t *testing.T) {
	cases := []struct {
		key   string
		hints mlxScheduleHints
		opts  map[string]any
		want  mlxSessionClass
	}{
		{"hermes:agent:main:discord:dm:1", mlxScheduleHints{Route: "openai", Stream: true}, nil, mlxClassInteractive},
		{"hermes:20260704_170355_abc", mlxScheduleHints{Route: "openai", Stream: false}, nil, mlxClassAuxiliary},
		{"ruby-trivia:bg:question", mlxScheduleHints{Route: "generate", Stream: false}, nil, mlxClassBackground},
		{"", mlxScheduleHints{Route: "generate", Stream: false}, nil, mlxClassBackground},
		{"", mlxScheduleHints{Route: "openai", Stream: false}, nil, mlxClassBackground},
		{"", mlxScheduleHints{Route: "openai", Stream: true}, nil, mlxClassAuxiliary},
		{"", mlxScheduleHints{Route: "generate", Stream: false}, map[string]any{"zerollama": map[string]any{"qos_class": "interactive"}}, mlxClassInteractive},
	}
	for _, tc := range cases {
		got, _ := classifyMLXSession(tc.key, tc.hints, tc.opts)
		if got != tc.want {
			t.Fatalf("classify(%q, %+v) = %s want %s", tc.key, tc.hints, got, tc.want)
		}
	}
}

func TestPrepareMLXSessionMutatesOpts(t *testing.T) {
	m := &Model{Digest: "abc123", Config: model.ConfigV2{ModelFormat: "safetensors"}}
	opts := map[string]any{
		"zerollama": map[string]any{
			"qos_class":     "background",
			"session_group": "ruby-trivia",
			"cache_scope":   "shared",
		},
	}
	ctx := ctxWithMLXScheduleHints(context.Background(), mlxScheduleHints{Route: "generate", Stream: false})
	ctx, meta := prepareMLXSession(ctx, m, opts)
	wantKey := "bg:" + schedulerModelKey(m) + ":ruby-trivia"
	if meta.SessionKey != wantKey {
		t.Fatalf("session key = %q want %q", meta.SessionKey, wantKey)
	}
	if opts["prompt_cache_key"] != wantKey {
		t.Fatalf("opts not injected: %v", opts["prompt_cache_key"])
	}
}

func TestMLXDeferPolicyMatrix(t *testing.T) {
	defer_, policy := mlxDeferPolicy(mlxClassInteractive, "agent:1", mlxClassAuxiliary, "aux:m", 0)
	if defer_ || policy != "interactive_preempt_cooldown" {
		t.Fatalf("interactive preempt: defer=%v policy=%s", defer_, policy)
	}

	defer_, policy = mlxDeferPolicy(mlxClassAuxiliary, "aux:m", mlxClassInteractive, "agent:1", 1)
	if !defer_ || policy != "lower_wait_interactive" {
		t.Fatalf("aux wait interactive inflight: defer=%v policy=%s", defer_, policy)
	}

	defer_, policy = mlxDeferPolicy(mlxClassBackground, "bg:m", mlxClassAuxiliary, "aux:m", 0)
	if !defer_ || policy != "background_wait_auxiliary" {
		t.Fatalf("bg wait aux: defer=%v policy=%s", defer_, policy)
	}
}
