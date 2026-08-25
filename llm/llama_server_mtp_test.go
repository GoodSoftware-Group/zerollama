package llm

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
)

func TestDisableDraftMTPForArchitecture(t *testing.T) {
	if !DisableDraftMTPForArchitecture("qwen35") || !DisableDraftMTPForArchitecture("qwen35moe") {
		t.Fatal("expected qwen35 family to disable draft-mtp")
	}
	if DisableDraftMTPForArchitecture("llama") {
		t.Fatal("llama should keep draft-mtp eligible")
	}
}

func TestAppendSpeculativeArgsNone(t *testing.T) {
	got := appendSpeculativeArgs([]string{"base"}, "", LlamaServerConfig{SpecType: "none", EnableMTP: true}, api.Options{Runner: api.Runner{DraftNumPredict: 4}})
	if !slices.Equal(got, []string{"base"}) {
		t.Fatalf("got %v", got)
	}
}

func TestAppendSpeculativeArgsNgram(t *testing.T) {
	got := appendSpeculativeArgs([]string{"base"}, "", LlamaServerConfig{SpecType: "ngram-simple"}, api.Options{})
	want := []string{
		"base", "--spec-type", "ngram-simple",
		"--spec-ngram-simple-size-n", "12",
		"--spec-ngram-simple-size-m", "48",
		"--spec-ngram-simple-min-hits", "1",
	}
	if !slices.Equal(got, want) {
		t.Fatalf("appendSpeculativeArgs = %v, want %v", got, want)
	}
}

func TestAppendSpeculativeArgsDFlash(t *testing.T) {
	t.Cleanup(resetLlamaServerHelpCache)
	// No binary → prefer ggml-org draft-dflash name.
	for _, specType := range []string{"dflash", "draft-dflash"} {
		got := appendSpeculativeArgs([]string{"base"}, "", LlamaServerConfig{
			SpecType:       specType,
			DraftModelPath: "drafter.gguf",
		}, api.Options{Runner: api.Runner{DraftNumPredict: 6}})
		want := []string{
			"base", "--spec-type", "draft-dflash",
			"--spec-draft-model", "drafter.gguf",
			"--spec-draft-n-max", "6",
		}
		if !slices.Equal(got, want) {
			t.Fatalf("appendSpeculativeArgs(%q) = %v, want %v", specType, got, want)
		}
	}

	const bin = "/tmp/llama-server-dflash-legacy"
	llamaServerHelpCache.Store(bin, "--spec-type none,draft-eagle3,draft-mtp,ngram-simple,dflash\n")
	got := appendSpeculativeArgs([]string{"base"}, bin, LlamaServerConfig{
		SpecType:       "dflash",
		DraftModelPath: "drafter.gguf",
	}, api.Options{Runner: api.Runner{DraftNumPredict: 6}})
	want := []string{
		"base", "--spec-type", "dflash",
		"--spec-draft-model", "drafter.gguf",
		"--spec-draft-n-max", "6",
	}
	if !slices.Equal(got, want) {
		t.Fatalf("legacy dflash token = %v, want %v", got, want)
	}
}

func TestAppendSpeculativeArgsEagle3(t *testing.T) {
	got := appendSpeculativeArgs([]string{"base"}, "", LlamaServerConfig{
		SpecType:       "draft-eagle3",
		DraftModelPath: "eagle3.gguf",
	}, api.Options{Runner: api.Runner{DraftNumPredict: 4}})
	want := []string{
		"base", "--spec-type", "draft-eagle3",
		"--spec-draft-model", "eagle3.gguf",
		"--spec-draft-n-max", "4",
	}
	if !slices.Equal(got, want) {
		t.Fatalf("appendSpeculativeArgs = %v, want %v", got, want)
	}
}

func TestAppendMTPDraftArgs(t *testing.T) {
	tests := []struct {
		name   string
		config LlamaServerConfig
		opts   api.Options
		want   []string
	}{
		{
			name: "no draft model leaves speculative decoding disabled",
			opts: api.Options{Runner: api.Runner{DraftNumPredict: 4}},
			want: []string{"base"},
		},
		{
			name:   "embedded draft uses configured draft depth",
			config: LlamaServerConfig{EnableMTP: true},
			opts:   api.Options{Runner: api.Runner{DraftNumPredict: 4}},
			want:   []string{"base", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4"},
		},
		{
			name:   "separate draft model uses configured draft depth",
			config: LlamaServerConfig{DraftModelPath: "draft.gguf"},
			opts:   api.Options{Runner: api.Runner{DraftNumPredict: 8}},
			want:   []string{"base", "--spec-type", "draft-mtp", "--spec-draft-n-max", "8", "--spec-draft-model", "draft.gguf"},
		},
		{
			name:   "zero draft depth disables speculative decoding",
			config: LlamaServerConfig{EnableMTP: true, DraftModelPath: "draft.gguf"},
			opts:   api.Options{Runner: api.Runner{DraftNumPredict: 0}},
			want:   []string{"base"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := appendMTPDraftArgs([]string{"base"}, "", tt.config, tt.opts)
			if !slices.Equal(got, tt.want) {
				t.Fatalf("appendMTPDraftArgs = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHasLegacyQwenMTPDraft(t *testing.T) {
	tests := []struct {
		name    string
		arch    string
		tensors []*ggml.Tensor
		want    bool
	}{
		{
			name:    "qwen35 legacy mtp marker",
			arch:    "qwen35",
			tensors: []*ggml.Tensor{{Name: "mtp.fc.weight"}},
			want:    true,
		},
		{
			name:    "qwen35moe legacy mtp marker",
			arch:    "qwen35moe",
			tensors: []*ggml.Tensor{{Name: "mtp.layers.0.attn_q.weight"}},
			want:    true,
		},
		{
			name:    "qwen35 without legacy mtp marker",
			arch:    "qwen35",
			tensors: nil,
			want:    false,
		},
		{
			name:    "other arch with mtp prefix",
			arch:    "qwen3next",
			tensors: []*ggml.Tensor{{Name: "mtp.fc.weight"}},
			want:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := hasLegacyQwenMTPDraft(tt.arch, tt.tensors); got != tt.want {
				t.Fatalf("hasLegacyQwenMTPDraft() = %v, want %v", got, tt.want)
			}
		})
	}
}
