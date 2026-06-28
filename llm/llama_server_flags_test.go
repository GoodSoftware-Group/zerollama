package llm

import (
	"os"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestAppendSpecDraftBackendSamplingArgSkipsWhenUnsupported(t *testing.T) {
	t.Cleanup(resetLlamaServerHelpCache)
	const bin = "/tmp/llama-server-old-fork"
	llamaServerHelpCache.Store(bin, "--spec-type draft-mtp\n--spec-draft-n-max 4\n")

	got := appendSpeculativeArgs([]string{"base"}, bin, LlamaServerConfig{
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

func TestAppendSpecDraftBackendSamplingArgSkipsWhenServerBinEmpty(t *testing.T) {
	t.Cleanup(resetLlamaServerHelpCache)
	got := appendMTPDraftArgs([]string{"base"}, "", LlamaServerConfig{EnableMTP: true}, api.Options{
		Runner: api.Runner{DraftNumPredict: 4},
	})
	want := []string{"base", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4"}
	if !slices.Equal(got, want) {
		t.Fatalf("appendMTPDraftArgs = %v, want %v", got, want)
	}
}

func TestAppendSpecDraftBackendSamplingArgSkipsElizaFork(t *testing.T) {
	const bin = "/Users/user1/Sites/inference/eliza-llama.cpp/build/bin/llama-server"
	if _, err := os.Stat(bin); err != nil {
		t.Skip("eliza llama-server not installed")
	}
	t.Cleanup(resetLlamaServerHelpCache)

	got := appendMTPDraftArgs([]string{"base"}, bin, LlamaServerConfig{EnableMTP: true}, api.Options{
		Runner: api.Runner{DraftNumPredict: 4},
	})
	for _, arg := range got {
		if arg == specDraftBackendSamplingFlag {
			t.Fatalf("eliza fork must not receive %s, got %v", specDraftBackendSamplingFlag, got)
		}
	}
}

func TestAppendSpecDraftBackendSamplingArgIncludesWhenSupported(t *testing.T) {
	t.Cleanup(resetLlamaServerHelpCache)
	const bin = "/tmp/llama-server-new"
	llamaServerHelpCache.Store(bin, "--spec-draft-backend-sampling, --no-spec-draft-backend-sampling\n")

	got := appendMTPDraftArgs([]string{"base"}, bin, LlamaServerConfig{EnableMTP: true}, api.Options{
		Runner: api.Runner{DraftNumPredict: 4},
	})
	want := []string{"base", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "--spec-draft-backend-sampling"}
	if !slices.Equal(got, want) {
		t.Fatalf("appendMTPDraftArgs = %v, want %v", got, want)
	}
}
