package server

import (
	"testing"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

func TestSuggestedManifestNumCtxCapsLargeTrain(t *testing.T) {
	kv := ggml.KV{
		"general.architecture": "llama",
		"llama.context_length": uint32(262144),
	}
	if got := suggestedManifestNumCtx(kv); got != defaultManifestNumCtxCap {
		t.Fatalf("got %d want %d", got, defaultManifestNumCtxCap)
	}
}

func TestSuggestedManifestNumCtxSmallTrain(t *testing.T) {
	kv := ggml.KV{
		"general.architecture": "llama",
		"llama.context_length": uint32(4096),
	}
	if got := suggestedManifestNumCtx(kv); got != 4096 {
		t.Fatalf("got %d", got)
	}
}

func TestGuessConfigFromGGUF(t *testing.T) {
	t.Setenv("ZEROLLAMA_DISABLE_GGUF_GUESS", "0")
	cfg := &model.ConfigV2{}
	GuessConfigFromGGUF(cfg, ggml.KV{
		"general.architecture":   "llama",
		"llama.context_length":   uint32(8192),
		"llama.embedding_length": uint32(4096),
		"general.file_type":      uint32(15),
	})
	if cfg.ModelFamily != "llama" {
		t.Fatalf("family=%q", cfg.ModelFamily)
	}
	if cfg.ContextLen != 8192 {
		t.Fatalf("context_len=%d", cfg.ContextLen)
	}
	if cfg.EmbedLen != 4096 {
		t.Fatalf("embed_len=%d", cfg.EmbedLen)
	}
	if cfg.FileType == "" {
		t.Fatalf("file_type not guessed")
	}
}

func TestGuessParametersFromGGUFFile(t *testing.T) {
	t.Setenv("ZEROLLAMA_DISABLE_GGUF_GUESS", "0")
	path, _ := createBinFile(t, ggml.KV{
		"general.architecture":  "qwen35",
		"qwen35.context_length": uint32(4096),
	}, nil)

	f, err := llm.LoadModelMetadata(path)
	if err != nil {
		t.Fatal(err)
	}

	params := map[string]any{}
	GuessParametersFromGGUF(params, f)
	if n, ok := params["num_ctx"].(int); !ok || n != 4096 {
		t.Fatalf("num_ctx=%v", params["num_ctx"])
	}
	stop, ok := params["stop"].([]string)
	if !ok || len(stop) != 1 || stop[0] != "<|im_end|>" {
		t.Fatalf("stop=%v", params["stop"])
	}
}

func TestGuessParametersCapsManifestNumCtx(t *testing.T) {
	t.Setenv("ZEROLLAMA_DISABLE_GGUF_GUESS", "0")
	path, _ := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
		"llama.context_length": uint32(131072),
	}, nil)

	f, err := llm.LoadModelMetadata(path)
	if err != nil {
		t.Fatal(err)
	}

	params := map[string]any{}
	GuessParametersFromGGUF(params, f)
	if n, ok := params["num_ctx"].(int); !ok || n != defaultManifestNumCtxCap {
		t.Fatalf("num_ctx=%v want %d", params["num_ctx"], defaultManifestNumCtxCap)
	}
}

func TestGGUFGuessDisabled(t *testing.T) {
	t.Setenv("ZEROLLAMA_DISABLE_GGUF_GUESS", "1")
	cfg := &model.ConfigV2{}
	GuessConfigFromGGUF(cfg, ggml.KV{
		"general.architecture": "llama",
		"llama.context_length": uint32(8192),
	})
	if cfg.ModelFamily != "" || cfg.ContextLen != 0 {
		t.Fatalf("expected no guess when disabled: %+v", cfg)
	}
}

func TestGuessFromBaseLayersSkipsProjector(t *testing.T) {
	t.Setenv("ZEROLLAMA_DISABLE_GGUF_GUESS", "0")
	llmPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
		"llama.context_length": uint32(4096),
	}, nil)
	projPath, _ := createBinFile(t, ggml.KV{
		"general.type":         "projector",
		"general.architecture": "clip",
	}, nil)

	llmGGML, err := llm.LoadModelMetadata(llmPath)
	if err != nil {
		t.Fatal(err)
	}
	projGGML, err := llm.LoadModelMetadata(projPath)
	if err != nil {
		t.Fatal(err)
	}

	cfg := &model.ConfigV2{}
	params := map[string]any{}
	guessFromBaseLayers(cfg, params, []*layerGGML{
		{GGML: projGGML},
		{GGML: llmGGML},
	})
	if cfg.ModelFamily != "llama" {
		t.Fatalf("family=%q", cfg.ModelFamily)
	}
	if n, ok := params["num_ctx"].(int); !ok || n != 4096 {
		t.Fatalf("num_ctx=%v", params["num_ctx"])
	}
}
