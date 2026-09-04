package server

import (
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestModelOptionsNumCtxPriority(t *testing.T) {
	tests := []struct {
		name           string
		envContextLen  string // empty means not set (uses 0 sentinel)
		defaultNumCtx  int    // VRAM-based default
		modelNumCtx    int    // 0 means not set in model
		requestNumCtx  int    // 0 means not set in request
		expectedNumCtx int
	}{
		{
			name:           "vram default when nothing else set",
			envContextLen:  "",
			defaultNumCtx:  32768,
			modelNumCtx:    0,
			requestNumCtx:  0,
			expectedNumCtx: 32768,
		},
		{
			name:           "env var overrides vram default",
			envContextLen:  "8192",
			defaultNumCtx:  32768,
			modelNumCtx:    0,
			requestNumCtx:  0,
			expectedNumCtx: 8192,
		},
		{
			name:           "model overrides vram default",
			envContextLen:  "",
			defaultNumCtx:  32768,
			modelNumCtx:    16384,
			requestNumCtx:  0,
			expectedNumCtx: 16384,
		},
		{
			name:           "model overrides env var",
			envContextLen:  "8192",
			defaultNumCtx:  32768,
			modelNumCtx:    16384,
			requestNumCtx:  0,
			expectedNumCtx: 16384,
		},
		{
			name:           "request overrides everything",
			envContextLen:  "8192",
			defaultNumCtx:  32768,
			modelNumCtx:    16384,
			requestNumCtx:  4096,
			expectedNumCtx: 4096,
		},
		{
			name:           "request overrides vram default",
			envContextLen:  "",
			defaultNumCtx:  32768,
			modelNumCtx:    0,
			requestNumCtx:  4096,
			expectedNumCtx: 4096,
		},
		{
			name:           "request overrides model",
			envContextLen:  "",
			defaultNumCtx:  32768,
			modelNumCtx:    16384,
			requestNumCtx:  4096,
			expectedNumCtx: 4096,
		},
		{
			name:           "low vram tier default",
			envContextLen:  "",
			defaultNumCtx:  4096,
			modelNumCtx:    0,
			requestNumCtx:  0,
			expectedNumCtx: 4096,
		},
		{
			name:           "high vram tier default",
			envContextLen:  "",
			defaultNumCtx:  262144,
			modelNumCtx:    0,
			requestNumCtx:  0,
			expectedNumCtx: 262144,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set or clear environment variable
			if tt.envContextLen != "" {
				t.Setenv("OLLAMA_CONTEXT_LENGTH", tt.envContextLen)
			}

			// Create server with VRAM-based default
			s := &Server{
				defaultNumCtx: tt.defaultNumCtx,
			}

			// Create model options (use float64 as FromMap expects JSON-style numbers)
			var modelOpts map[string]any
			if tt.modelNumCtx != 0 {
				modelOpts = map[string]any{"num_ctx": float64(tt.modelNumCtx)}
			}
			model := &Model{
				Options: modelOpts,
			}

			// Create request options (use float64 as FromMap expects JSON-style numbers)
			var requestOpts map[string]any
			if tt.requestNumCtx != 0 {
				requestOpts = map[string]any{"num_ctx": float64(tt.requestNumCtx)}
			}

			opts, err := s.modelOptions(model, requestOpts)
			if err != nil {
				t.Fatalf("modelOptions failed: %v", err)
			}

			if opts.NumCtx != tt.expectedNumCtx {
				t.Errorf("NumCtx = %d, want %d", opts.NumCtx, tt.expectedNumCtx)
			}
		})
	}
}

func TestModelOptionsDraftNumPredictDefault(t *testing.T) {
	s := &Server{defaultNumCtx: 4096}

	tests := []struct {
		name        string
		model       *Model
		requestOpts map[string]any
		want        int
	}{
		{
			name:  "separate draft model keeps default enabled",
			model: &Model{DraftPath: "draft.gguf"},
			want:  4,
		},
		{
			name:  "embedded draft requires explicit parameter",
			model: &Model{},
			want:  0,
		},
		{
			name:  "embedded MTP in GGUF keeps default enabled",
			model: &Model{EmbeddedMTP: true},
			want:  4,
		},
		{
			name:  "model parameter enables embedded draft",
			model: &Model{Options: map[string]any{"draft_num_predict": float64(4)}},
			want:  4,
		},
		{
			name:        "request parameter enables embedded draft",
			model:       &Model{},
			requestOpts: map[string]any{"draft_num_predict": float64(8)},
			want:        8,
		},
		{
			name:        "request can disable separate draft model",
			model:       &Model{DraftPath: "draft.gguf"},
			requestOpts: map[string]any{"draft_num_predict": float64(0)},
			want:        0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts, err := s.modelOptions(tt.model, tt.requestOpts)
			if err != nil {
				t.Fatalf("modelOptions() error = %v", err)
			}
			if opts.DraftNumPredict != tt.want {
				t.Fatalf("DraftNumPredict = %d, want %d", opts.DraftNumPredict, tt.want)
			}
		})
	}
}

func TestModelOptionsGenerationConfig(t *testing.T) {
	s := &Server{defaultNumCtx: 4096}
	m := &Model{
		Config: model.ConfigV2{ModelFormat: "safetensors"},
		GenSampling: map[string]any{
			"temperature": 0.6,
			"top_k":       20,
		},
	}
	opts, err := s.modelOptions(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	if opts.Temperature != 0.6 {
		t.Fatalf("temperature=%v want 0.6 from generation_config", opts.Temperature)
	}
	if opts.TopK != 20 {
		t.Fatalf("top_k=%d want 20", opts.TopK)
	}

	opts, err = s.modelOptions(m, map[string]any{"temperature": 0.2})
	if err != nil {
		t.Fatal(err)
	}
	if opts.Temperature != 0.2 {
		t.Fatalf("request temperature should win, got %v", opts.Temperature)
	}

	m.Options = map[string]any{"temperature": 0.3}
	opts, err = s.modelOptions(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	if opts.Temperature != 0.3 {
		t.Fatalf("PARAMETER should win over generation_config, got %v", opts.Temperature)
	}
}

func TestModelOptionsMLXFamilyTopP(t *testing.T) {
	s := &Server{defaultNumCtx: 4096}
	m := &Model{Config: model.ConfigV2{ModelFormat: "safetensors"}}
	opts, err := s.modelOptions(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	if opts.TopP != 0.95 {
		t.Fatalf("MLX family top_p=%v want 0.95 when generation_config omits it", opts.TopP)
	}
	opts, err = s.modelOptions(m, map[string]any{"top_p": 0.5})
	if err != nil {
		t.Fatal(err)
	}
	if opts.TopP != 0.5 {
		t.Fatalf("request top_p should win, got %v", opts.TopP)
	}
}

func TestModelOptionsDropsHFIdentitySampling(t *testing.T) {
	s := &Server{defaultNumCtx: 4096}
	m := &Model{
		Config:  model.ConfigV2{ModelFormat: "safetensors"},
		Options: map[string]any{"temperature": 1.0, "top_p": 1.0},
	}
	opts, err := s.modelOptions(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	if opts.Temperature != 0.8 {
		t.Fatalf("identity temperature 1.0 should not override default 0.8, got %v", opts.Temperature)
	}
	if opts.TopP != 0.95 {
		t.Fatalf("identity top_p 1.0 should get MLX family 0.95, got %v", opts.TopP)
	}
	opts, err = s.modelOptions(m, map[string]any{"temperature": 1.0})
	if err != nil {
		t.Fatal(err)
	}
	if opts.Temperature != 1.0 {
		t.Fatalf("request temperature 1.0 must still apply, got %v", opts.Temperature)
	}
}

func TestModelOptionsDeepseekV4GreedyDefault(t *testing.T) {
	s := &Server{defaultNumCtx: 4096}
	m := &Model{Config: model.ConfigV2{ModelFormat: "safetensors", ModelFamily: "DeepseekV4ForCausalLM"}}
	opts, err := s.modelOptions(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	if opts.Temperature != 0 {
		t.Fatalf("DSv4 default temperature=%v want 0 (mlx-lm greedy)", opts.Temperature)
	}
	opts, err = s.modelOptions(m, map[string]any{"temperature": 0.8})
	if err != nil {
		t.Fatal(err)
	}
	if opts.Temperature != 0.8 {
		t.Fatalf("request temperature should win, got %v", opts.Temperature)
	}
}
