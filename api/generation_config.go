package api

import (
	"encoding/json"
	"math"
)

// SamplingMapFromGenerationConfig copies Hugging Face generation_config.json
// sampling fields into Ollama option keys (mlx-serve: body > PARAMETER > this > server defaults).
func SamplingMapFromGenerationConfig(data []byte) map[string]any {
	if len(data) == 0 {
		return nil
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil
	}
	return SamplingMapFromGenerationConfigMap(raw)
}

// SamplingMapFromGenerationConfigMap is the map form of SamplingMapFromGenerationConfig.
func SamplingMapFromGenerationConfigMap(raw map[string]any) map[string]any {
	if len(raw) == 0 {
		return nil
	}
	out := make(map[string]any)
	if v, ok := jsonFloat(raw["temperature"]); ok && v != 1 {
		// Why skip 1.0: HF generation_config uses temperature 1 + top_p 1 as
		// identity ("sample freely"). Baking that as Ollama PARAMETER turns
		// DeepSeek-V4 Flash into Chinese number-salad on "hello".
		out["temperature"] = v
	}
	if v, ok := jsonFloat(raw["top_p"]); ok && v > 0 && v != 1 {
		out["top_p"] = v
	}
	if n, ok := jsonInt(raw["top_k"]); ok && n > 0 {
		out["top_k"] = n
	}
	if v, ok := jsonFloat(raw["typical_p"]); ok && v > 0 {
		out["typical_p"] = v
	}
	if v, ok := jsonFloat(raw["repetition_penalty"]); ok && v > 0 {
		out["repeat_penalty"] = v
	}
	if v, ok := jsonFloat(raw["presence_penalty"]); ok {
		out["presence_penalty"] = v
	}
	if v, ok := jsonFloat(raw["frequency_penalty"]); ok {
		out["frequency_penalty"] = v
	}
	if ds, ok := raw["do_sample"].(bool); ok && !ds {
		out["temperature"] = 0.0
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// WithoutHFIdentitySampling drops temperature 1 and top_p 1. Hugging Face
// generation_config.json uses those as "unset / sample freely", not a recipe.
func WithoutHFIdentitySampling(opts map[string]any) map[string]any {
	if len(opts) == 0 {
		return opts
	}
	out := make(map[string]any, len(opts))
	for k, v := range opts {
		out[k] = v
	}
	if v, ok := jsonFloat(out["temperature"]); ok && v == 1 {
		delete(out, "temperature")
	}
	if v, ok := jsonFloat(out["top_p"]); ok && v == 1 {
		delete(out, "top_p")
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func jsonFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		if math.IsNaN(n) || math.IsInf(n, 0) {
			return 0, false
		}
		return n, true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	default:
		return 0, false
	}
}

func jsonInt(v any) (int, bool) {
	f, ok := jsonFloat(v)
	if !ok {
		return 0, false
	}
	n := int(f)
	if float64(n) != f {
		return 0, false
	}
	return n, true
}
