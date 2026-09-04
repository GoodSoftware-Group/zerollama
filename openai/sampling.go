package openai

import (
	"encoding/json"

	"github.com/ollama/ollama/api"
)

// samplingOpts is the local-LLM / mlx-serve sampling surface shared by
// /v1/chat/completions, /v1/completions, and /v1/responses.
type samplingOpts struct {
	Temperature         *float64
	TopP                *float64
	MinP                *float64
	TypicalP            *float64
	FrequencyPenalty    *float64
	PresencePenalty     *float64
	RepetitionPenalty   *float64
	RepeatPenalty       *float64
	TopK                *int
	Seed                *int
	MaxTokens           *int
	MaxCompletionTokens *int
}

func (s samplingOpts) apply(options map[string]any) {
	if s.MaxTokens != nil {
		options["num_predict"] = *s.MaxTokens
	} else if s.MaxCompletionTokens != nil {
		options["num_predict"] = *s.MaxCompletionTokens
	}
	putF64(options, "temperature", s.Temperature)
	putF64(options, "top_p", s.TopP)
	putF64(options, "min_p", s.MinP)
	putF64(options, "typical_p", s.TypicalP)
	putF64(options, "frequency_penalty", s.FrequencyPenalty)
	putF64(options, "presence_penalty", s.PresencePenalty)
	if s.RepetitionPenalty != nil {
		options["repeat_penalty"] = *s.RepetitionPenalty
	} else if s.RepeatPenalty != nil {
		options["repeat_penalty"] = *s.RepeatPenalty
	}
	putInt(options, "top_k", s.TopK)
	putInt(options, "seed", s.Seed)
}

func putF64(options map[string]any, key string, v *float64) {
	if v != nil {
		options[key] = *v
	}
}

func putInt(options map[string]any, key string, v *int) {
	if v != nil {
		options[key] = *v
	}
}

func putLogitBias(options map[string]any, raw map[string]float64) error {
	if len(raw) == 0 {
		return nil
	}
	parsed, err := api.ParseLogitBias(raw)
	if err != nil {
		return err
	}
	if len(parsed) > 0 {
		options["logit_bias"] = parsed
	}
	return nil
}

func f32as64(v *float32) *float64 {
	if v == nil {
		return nil
	}
	x := float64(*v)
	return &x
}

func overlayF64(dst **float64, raw json.RawMessage) {
	if dst == nil || *dst != nil || len(raw) == 0 {
		return
	}
	var v float64
	if json.Unmarshal(raw, &v) != nil {
		return
	}
	*dst = &v
}

func overlayInt(dst **int, raw json.RawMessage) {
	if dst == nil || *dst != nil || len(raw) == 0 {
		return
	}
	var v int
	if json.Unmarshal(raw, &v) != nil {
		return
	}
	*dst = &v
}

func overlayF32(dst **float32, raw json.RawMessage) {
	if dst == nil || *dst != nil || len(raw) == 0 {
		return
	}
	var v float32
	if json.Unmarshal(raw, &v) != nil {
		return
	}
	*dst = &v
}

func overlaySamplingFromRaw(dst *samplingOverlay, raw map[string]json.RawMessage) {
	if dst == nil || len(raw) == 0 {
		return
	}
	overlayF64(dst.Temperature, raw["temperature"])
	overlayF64(dst.TopP, raw["top_p"])
	overlayF64(dst.MinP, raw["min_p"])
	overlayF64(dst.TypicalP, raw["typical_p"])
	overlayF64(dst.FrequencyPenalty, raw["frequency_penalty"])
	overlayF64(dst.PresencePenalty, raw["presence_penalty"])
	overlayF64(dst.RepetitionPenalty, raw["repetition_penalty"])
	overlayF64(dst.RepeatPenalty, raw["repeat_penalty"])
	overlayInt(dst.TopK, raw["top_k"])
	overlayInt(dst.Seed, raw["seed"])
	overlayInt(dst.MaxTokens, raw["max_tokens"])
	overlayInt(dst.MaxCompletionTokens, raw["max_completion_tokens"])
	overlayBool(dst.EnablePLD, raw["enable_pld"])
	overlayBool(dst.EnableMTP, raw["enable_mtp"])
	overlayBool(dst.EnableDrafter, raw["enable_drafter"])
}

func overlayBool(dst **bool, raw json.RawMessage) {
	if dst == nil || *dst != nil || len(raw) == 0 {
		return
	}
	var v bool
	if json.Unmarshal(raw, &v) != nil {
		return
	}
	*dst = &v
}

type samplingOverlay struct {
	Temperature         **float64
	TopP                **float64
	MinP                **float64
	TypicalP            **float64
	FrequencyPenalty    **float64
	PresencePenalty     **float64
	RepetitionPenalty   **float64
	RepeatPenalty       **float64
	TopK                **int
	Seed                **int
	MaxTokens           **int
	MaxCompletionTokens **int
	EnablePLD           **bool
	EnableMTP           **bool
	EnableDrafter       **bool
}

func overlayLogitBias(dst *map[string]float64, raw json.RawMessage) {
	if dst == nil || *dst != nil || len(raw) == 0 {
		return
	}
	var v map[string]float64
	if json.Unmarshal(raw, &v) != nil || len(v) == 0 {
		return
	}
	*dst = v
}
