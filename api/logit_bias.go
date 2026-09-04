package api

import (
	"fmt"
	"strconv"
)

const maxLogitBiasEntries = 256

// ParseLogitBias accepts OpenAI logit_bias maps (token-id string keys) or
// an already-parsed map[int32]float32.
func ParseLogitBias(v any) (map[int32]float32, error) {
	if v == nil {
		return nil, nil
	}
	switch m := v.(type) {
	case map[int32]float32:
		return copyLogitBias(m)
	case map[string]float32:
		out := make(map[int32]float32, len(m))
		for k, bias := range m {
			id, err := parseLogitBiasTokenID(k)
			if err != nil {
				return nil, err
			}
			out[id] = bias
		}
		return copyLogitBias(out)
	case map[string]float64:
		out := make(map[int32]float32, len(m))
		for k, bias := range m {
			id, err := parseLogitBiasTokenID(k)
			if err != nil {
				return nil, err
			}
			out[id] = float32(bias)
		}
		return copyLogitBias(out)
	case map[string]any:
		out := make(map[int32]float32, len(m))
		for k, raw := range m {
			id, err := parseLogitBiasTokenID(k)
			if err != nil {
				return nil, err
			}
			f, ok := coerceFloat64(raw)
			if !ok {
				return nil, fmt.Errorf("logit_bias[%s] must be a number", k)
			}
			out[id] = float32(f)
		}
		return copyLogitBias(out)
	default:
		return nil, fmt.Errorf("logit_bias must be an object of token id to bias")
	}
}

func parseLogitBiasTokenID(k string) (int32, error) {
	n, err := strconv.ParseInt(k, 10, 32)
	if err != nil || n < 0 {
		return 0, fmt.Errorf("logit_bias key %q must be a non-negative token id", k)
	}
	return int32(n), nil
}

func copyLogitBias(m map[int32]float32) (map[int32]float32, error) {
	if len(m) == 0 {
		return nil, nil
	}
	if len(m) > maxLogitBiasEntries {
		return nil, fmt.Errorf("logit_bias has %d entries; max is %d", len(m), maxLogitBiasEntries)
	}
	out := make(map[int32]float32, len(m))
	for id, bias := range m {
		if id < 0 {
			return nil, fmt.Errorf("logit_bias token id %d is negative", id)
		}
		out[id] = bias
	}
	return out, nil
}
