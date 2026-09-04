package server

import (
	"encoding/json"
	"strings"
)

// TensorNameHasInCheckpointMTP is true for Qwen 3.5/3.6 in-weight mtp.* tensors
// (not a Gemma drafter/ sidecar). Used so /v1/models supports_mtp matches load.
func TensorNameHasInCheckpointMTP(name string) bool {
	n := strings.ToLower(strings.TrimSpace(name))
	if n == "" {
		return false
	}
	return strings.Contains(n, "mtp.")
}

// ConfigMapHasInCheckpointMTP detects HF config.json nextn / mtp fields.
func ConfigMapHasInCheckpointMTP(raw map[string]any) bool {
	if raw == nil {
		return false
	}
	if configIntPositive(raw, "num_nextn_predict_layers") {
		return true
	}
	if nested, ok := raw["text_config"].(map[string]any); ok && configIntPositive(nested, "num_nextn_predict_layers") {
		return true
	}
	if v, ok := raw["mtp"]; ok && v != nil {
		return true
	}
	return false
}

func configIntPositive(raw map[string]any, key string) bool {
	v, ok := raw[key]
	if !ok || v == nil {
		return false
	}
	switch n := v.(type) {
	case int:
		return n > 0
	case int32:
		return n > 0
	case int64:
		return n > 0
	case float64:
		return n > 0
	case json.Number:
		i, err := n.Int64()
		return err == nil && i > 0
	default:
		return false
	}
}
