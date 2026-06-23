package model

import (
	"encoding/json"
	"strings"
)

type quantEntry struct {
	Bits      int    `json:"bits"`
	GroupSize int    `json:"group_size"`
	Mode      string `json:"mode"`
}

// QuantConfigFields holds mutable quantization settings parsed from config.json.
type QuantConfigFields struct {
	QuantGroupSize *int
	QuantBits      *int
	QuantMode      *string
	TensorQuant    map[string]*TensorQuantInfo
}

func quantTypeForEntry(entry quantEntry, fallbackBits, fallbackGroup int, fallbackMode string) string {
	bits := entry.Bits
	if bits == 0 {
		bits = fallbackBits
	}
	mode := entry.Mode
	if mode == "" {
		mode = fallbackMode
	}
	switch strings.ToLower(mode) {
	case "affine", "":
		switch bits {
		case 3:
			return "int3"
		case 4:
			return "int4"
		case 6:
			return "int6"
		case 8:
			return "int8"
		}
	case "nvfp4":
		return "nvfp4"
	case "mxfp4":
		return "mxfp4"
	case "mxfp8":
		return "mxfp8"
	}
	return ""
}

// ApplyQuantizationFromConfig merges HuggingFace quantization_config into model
// quant defaults and per-tensor metadata. MLX imports often omit quant metadata
// from tensor blob headers; config.json is the source of truth.
func ApplyQuantizationFromConfig(configData []byte, cfg *QuantConfigFields) {
	if cfg == nil {
		return
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(configData, &raw); err != nil {
		return
	}
	qraw, ok := raw["quantization_config"]
	if !ok {
		qraw, ok = raw["quantization"]
	}
	if !ok {
		return
	}

	var qmap map[string]json.RawMessage
	if err := json.Unmarshal(qraw, &qmap); err != nil {
		return
	}

	var global quantEntry
	if err := json.Unmarshal(qraw, &global); err == nil {
		if global.GroupSize > 0 && cfg.QuantGroupSize != nil {
			*cfg.QuantGroupSize = global.GroupSize
		}
		if global.Bits > 0 && cfg.QuantBits != nil {
			*cfg.QuantBits = global.Bits
		}
		if global.Mode != "" && cfg.QuantMode != nil {
			*cfg.QuantMode = global.Mode
		}
	}

	fallbackBits, fallbackGroup, fallbackMode := 0, 0, ""
	if cfg.QuantBits != nil {
		fallbackBits = *cfg.QuantBits
	}
	if cfg.QuantGroupSize != nil {
		fallbackGroup = *cfg.QuantGroupSize
	}
	if cfg.QuantMode != nil {
		fallbackMode = *cfg.QuantMode
	}
	if fallbackMode == "" && fallbackBits > 0 && cfg.QuantMode != nil {
		*cfg.QuantMode = "affine"
		fallbackMode = "affine"
	}

	if cfg.TensorQuant == nil {
		cfg.TensorQuant = make(map[string]*TensorQuantInfo)
	}

	for key, val := range qmap {
		switch key {
		case "bits", "group_size", "mode", "quant_method", "weight_block_size":
			continue
		}
		var entry quantEntry
		if err := json.Unmarshal(val, &entry); err != nil {
			continue
		}
		quantType := quantTypeForEntry(entry, fallbackBits, fallbackGroup, fallbackMode)
		if quantType == "" {
			continue
		}
		groupSize := entry.GroupSize
		if groupSize == 0 {
			groupSize = fallbackGroup
		}
		cfg.TensorQuant[key+".weight"] = &TensorQuantInfo{
			QuantType: quantType,
			GroupSize: groupSize,
		}
	}
}
