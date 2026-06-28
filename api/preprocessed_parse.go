package api

import (
	"encoding/json"
	"fmt"
)

// FlattenJSONFloats coerces HF/SGLang tensor JSON (1D–3D float arrays) into a flat slice.
func FlattenJSONFloats(raw json.RawMessage) ([]float32, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("empty float tensor")
	}
	// Try nested shapes before 1D: json into []float32 mis-parses [[...]] as [0,0,...].
	var mat3 [][][]float32
	if err := json.Unmarshal(raw, &mat3); err == nil && len(mat3) > 0 {
		var flat []float32
		for _, batch := range mat3 {
			for _, row := range batch {
				flat = append(flat, row...)
			}
		}
		if len(flat) == 0 {
			return nil, fmt.Errorf("empty float tensor")
		}
		return flat, nil
	}
	var mat2 [][]float32
	if err := json.Unmarshal(raw, &mat2); err == nil && len(mat2) > 0 {
		var flat []float32
		for _, row := range mat2 {
			flat = append(flat, row...)
		}
		if len(flat) == 0 {
			return nil, fmt.Errorf("empty float tensor")
		}
		return flat, nil
	}
	var flat []float32
	if err := json.Unmarshal(raw, &flat); err == nil {
		if len(flat) == 0 {
			return nil, fmt.Errorf("empty float tensor")
		}
		return flat, nil
	}
	return nil, fmt.Errorf("pixel_values must be a JSON float array")
}

// ParseGridTHW accepts SGLang image_grid_thw ([T,H,W] or [[T,H,W]]) or grid_thw alias.
func ParseGridTHW(imageGridTHW, gridTHW json.RawMessage) ([]int, error) {
	if thw, err := parseGridTHWMessage(imageGridTHW); err == nil {
		return thw, nil
	}
	return parseGridTHWMessage(gridTHW)
}

func parseGridTHWMessage(raw json.RawMessage) ([]int, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("missing grid")
	}
	var single []int
	if err := json.Unmarshal(raw, &single); err == nil && len(single) == 3 {
		return single, nil
	}
	var batch [][]int
	if err := json.Unmarshal(raw, &batch); err == nil && len(batch) > 0 && len(batch[0]) == 3 {
		return batch[0], nil
	}
	return nil, fmt.Errorf("grid_thw must be [T,H,W] or [[T,H,W]]")
}
