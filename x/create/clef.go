package create

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"github.com/ollama/ollama/x/safetensors"
)

// prepareClefInventory merges Clef joint_head weights into the inventory and
// normalizes the output architecture to ClefForDecision without editing source files.
func prepareClefInventory(modelDir string, cfg sourceModelConfig, rawConfig json.RawMessage) (sourceModelConfig, json.RawMessage, map[string]struct{}, error) {
	if cfg.Architecture() != "Qwen3_5ForConditionalGeneration" {
		return cfg, rawConfig, nil, fmt.Errorf("unsupported Clef backbone %q", cfg.Architecture())
	}
	var backbone struct {
		TextConfig struct {
			HiddenSize int `json:"hidden_size"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(rawConfig, &backbone); err != nil {
		return cfg, rawConfig, nil, err
	}
	var headCfg struct {
		HiddenSize    int `json:"hidden_size"`
		Width         int `json:"width"`
		RoutingLayers int `json:"routing_layers"`
		Layers        int `json:"layers"`
		Heads         int `json:"heads"`
		Feedforward   int `json:"feedforward"`
	}
	data, err := os.ReadFile(filepath.Join(modelDir, "joint_head_config.json"))
	if err != nil {
		return cfg, rawConfig, nil, fmt.Errorf("read Clef joint head config: %w", err)
	}
	if err := json.Unmarshal(data, &headCfg); err != nil {
		return cfg, rawConfig, nil, fmt.Errorf("parse Clef joint head config: %w", err)
	}
	if headCfg.HiddenSize <= 0 || headCfg.HiddenSize != backbone.TextConfig.HiddenSize || headCfg.Width <= 0 || headCfg.Heads <= 0 || headCfg.Width%headCfg.Heads != 0 || headCfg.Feedforward <= 0 || headCfg.Layers < 1 || headCfg.Layers > 16 || headCfg.RoutingLayers < 1 || headCfg.RoutingLayers > 16 {
		return cfg, rawConfig, nil, fmt.Errorf("invalid Clef joint head configuration for backbone")
	}
	head, err := safetensors.OpenForExtraction(filepath.Join(modelDir, "joint_head.safetensors"))
	if err != nil {
		return cfg, rawConfig, nil, fmt.Errorf("read Clef joint head: %w", err)
	}
	defer head.Close()
	for name, shape := range map[string][]int32{
		"memory_projection.weight": {int32(headCfg.Width), int32(headCfg.HiddenSize)},
		"type_embedding.weight":    {3, int32(headCfg.Width)},
	} {
		tensor, err := head.GetTensor(name)
		if err != nil || !slices.Equal(tensor.Shape, shape) {
			return cfg, rawConfig, nil, fmt.Errorf("Clef joint head requires %s with shape %v", name, shape)
		}
	}
	extraFiles := map[string]struct{}{"joint_head.safetensors": {}}
	for _, name := range head.ListTensors() {
		if _, err := head.GetTensor(name); err != nil {
			return cfg, rawConfig, nil, fmt.Errorf("read Clef head tensor %s: %w", name, err)
		}
	}

	var output map[string]json.RawMessage
	if err := json.Unmarshal(rawConfig, &output); err != nil {
		return cfg, rawConfig, nil, err
	}
	output["architectures"] = json.RawMessage(`["ClefForDecision"]`)
	rawConfig, err = json.Marshal(output)
	if err != nil {
		return cfg, rawConfig, nil, err
	}
	cfg.Architectures = []string{"ClefForDecision"}
	return cfg, rawConfig, extraFiles, nil
}

func clefWeightFiles(dir string, files []string) ([]string, error) {
	info, err := os.Stat(filepath.Join(dir, "joint_head.safetensors"))
	if os.IsNotExist(err) {
		return files, nil
	}
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("Clef joint_head.safetensors must be a regular file")
	}
	if !slices.Contains(files, "joint_head.safetensors") {
		files = append(files, "joint_head.safetensors")
		slices.Sort(files)
	}
	return files, nil
}
