package latents

import (
	"encoding/json"
	"fmt"
	"os"
)

// FinalStepMeta describes a denoise handoff: sample + noise for the last Euler step.
type FinalStepMeta struct {
	SamplePath string `json:"sample"`
	NoisePath  string `json:"noise"`
	StepIdx    int    `json:"step_idx"`
	Steps      int    `json:"steps"`
	Width      int32  `json:"width"`
	Height     int32  `json:"height"`
}

func WriteFinalStepMeta(path string, meta *FinalStepMeta) error {
	data, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}

func LoadFinalStepMeta(path string) (*FinalStepMeta, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var meta FinalStepMeta
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, err
	}
	if meta.SamplePath == "" || meta.NoisePath == "" {
		return nil, fmt.Errorf("invalid denoise handoff meta")
	}
	return &meta, nil
}
