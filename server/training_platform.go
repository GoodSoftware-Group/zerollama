package server

import (
	"encoding/json"
	"errors"
	"runtime"
)

// ErrTrainingQloraUnsupported is returned when QLoRA is requested on a host without CUDA.
var ErrTrainingQloraUnsupported = errors.New(
	"QLoRA requires CUDA (bitsandbytes); use LoRA without use_qlora on Apple Silicon",
)

func trainingPlatformSupported() error {
	return nil
}

// TrainingSubmitUnsupported reports platform errors that should map to HTTP 503.
func TrainingSubmitUnsupported(err error) bool {
	return errors.Is(err, ErrTrainingQloraUnsupported)
}

func checkTrainingQloraPayload(kind string, payload json.RawMessage) error {
	if kind != "train" || len(payload) == 0 {
		return nil
	}
	var p struct {
		UseQlora bool `json:"use_qlora"`
	}
	if err := json.Unmarshal(payload, &p); err != nil {
		return nil
	}
	if p.UseQlora && runtime.GOOS == "darwin" {
		return ErrTrainingQloraUnsupported
	}
	return nil
}

// trainingPlatformLabel returns a short label for logs and health extras.
func trainingPlatformLabel() string {
	if runtime.GOOS == "darwin" {
		return "mps"
	}
	return "cuda"
}
