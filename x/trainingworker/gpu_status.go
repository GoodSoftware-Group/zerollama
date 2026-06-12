package trainingworker

import (
	"encoding/json"
	"strings"
)

// TrainingGPUStatus summarizes whether training is using or queued for the GPU.
type TrainingGPUStatus struct {
	TrainingActive bool
	QueueRunning   int
	QueuePending   int
	ModelLoaded    string
	CudaAvailable  bool
	MpsAvailable   bool
}

// trainingOccupiesGPU reports whether inference should yield to training work.
func trainingOccupiesGPU(extrasJSON string) (TrainingGPUStatus, bool) {
	if extrasJSON == "" {
		return TrainingGPUStatus{}, false
	}
	var extras struct {
		TrainingActive bool `json:"training_active"`
		CudaAvailable  bool `json:"cuda_available"`
		MpsAvailable   bool `json:"mps_available"`
		ModelLoaded    any  `json:"model_loaded"`
		Queue          struct {
			Running int `json:"running"`
			Pending int `json:"pending"`
		} `json:"queue"`
	}
	if err := json.Unmarshal([]byte(extrasJSON), &extras); err != nil {
		return TrainingGPUStatus{}, false
	}
	modelName := trainingModelName(extras.ModelLoaded)
	st := TrainingGPUStatus{
		TrainingActive: extras.TrainingActive,
		QueueRunning:   extras.Queue.Running,
		QueuePending:   extras.Queue.Pending,
		ModelLoaded:    modelName,
		CudaAvailable:  extras.CudaAvailable,
		MpsAvailable:   extras.MpsAvailable,
	}
	// Pending jobs do not block inference until one is running (inference-first default).
	busy := st.TrainingActive ||
		st.QueueRunning > 0 ||
		trainingModelHoldsGPU(st.CudaAvailable, st.MpsAvailable, modelName)
	return st, busy
}

func trainingModelName(v any) string {
	switch x := v.(type) {
	case string:
		return strings.TrimSpace(x)
	default:
		return ""
	}
}

func trainingModelHoldsGPU(cudaAvailable, mpsAvailable bool, modelName string) bool {
	return (cudaAvailable || mpsAvailable) && modelName != ""
}
