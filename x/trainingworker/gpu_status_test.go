package trainingworker

import "testing"

func TestTrainingOccupiesGPU(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		json string
		want bool
	}{
		{"idle", `{"training_active":false,"cuda_available":true,"model_loaded":null,"queue":{"running":0,"pending":2}}`, false},
		{"active", `{"training_active":true,"cuda_available":true,"queue":{"running":0}}`, true},
		{"running", `{"training_active":false,"cuda_available":true,"queue":{"running":1}}`, true},
		{"model_loaded", `{"training_active":false,"cuda_available":true,"model_loaded":"Qwen/Qwen2.5-0.5B","queue":{"running":0}}`, true},
		{"model_loaded_cpu", `{"training_active":false,"cuda_available":false,"model_loaded":"Qwen/Qwen2.5-0.5B","queue":{"running":0}}`, false},
		{"empty", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, busy := trainingOccupiesGPU(tc.json)
			if busy != tc.want {
				t.Fatalf("busy=%v want %v", busy, tc.want)
			}
		})
	}
}
