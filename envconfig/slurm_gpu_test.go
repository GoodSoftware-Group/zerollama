package envconfig

import "testing"

func TestSlurmGPUsOnNodeZeroIsNotADevice(t *testing.T) {
	t.Setenv("ZEROLLAMA_SLURM_GUEST", "1")
	t.Setenv("CUDA_VISIBLE_DEVICES", "")
	t.Setenv("SLURM_STEP_GPUS", "")
	t.Setenv("SLURM_JOB_GPUS", "")
	t.Setenv("SLURM_GPUS_ON_NODE", "0")
	if SlurmGuestHasGPU() {
		t.Fatal("a count of 0 is not a GPU")
	}
	t.Setenv("SLURM_GPUS_ON_NODE", "1")
	if !SlurmGuestHasGPU() {
		t.Fatal("a count of 1 is a GPU")
	}
	t.Setenv("SLURM_GPUS_ON_NODE", "")
	t.Setenv("CUDA_VISIBLE_DEVICES", "0")
	if !SlurmGuestHasGPU() {
		t.Fatal("CUDA device 0 is a GPU")
	}
}
