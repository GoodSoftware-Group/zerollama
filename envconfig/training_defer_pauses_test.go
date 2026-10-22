package envconfig

import "testing"

func TestTrainingDeferPausesGgml(t *testing.T) {
	t.Setenv("ZEROLLAMA_TRAINING_QUEUE_ON_BUSY", "")
	t.Setenv("ZEROLLAMA_TRAINING_DEFER_PAUSES_GGML", "")
	if TrainingDeferPausesGgml() {
		t.Fatal("default off when queue_on_busy unset")
	}

	t.Setenv("ZEROLLAMA_TRAINING_QUEUE_ON_BUSY", "1")
	t.Setenv("ZEROLLAMA_TRAINING_DEFER_PAUSES_GGML", "")
	if !TrainingDeferPausesGgml() {
		t.Fatal("should follow queue_on_busy=1 by default")
	}

	t.Setenv("ZEROLLAMA_TRAINING_DEFER_PAUSES_GGML", "0")
	if TrainingDeferPausesGgml() {
		t.Fatal("explicit 0 must win")
	}

	t.Setenv("ZEROLLAMA_TRAINING_QUEUE_ON_BUSY", "0")
	t.Setenv("ZEROLLAMA_TRAINING_DEFER_PAUSES_GGML", "1")
	if !TrainingDeferPausesGgml() {
		t.Fatal("explicit 1 must win")
	}
}
