package server

import (
	"runtime"
	"testing"
)

func TestTrainingPlatformSupported(t *testing.T) {
	if err := trainingPlatformSupported(); err != nil {
		t.Fatalf("unexpected: %v", err)
	}
}

func TestTrainingPlatformLabelDarwin(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("darwin only")
	}
	if trainingPlatformLabel() != "mps" {
		t.Fatalf("label=%q", trainingPlatformLabel())
	}
}

func TestTrainingPlatformLabelNonDarwin(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("non-darwin only")
	}
	if trainingPlatformLabel() != "cuda" {
		t.Fatalf("label=%q", trainingPlatformLabel())
	}
}

func TestTrainingSubmitUnsupportedQlora(t *testing.T) {
	if !TrainingSubmitUnsupported(ErrTrainingQloraUnsupported) {
		t.Fatal("expected QLoRA error to map to unsupported")
	}
}

func TestCheckTrainingQloraPayloadDarwin(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("darwin only")
	}
	err := checkTrainingQloraPayload("train", []byte(`{"use_qlora":true}`))
	if err == nil {
		t.Fatal("expected error")
	}
	if !TrainingSubmitUnsupported(err) {
		t.Fatalf("err=%v", err)
	}
}

func TestCheckTrainingQloraPayloadLoRAOk(t *testing.T) {
	err := checkTrainingQloraPayload("train", []byte(`{"use_lora":true,"use_qlora":false}`))
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
}
