package model

import "testing"

func TestIsLlamaASRName(t *testing.T) {
	for _, name := range []string{"parakeet", "PARAKEET", " nemotron_transcribe ", "nemotron-transcribe"} {
		if !IsLlamaASRName(name) {
			t.Fatalf("IsLlamaASRName(%q) = false", name)
		}
	}
	for _, name := range []string{"", "clip", "whisper", "qwen35", "nemotron"} {
		if IsLlamaASRName(name) {
			t.Fatalf("IsLlamaASRName(%q) = true", name)
		}
	}
}

func TestLlamaASRKV(t *testing.T) {
	if !LlamaASRKV("general.architecture", "parakeet") {
		t.Fatal("architecture")
	}
	if !LlamaASRKV("clip.projector_type", "nemotron_transcribe") {
		t.Fatal("clip projector")
	}
	if !LlamaASRKV("clip.audio.projector_type", "parakeet") {
		t.Fatal("audio projector")
	}
	if !LlamaASRKV("clip.projector_type", "nemotron-transcribe") {
		t.Fatal("hyphenated projector")
	}
	if LlamaASRKV("general.architecture", "clip") || LlamaASRKV("general.description", "parakeet") {
		t.Fatal("unrelated keys must not match")
	}
}
