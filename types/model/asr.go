package model

import "strings"

// GGUF families and llama.cpp mtmd projector types that transcribe through
// llama-server. Parakeet is the conformer encoder. nemotron_transcribe is the
// NeMo Parallel Expert Encoder whose Streaming Sortformer cache is fused into
// the ASR states (the HTTP transcript does not include speaker-turn labels).
const (
	ASRParakeet            = "parakeet"
	ASRNemotronTranscribe  = "nemotron_transcribe"
	ASRNemotronTranscribe2 = "nemotron-transcribe"
)

// IsLlamaASRName reports whether s is a Parakeet or Nemotron-transcribe
// architecture or projector type.
func IsLlamaASRName(name string) bool {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case ASRParakeet, ASRNemotronTranscribe, ASRNemotronTranscribe2:
		return true
	default:
		return false
	}
}

// LlamaASRKV reports whether a GGUF metadata pair names a llama-server ASR model.
// Keys are the literal llama.cpp mtmd strings (clip.projector_type, and the
// architecture-prefixed form the Go reader stores).
func LlamaASRKV(key, value string) bool {
	k := strings.ToLower(strings.TrimSpace(key))
	switch {
	case k == "general.architecture",
		k == "clip.projector_type",
		k == "clip.audio.projector_type",
		strings.HasSuffix(k, ".projector_type"):
		return IsLlamaASRName(value)
	default:
		return false
	}
}
