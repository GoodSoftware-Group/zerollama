package model

type Capability string

const (
	CapabilityCompletion = Capability("completion")
	CapabilityTools      = Capability("tools")
	CapabilityInsert     = Capability("insert")
	CapabilityVision     = Capability("vision")
	// CapabilityVideo marks models that accept video-understanding requests (expanded frames).
	// Distinct from CapabilityVision so callers can fail fast with a clear error when video is present.
	CapabilityVideo     = Capability("video")
	CapabilityEmbedding = Capability("embedding")
	CapabilityThinking  = Capability("thinking")
	CapabilityImage     = Capability("image")
	CapabilityAudio     = Capability("audio")
	CapabilitySpeech    = Capability("speech") // text-to-speech
)

func (c Capability) String() string {
	return string(c)
}
