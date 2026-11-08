package modality

import (
	"context"
	"io"

	"github.com/ollama/ollama/server/modality/remotetts"
	"github.com/ollama/ollama/types/model"
)

// SpeechRemote POSTs an OpenAI-compatible /v1/audio/speech request to a GPU/edge TTS server
// (Chatterbox, Orpheus, Kokoro, Irodori, etc.). URL resolution:
//  1. backend_paths.tts_url (full …/v1/audio/speech or base URL)
//  2. OLLAMA_TTS_URL fleet default
//
// seconds is X-Audio-Seconds when the upstream value is a positive duration.
func SpeechRemote(ctx context.Context, cfg model.ConfigV2, localModel, text, voice, responseFormat, emotion string, speed *float64) ([]byte, string, string, error) {
	return remotetts.SpeechRemote(ctx, cfg, localModel, text, voice, responseFormat, emotion, speed)
}

// SpeechRemoteStream forwards stream:true and returns the upstream body.
// seconds is X-Audio-Seconds when the upstream value is a positive duration.
// The caller closes r, which also ends the request deadline.
func SpeechRemoteStream(ctx context.Context, cfg model.ConfigV2, localModel, text, voice, responseFormat, emotion string, speed *float64) (contentType, seconds string, r io.ReadCloser, err error) {
	return remotetts.SpeechRemoteStream(ctx, cfg, localModel, text, voice, responseFormat, emotion, speed)
}
