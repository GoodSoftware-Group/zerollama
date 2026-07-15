package modality

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// remoteTTSClient bounds time-to-first-byte; overall deadline comes from remoteTTSCtx.
var remoteTTSClient = &http.Client{
	Transport: func() http.RoundTripper {
		t := http.DefaultTransport.(*http.Transport).Clone()
		t.ResponseHeaderTimeout = 120 * time.Second
		return t
	}(),
}

// remoteSpeechBody is the OpenAI-compatible JSON body (kept local to avoid
// modality↔openai import cycles via openai tests).
type remoteSpeechBody struct {
	Model          string   `json:"model"`
	Input          string   `json:"input"`
	Voice          string   `json:"voice,omitempty"`
	ResponseFormat string   `json:"response_format,omitempty"`
	Speed          *float64 `json:"speed,omitempty"`
	Emotion        string   `json:"emotion,omitempty"`
}

// SpeechRemote POSTs an OpenAI-compatible /v1/audio/speech request to a GPU/edge TTS server
// (Chatterbox, Orpheus, Kokoro, etc.). URL resolution:
//  1. backend_paths.tts_url (full …/v1/audio/speech or base URL)
//  2. OLLAMA_TTS_URL fleet default
func SpeechRemote(ctx context.Context, cfg model.ConfigV2, localModel, text, voice, responseFormat, emotion string, speed *float64) ([]byte, string, error) {
	ctx, cancel := remoteTTSCtx(ctx)
	defer cancel()

	endpoint, err := resolveTTSEndpoint(cfg)
	if err != nil {
		return nil, "", err
	}

	upstream := PathFor(cfg, "tts_upstream_model")
	if upstream == "" {
		upstream = localModel
	}
	voice = strings.TrimSpace(voice)
	if voice == "" {
		voice = PathFor(cfg, "tts_default_voice")
	}

	payload, err := json.Marshal(remoteSpeechBody{
		Model:          upstream,
		Input:          text,
		Voice:          voice,
		ResponseFormat: responseFormat,
		Speed:          speed,
		Emotion:        emotion,
	})
	if err != nil {
		return nil, "", err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, "", err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if ref := PathFor(cfg, "tts_ref_audio"); ref != "" {
		// Hint for clone-capable servers; ignored by plain OpenAI speech shims.
		httpReq.Header.Set("X-TTS-Ref-Audio", ref)
	}

	resp, err := remoteTTSClient.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			return nil, "", fmt.Errorf("remote-tts: deadline exceeded or cancelled: %w", ctx.Err())
		}
		return nil, "", fmt.Errorf("remote-tts: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20)) // 64 MiB
	if err != nil {
		return nil, "", fmt.Errorf("remote-tts: read body: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg := strings.TrimSpace(string(data))
		if len(msg) > 512 {
			msg = msg[:512] + "…"
		}
		return nil, "", fmt.Errorf("remote-tts: upstream %s: %s", resp.Status, msg)
	}
	ct := resp.Header.Get("Content-Type")
	if ct == "" {
		ct = "audio/wav"
	}
	return data, ct, nil
}

func resolveTTSEndpoint(cfg model.ConfigV2) (string, error) {
	raw := strings.TrimSpace(PathFor(cfg, "tts_url"))
	if raw == "" {
		raw = envconfig.TTSURL()
	}
	if raw == "" {
		return "", fmt.Errorf("set backend_paths.tts_url or OLLAMA_TTS_URL for modality_backends.speech=remote-tts")
	}
	return normalizeSpeechURL(raw), nil
}

func normalizeSpeechURL(raw string) string {
	u := strings.TrimRight(strings.TrimSpace(raw), "/")
	if strings.HasSuffix(u, "/v1/audio/speech") {
		return u
	}
	if strings.HasSuffix(u, "/audio/speech") {
		return u
	}
	return u + "/v1/audio/speech"
}
