package openai

import (
	"encoding/base64"
	"errors"
	"math"
	"strconv"
	"strings"
)

// CleanSpeechLang checks the optional lang field forwarded to a llama-server
// TTS template. Empty is fine. Piper ignores it.
func CleanSpeechLang(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	if len(s) > 32 {
		return "", errors.New("lang is too long")
	}
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
		default:
			return "", errors.New("lang must be letters, digits, '-' or '_'")
		}
	}
	return s, nil
}

const maxSpeakerRefBytes = 8 << 20

// DecodeSpeakerRef decodes a speaker-reference clip. A data: URI prefix is
// stripped. Empty input returns nil.
func DecodeSpeakerRef(s string) ([]byte, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	if i := strings.IndexByte(s, ','); i >= 0 && strings.Contains(strings.ToLower(s[:i]), "base64") {
		s = s[i+1:]
	}
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		b, err = base64.RawStdEncoding.DecodeString(s)
	}
	if err != nil {
		return nil, errors.New("speaker_ref is not base64")
	}
	if len(b) == 0 {
		return nil, errors.New("speaker_ref is empty")
	}
	if len(b) > maxSpeakerRefBytes {
		return nil, errors.New("speaker_ref is larger than 8MiB")
	}
	return b, nil
}

// CleanSpeechTopK checks the optional acoustic top-k. Empty uses the model
// default. A set value is from 0 through MaxInt32.
func CleanSpeechTopK(v *int) error {
	if v == nil {
		return nil
	}
	if *v < 0 || *v > math.MaxInt32 {
		return errors.New("top_k must be between 0 and 2147483647")
	}
	return nil
}

// CleanSpeechTopP checks the optional acoustic top-p. Empty uses the model
// default. A set value is from 0 through 1.
func CleanSpeechTopP(v *float64) error {
	if v == nil {
		return nil
	}
	if math.IsNaN(*v) || *v < 0 || *v > 1 {
		return errors.New("top_p must be between 0 and 1")
	}
	return nil
}

// CleanSpeechRepeatPenalty checks the optional backbone repetition penalty.
// Empty leaves llama-server at 1.05. A set value must be finite and greater than 0.
func CleanSpeechRepeatPenalty(v *float64) error {
	if v == nil {
		return nil
	}
	if math.IsNaN(*v) || math.IsInf(*v, 0) || *v <= 0 {
		return errors.New("repeat_penalty must be greater than 0")
	}
	return nil
}

// CleanTranscriptionTemperature checks the optional transcription temperature.
// Empty means the server default. A set value is between 0 and 1.
func CleanTranscriptionTemperature(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || f < 0 || f > 1 {
		return "", errors.New("temperature must be between 0 and 1")
	}
	return strconv.FormatFloat(f, 'f', -1, 64), nil
}
