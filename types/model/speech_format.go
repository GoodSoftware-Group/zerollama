package model

import (
	"fmt"
	"strconv"
	"strings"
)

// SpeechContainer is the audio container a local speech backend can return.
// An empty format is wav. pcm is allowed only when the backend can emit raw
// samples (llama-server gen projectors). Piper accepts wav. Remote engines
// are not checked here.
func SpeechContainer(format string, allowPCM bool) (string, error) {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "", "wav":
		return "wav", nil
	case "pcm":
		if allowPCM {
			return "pcm", nil
		}
		return "", fmt.Errorf("response_format must be wav")
	default:
		if allowPCM {
			return "", fmt.Errorf("response_format must be wav or pcm")
		}
		return "", fmt.Errorf("response_format must be wav")
	}
}

// PCMContentType fills a bare audio/pcm type with the gen projector's rate.
// llama-server PCM is little-endian float32. A header that already has rate=
// is returned unchanged, including a non-pcm type.
func PCMContentType(header string, rate int) string {
	h := strings.TrimSpace(header)
	if h == "" {
		h = "audio/pcm"
	}
	lower := strings.ToLower(h)
	if !strings.Contains(lower, "audio/pcm") || rate <= 0 || strings.Contains(lower, "rate=") {
		return h
	}
	h += ";rate=" + strconv.Itoa(rate)
	if !strings.Contains(lower, "encoding=") {
		h += ";encoding=float"
	}
	if !strings.Contains(lower, "bits=") {
		h += ";bits=32"
	}
	return h
}
