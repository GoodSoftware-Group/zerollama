package modality

import (
	"bytes"
	"path/filepath"
	"strings"
)

// AudioInputExt picks a filename extension for subprocess STT tools. Prefer the
// upload filename; fall back to sniffing common container magic bytes.
func AudioInputExt(originalFilename string, data []byte) string {
	if ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(originalFilename)), "."); ext != "" {
		switch ext {
		case "wav", "mp3", "m4a", "aac", "ogg", "opus", "webm", "flac", "mp4", "mpeg", "mpga":
			return ext
		}
	}
	if len(data) < 12 {
		return "wav"
	}
	if bytes.HasPrefix(data, []byte{0x49, 0x44, 0x33}) || (len(data) >= 2 && data[0] == 0xff && data[1]&0xe0 == 0xe0) {
		return "mp3"
	}
	if len(data) >= 12 && string(data[0:4]) == "RIFF" && string(data[8:12]) == "WAVE" {
		return "wav"
	}
	if len(data) >= 4 && string(data[0:4]) == "fLaC" {
		return "flac"
	}
	if len(data) >= 4 && string(data[0:4]) == "OggS" {
		return "ogg"
	}
	// WebM / Matroska EBML header
	if len(data) >= 4 && data[0] == 0x1a && data[1] == 0x45 && data[2] == 0xdf && data[3] == 0xa3 {
		return "webm"
	}
	return "wav"
}
