package voicepipe

import (
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"unicode/utf8"
)

// firstClauseRunes is how soon the first words are spoken when the model has
// not emitted punctuation. Later stretches use clauseRunes, so the rest of
// the reply is not cut into tiny pieces. Piper is already loaded.
const firstClauseRunes = 16

// clauseRunes is how long a later stretch without punctuation may grow before
// it is spoken.
const clauseRunes = 48

// Clip is one spoken utterance. PCM is little-endian signed 16-bit mono.
// MIME audio/pcm concatenates across deltas. Any other MIME is one clip.
type Clip struct {
	MIME string
	Rate int
	PCM  []byte
	Raw  []byte
}

// SentenceStream feeds completion tokens to the speaker. A sentence end, a
// clause comma, semicolon, or colon, or a stretch of characters at a word
// boundary is spoken immediately. The first stretch is firstClauseRunes.
// A later stretch is clauseRunes. Flush emits a trailing fragment.
type SentenceStream struct {
	buf     strings.Builder
	started bool
}

// Push feeds the next completion tokens.
func (s *SentenceStream) Push(delta string, emit func(string) error) error {
	for _, r := range delta {
		s.buf.WriteRune(r)
		if r == '.' || r == '!' || r == '?' || r == '\n' || r == ',' || r == ';' || r == ':' {
			if err := s.emit(emit); err != nil {
				return err
			}
			continue
		}
		if err := s.emitOverflow(emit); err != nil {
			return err
		}
	}
	return nil
}

// Flush emits any text that had no closing punctuation.
func (s *SentenceStream) Flush(emit func(string) error) error {
	return s.emit(emit)
}

func (s *SentenceStream) emit(emit func(string) error) error {
	text := strings.TrimSpace(s.buf.String())
	s.buf.Reset()
	if text == "" || emit == nil {
		return nil
	}
	s.started = true
	return emit(text)
}

// emitOverflow speaks the words already buffered once they pass the stretch
// limit, and keeps the partial word that follows the last space.
func (s *SentenceStream) emitOverflow(emit func(string) error) error {
	raw := s.buf.String()
	limit := clauseRunes
	if !s.started {
		limit = firstClauseRunes
	}
	if utf8.RuneCountInString(strings.TrimSpace(raw)) < limit {
		return nil
	}
	cut := strings.LastIndexAny(raw, " \t")
	if cut <= 0 {
		return nil
	}
	head := strings.TrimSpace(raw[:cut])
	tail := strings.TrimLeft(raw[cut:], " \t")
	s.buf.Reset()
	s.buf.WriteString(tail)
	if head == "" || emit == nil {
		return nil
	}
	s.started = true
	return emit(head)
}

// Sentences splits a reply on sentence punctuation and newlines.
func Sentences(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	var out []string
	var b strings.Builder
	flush := func() {
		if t := strings.TrimSpace(b.String()); t != "" {
			out = append(out, t)
		}
		b.Reset()
	}
	for _, r := range s {
		b.WriteRune(r)
		if r == '.' || r == '!' || r == '?' || r == '\n' {
			flush()
		}
	}
	flush()
	return out
}

// WrapPCM16 builds a mono 16-bit WAV around raw little-endian samples.
func WrapPCM16(rate int, pcm []byte) ([]byte, error) {
	if rate <= 0 {
		return nil, errors.New("input audio rate is required")
	}
	if len(pcm)%2 != 0 {
		return nil, errors.New("pcm16 length must be even")
	}
	return wav(1, rate, 16, pcm), nil
}

// ParseWAV returns mono s16le samples. Stereo keeps the first channel.
// Float32 samples are scaled into s16le.
func ParseWAV(in []byte) (rate int, pcm []byte, err error) {
	if len(in) < 12 || string(in[0:4]) != "RIFF" || string(in[8:12]) != "WAVE" {
		return 0, nil, errors.New("audio is not a wav")
	}
	var format, channels, bits uint16
	var data []byte
	off := 12
	for off+8 <= len(in) {
		id := string(in[off : off+4])
		n := int(binary.LittleEndian.Uint32(in[off+4 : off+8]))
		off += 8
		if n < 0 || off+n > len(in) {
			return 0, nil, errors.New("wav chunk is truncated")
		}
		chunk := in[off : off+n]
		switch id {
		case "fmt ":
			if len(chunk) < 16 {
				return 0, nil, errors.New("wav fmt is truncated")
			}
			format = binary.LittleEndian.Uint16(chunk[0:2])
			channels = binary.LittleEndian.Uint16(chunk[2:4])
			rate = int(binary.LittleEndian.Uint32(chunk[4:8]))
			bits = binary.LittleEndian.Uint16(chunk[14:16])
		case "data":
			data = chunk
		}
		off += n
		if n%2 == 1 {
			off++
		}
	}
	if rate <= 0 || channels == 0 || data == nil {
		return 0, nil, errors.New("wav is missing fmt or data")
	}
	switch {
	case format == 1 && bits == 16:
		pcm, err = downmixS16(data, int(channels))
	case format == 3 && bits == 32:
		pcm, err = downmixF32(data, int(channels))
	default:
		return 0, nil, errors.New("wav must be 16-bit pcm or 32-bit float")
	}
	if err != nil {
		return 0, nil, err
	}
	return rate, pcm, nil
}

func wav(channels, rate, bits int, data []byte) []byte {
	n := len(data)
	out := make([]byte, 44+n)
	copy(out[0:4], "RIFF")
	binary.LittleEndian.PutUint32(out[4:8], uint32(36+n))
	copy(out[8:12], "WAVE")
	copy(out[12:16], "fmt ")
	binary.LittleEndian.PutUint32(out[16:20], 16)
	binary.LittleEndian.PutUint16(out[20:22], 1)
	binary.LittleEndian.PutUint16(out[22:24], uint16(channels))
	binary.LittleEndian.PutUint32(out[24:28], uint32(rate))
	binary.LittleEndian.PutUint32(out[28:32], uint32(rate*channels*bits/8))
	binary.LittleEndian.PutUint16(out[32:34], uint16(channels*bits/8))
	binary.LittleEndian.PutUint16(out[34:36], uint16(bits))
	copy(out[36:40], "data")
	binary.LittleEndian.PutUint32(out[40:44], uint32(n))
	copy(out[44:], data)
	return out
}

func downmixS16(data []byte, channels int) ([]byte, error) {
	if channels < 1 || len(data)%(2*channels) != 0 {
		return nil, errors.New("wav data does not match the channel count")
	}
	if channels == 1 {
		out := make([]byte, len(data))
		copy(out, data)
		return out, nil
	}
	frames := len(data) / (2 * channels)
	out := make([]byte, frames*2)
	for i := 0; i < frames; i++ {
		copy(out[i*2:i*2+2], data[i*2*channels:i*2*channels+2])
	}
	return out, nil
}

func downmixF32(data []byte, channels int) ([]byte, error) {
	if channels < 1 || len(data)%(4*channels) != 0 {
		return nil, errors.New("wav data does not match the channel count")
	}
	frames := len(data) / (4 * channels)
	out := make([]byte, frames*2)
	for i := 0; i < frames; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4*channels : i*4*channels+4])
		f := math.Float32frombits(bits)
		if f > 1 {
			f = 1
		}
		if f < -1 {
			f = -1
		}
		s := int16(f * 32767)
		binary.LittleEndian.PutUint16(out[i*2:i*2+2], uint16(s))
	}
	return out, nil
}
