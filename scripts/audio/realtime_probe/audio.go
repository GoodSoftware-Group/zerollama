package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

// sessionInputRate reads audio.input.format.rate from a session.created or
// session.updated event. llama-server is the source of the rate; the probe
// resamples the wav to it before any append.
func sessionInputRate(ev map[string]any) (int, bool) {
	session, _ := ev["session"].(map[string]any)
	audio, _ := session["audio"].(map[string]any)
	input, _ := audio["input"].(map[string]any)
	format, _ := input["format"].(map[string]any)
	return jsonInt(format["rate"])
}

// sessionOutputRate reads audio.output.format.rate. Conversation models that
// speak (LFM3) include it. A transcription session has no output rate.
func sessionOutputRate(ev map[string]any) (int, bool) {
	session, _ := ev["session"].(map[string]any)
	audio, _ := session["audio"].(map[string]any)
	output, _ := audio["output"].(map[string]any)
	format, _ := output["format"].(map[string]any)
	return jsonInt(format["rate"])
}

func jsonInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		if n <= 0 || n > 192000 || n != math.Trunc(n) {
			return 0, false
		}
		return int(n), true
	case json.Number:
		i, err := n.Int64()
		if err != nil || i <= 0 || i > 192000 {
			return 0, false
		}
		return int(i), true
	default:
		return 0, false
	}
}

// wavHeaderRate reads the sample rate from a RIFF WAVE fmt chunk.
// The format code can be integer PCM or float. A missing or absurd rate returns false.
func wavHeaderRate(b []byte) (int, bool) {
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return 0, false
	}
	off := 12
	for off+8 <= len(b) {
		id := string(b[off : off+4])
		size := int(binary.LittleEndian.Uint32(b[off+4 : off+8]))
		off += 8
		if size < 0 || off+size > len(b) {
			return 0, false
		}
		if id == "fmt " && size >= 8 {
			rate := int(binary.LittleEndian.Uint32(b[off+4 : off+8]))
			if rate > 0 && rate <= 192000 {
				return rate, true
			}
			return 0, false
		}
		off += size
		if size%2 == 1 {
			off++
		}
	}
	return 0, false
}

// audioSeconds is the clip length. PCM uses the known rate and the sample
// width in the content type (float32 when the width is omitted). A WAV uses
// the fmt byte rate and the data chunk.
func audioSeconds(ct string, audio []byte, rate int) (float64, bool) {
	lower := strings.ToLower(ct)
	if strings.Contains(lower, "audio/pcm") {
		if rate <= 0 || len(audio) == 0 {
			return 0, false
		}
		frame := 4
		switch {
		case strings.Contains(lower, "bits=8"):
			frame = 1
		case strings.Contains(lower, "bits=16"):
			frame = 2
		case strings.Contains(lower, "bits=24"):
			frame = 3
		}
		if len(audio)%frame != 0 {
			return 0, false
		}
		return float64(len(audio)/frame) / float64(rate), true
	}
	return wavSeconds(audio)
}

func wavSeconds(b []byte) (float64, bool) {
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return 0, false
	}
	var byteRate uint32
	var dataLen int
	var gotFmt, gotData bool
	off := 12
	for off+8 <= len(b) {
		id := string(b[off : off+4])
		size := int(binary.LittleEndian.Uint32(b[off+4 : off+8]))
		off += 8
		if size < 0 || off+size > len(b) {
			return 0, false
		}
		body := b[off : off+size]
		switch id {
		case "fmt ":
			if len(body) < 16 {
				return 0, false
			}
			byteRate = binary.LittleEndian.Uint32(body[8:12])
			gotFmt = true
		case "data":
			dataLen = len(body)
			gotData = true
		}
		off += size
		if size%2 == 1 {
			off++
		}
	}
	if !gotFmt || !gotData || byteRate == 0 || dataLen <= 0 {
		return 0, false
	}
	return float64(dataLen) / float64(byteRate), true
}

// headerAudioSeconds reads X-Audio-Seconds. Zero, negative, and non-numeric
// values are ignored so the body length can still be used.
func headerAudioSeconds(v string) (float64, bool) {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0, false
	}
	n, err := strconv.ParseFloat(v, 64)
	if err != nil || math.IsNaN(n) || math.IsInf(n, 0) || n <= 0 || n > 86400 {
		return 0, false
	}
	return n, true
}

// contentTypeRate reads rate= from a Content-Type such as audio/pcm;rate=24000.
func contentTypeRate(ct string) (int, bool) {
	lower := strings.ToLower(ct)
	i := strings.Index(lower, "rate=")
	if i < 0 {
		return 0, false
	}
	rest := strings.TrimSpace(ct[i+len("rate="):])
	end := 0
	for end < len(rest) && rest[end] >= '0' && rest[end] <= '9' {
		end++
	}
	if end == 0 {
		return 0, false
	}
	n, err := strconv.Atoi(rest[:end])
	if err != nil || n <= 0 || n > 192000 {
		return 0, false
	}
	return n, true
}

// wavPCM16 reads a PCM integer wav and returns mono int16 samples plus the
// file's sample rate. Stereo is averaged. 8-bit PCM is shifted to signed 16-bit.
func wavPCM16(r io.Reader) (samples []int16, rate int, err error) {
	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, 0, err
	}
	if len(raw) < 12 || string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a RIFF WAVE file")
	}
	var (
		format          uint16 = 1
		channels, bits  uint16
		gotFmt, gotData bool
		data            []byte
	)
	off := 12
	for off+8 <= len(raw) {
		id := string(raw[off : off+4])
		size := int(binary.LittleEndian.Uint32(raw[off+4 : off+8]))
		off += 8
		if size < 0 || off+size > len(raw) {
			return nil, 0, fmt.Errorf("wav chunk %s overruns the file", id)
		}
		body := raw[off : off+size]
		switch id {
		case "fmt ":
			if len(body) < 16 {
				return nil, 0, fmt.Errorf("wav fmt chunk is short")
			}
			format = binary.LittleEndian.Uint16(body[0:2])
			channels = binary.LittleEndian.Uint16(body[2:4])
			rate = int(binary.LittleEndian.Uint32(body[4:8]))
			bits = binary.LittleEndian.Uint16(body[14:16])
			gotFmt = true
		case "data":
			data = body
			gotData = true
		}
		off += size
		if size%2 == 1 {
			off++
		}
	}
	if !gotFmt || !gotData {
		return nil, 0, fmt.Errorf("wav is missing fmt or data")
	}
	if format != 1 {
		return nil, 0, fmt.Errorf("wav format %d is not PCM; convert with ffmpeg -ac 1 -ar <rate> -sample_fmt s16", format)
	}
	if channels < 1 || rate <= 0 {
		return nil, 0, fmt.Errorf("wav has no channels or sample rate")
	}
	switch bits {
	case 16:
		if len(data)%2 != 0 {
			return nil, 0, fmt.Errorf("wav data length is odd")
		}
		n := len(data) / 2
		frame := make([]int16, n)
		for i := 0; i < n; i++ {
			frame[i] = int16(binary.LittleEndian.Uint16(data[i*2:]))
		}
		samples = downmix(frame, int(channels))
	case 8:
		frame := make([]int16, len(data))
		for i, b := range data {
			frame[i] = int16(int(b)-128) << 8
		}
		samples = downmix(frame, int(channels))
	default:
		return nil, 0, fmt.Errorf("wav bit depth %d is not 16 or 8", bits)
	}
	if len(samples) == 0 {
		return nil, 0, fmt.Errorf("wav has no samples")
	}
	return samples, rate, nil
}

func downmix(frame []int16, channels int) []int16 {
	if channels <= 1 {
		return frame
	}
	n := len(frame) / channels
	out := make([]int16, n)
	for i := 0; i < n; i++ {
		var sum int
		for c := 0; c < channels; c++ {
			sum += int(frame[i*channels+c])
		}
		out[i] = int16(sum / channels)
	}
	return out
}

// resampleLinear stretches or squeezes mono PCM so the buffer matches the
// rate from session.created. Equal rates return a copy.
func resampleLinear(in []int16, from, to int) []int16 {
	if from <= 0 || to <= 0 || len(in) == 0 || from == to {
		out := make([]int16, len(in))
		copy(out, in)
		return out
	}
	outLen := int(int64(len(in)) * int64(to) / int64(from))
	if outLen < 1 {
		outLen = 1
	}
	out := make([]int16, outLen)
	last := len(in) - 1
	for i := range out {
		src := float64(i) * float64(from) / float64(to)
		j := int(src)
		if j >= last {
			out[i] = in[last]
			continue
		}
		frac := src - float64(j)
		v := float64(in[j])*(1-frac) + float64(in[j+1])*frac
		out[i] = int16(math.Round(v))
	}
	return out
}

// pcm16Bytes is little-endian PCM16, the encoding input_audio_buffer.append expects.
func pcm16Bytes(samples []int16) []byte {
	out := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(s))
	}
	return out
}

// chunkSamples splits samples into blocks of about chunkMs at rate.
func chunkSamples(samples []int16, rate, chunkMs int) [][]int16 {
	if chunkMs <= 0 {
		chunkMs = 320
	}
	if rate <= 0 {
		rate = 16000
	}
	n := rate * chunkMs / 1000
	if n < 1 {
		n = 1
	}
	var out [][]int16
	for len(samples) > 0 {
		take := n
		if take > len(samples) {
			take = len(samples)
		}
		part := make([]int16, take)
		copy(part, samples[:take])
		out = append(out, part)
		samples = samples[take:]
	}
	return out
}
