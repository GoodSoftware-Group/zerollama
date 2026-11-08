package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestSessionInputRate(t *testing.T) {
	ev := map[string]any{
		"type": "session.created",
		"session": map[string]any{
			"audio": map[string]any{
				"input": map[string]any{
					"format": map[string]any{"type": "audio/pcm", "rate": float64(16000)},
				},
			},
		},
	}
	rate, ok := sessionInputRate(ev)
	if !ok || rate != 16000 {
		t.Fatalf("rate=%d ok=%v", rate, ok)
	}
	if _, ok := sessionInputRate(map[string]any{"type": "error"}); ok {
		t.Fatal("error event has no rate")
	}
	ev["session"].(map[string]any)["audio"].(map[string]any)["output"] = map[string]any{
		"format": map[string]any{"type": "audio/pcm", "rate": float64(24000)},
	}
	out, ok := sessionOutputRate(ev)
	if !ok || out != 24000 {
		t.Fatalf("output rate=%d ok=%v", out, ok)
	}
}

func TestSpeechOutputRate(t *testing.T) {
	rate, ok := contentTypeRate("audio/pcm;rate=24000;encoding=float;bits=32")
	if !ok || rate != 24000 {
		t.Fatal(rate, ok)
	}
	if _, ok := contentTypeRate("audio/wav"); ok {
		t.Fatal("wav type has no rate")
	}
	rate, ok = wavHeaderRate(testWav(t, 16000, []int16{1, 2}))
	if !ok || rate != 16000 {
		t.Fatal(rate, ok)
	}
	if _, ok := wavHeaderRate([]byte("RIFF")); ok {
		t.Fatal("short")
	}
	wrapped := wavWrapFloat([]byte{0, 0, 0, 0}, 24000)
	rate, ok = wavHeaderRate(wrapped)
	if !ok || rate != 24000 || string(wrapped[20:22]) != "\x03\x00" {
		t.Fatal(rate, ok, wrapped[20:22])
	}
	sec, ok := headerAudioSeconds("1.5")
	if !ok || sec != 1.5 {
		t.Fatal(sec, ok)
	}
	if _, ok := headerAudioSeconds("0"); ok {
		t.Fatal("zero")
	}
	sec, ok = audioSeconds("audio/pcm;rate=24000;encoding=float;bits=32", []byte{0, 0, 0, 0, 0, 0, 0, 0}, 24000)
	if !ok || math.Abs(sec-2.0/24000) > 1e-12 {
		t.Fatal(sec, ok)
	}
	sec, ok = audioSeconds("audio/wav", testWav(t, 16000, []int16{1, 2}), 0)
	if !ok || math.Abs(sec-2.0/16000) > 1e-12 {
		t.Fatal(sec, ok)
	}
}

func TestWavAndResample(t *testing.T) {
	samples, rate, err := wavPCM16(bytes.NewReader(testWav(t, 8000, []int16{0, 1000, -1000, 2000})))
	if err != nil {
		t.Fatal(err)
	}
	if rate != 8000 || len(samples) != 4 || samples[1] != 1000 {
		t.Fatalf("decoded %+v @ %d", samples, rate)
	}
	out := resampleLinear(samples, 8000, 16000)
	if len(out) != 8 {
		t.Fatalf("resampled len %d", len(out))
	}
	if math.Abs(float64(out[0])) > 1 {
		t.Fatalf("start sample %d", out[0])
	}
	same := resampleLinear(samples, 8000, 8000)
	if len(same) != len(samples) || same[1] != samples[1] {
		t.Fatal("equal rate should copy")
	}
	chunks := chunkSamples(out, 16000, 1000)
	if len(chunks) != 1 || len(chunks[0]) != 8 {
		t.Fatalf("chunks %+v", chunks)
	}
	pcm := pcm16Bytes([]int16{0x0102})
	if pcm[0] != 0x02 || pcm[1] != 0x01 {
		t.Fatalf("endian %x", pcm)
	}
}

func testWav(t *testing.T, rate int, samples []int16) []byte {
	t.Helper()
	data := pcm16Bytes(samples)
	var buf bytes.Buffer
	write := func(b []byte) { buf.Write(b) }
	u32 := func(v uint32) {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], v)
		write(b[:])
	}
	u16 := func(v uint16) {
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], v)
		write(b[:])
	}
	write([]byte("RIFF"))
	u32(uint32(36 + len(data)))
	write([]byte("WAVE"))
	write([]byte("fmt "))
	u32(16)
	u16(1)
	u16(1)
	u32(uint32(rate))
	u32(uint32(rate * 2))
	u16(2)
	u16(16)
	write([]byte("data"))
	u32(uint32(len(data)))
	write(data)
	return buf.Bytes()
}
