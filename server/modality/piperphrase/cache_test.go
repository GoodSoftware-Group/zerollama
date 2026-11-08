package piperphrase

import (
	"bytes"
	"errors"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/server/modality/whispercli"
)

func TestSentenceSilence(t *testing.T) {
	got, err := SentenceSilence(nil)
	if err != nil || got != "" {
		t.Fatal(err, got)
	}
	v := 0.20
	if got, err = SentenceSilence(&v); err != nil || got != "0.2" {
		t.Fatal(err, got)
	}
	zero := 0.0
	if got, err = SentenceSilence(&zero); err != nil || got != "0" {
		t.Fatal(err, got)
	}
	bad := 6.0
	if _, err = SentenceSilence(&bad); err == nil {
		t.Fatal("expected reject")
	}
}

func TestNoiseScale(t *testing.T) {
	got, err := NoiseScale(nil)
	if err != nil || got != "" {
		t.Fatal(err, got)
	}
	v := 0.667
	if got, err = NoiseScale(&v); err != nil || got != "0.667" {
		t.Fatal(err, got)
	}
	zero := 0.0
	if got, err = NoiseScale(&zero); err != nil || got != "0" {
		t.Fatal(err, got)
	}
	bad := 3.0
	if _, err = NoiseScale(&bad); err == nil {
		t.Fatal("expected reject")
	}
}

func TestNoiseW(t *testing.T) {
	got, err := NoiseW(nil)
	if err != nil || got != "" {
		t.Fatal(err, got)
	}
	v := 0.80
	if got, err = NoiseW(&v); err != nil || got != "0.8" {
		t.Fatal(err, got)
	}
	zero := 0.0
	if got, err = NoiseW(&zero); err != nil || got != "0" {
		t.Fatal(err, got)
	}
	bad := 3.0
	if _, err = NoiseW(&bad); err == nil {
		t.Fatal("expected reject")
	}
}

func TestSpeakerID(t *testing.T) {
	got, err := SpeakerID(nil)
	if err != nil || got != "" {
		t.Fatal(err, got)
	}
	zero := 0
	if got, err = SpeakerID(&zero); err != nil || got != "" {
		t.Fatal(err, got)
	}
	id := 7
	if got, err = SpeakerID(&id); err != nil || got != "7" {
		t.Fatal(err, got)
	}
	bad := 4096
	if _, err = SpeakerID(&bad); err == nil {
		t.Fatal("expected reject")
	}
}

func TestPhonemeSilence(t *testing.T) {
	got, err := PhonemeSilence(nil)
	if err != nil || got != "" {
		t.Fatal(err, got)
	}
	got, err = PhonemeSilence(map[string]float64{".": 0.30, ",": 0})
	if err != nil || got != `[[",","0"],[".","0.3"]]` {
		t.Fatal(err, got)
	}
	args, err := PhonemeSilenceArgs(got)
	if err != nil || strings.Join(args, " ") != "--phoneme_silence , 0 --phoneme_silence . 0.3" {
		t.Fatal(err, args)
	}
	if _, err = PhonemeSilence(map[string]float64{"ab": 0.1}); err == nil {
		t.Fatal("expected reject")
	}
	if _, err = PhonemeSilence(map[string]float64{".": 6}); err == nil {
		t.Fatal("expected reject")
	}
}

func TestTashkeelArgs(t *testing.T) {
	args := TashkeelArgs("  /models/libtashkeel.ort  ")
	if len(args) != 2 || args[0] != "--tashkeel_model" || args[1] != "/models/libtashkeel.ort" {
		t.Fatal(args)
	}
	if TashkeelArgs("  ") != nil {
		t.Fatal("empty path")
	}
}

func TestESpeakArgs(t *testing.T) {
	args := ESpeakArgs("  /share/espeak-ng-data  ")
	if len(args) != 2 || args[0] != "--espeak_data" || args[1] != "/share/espeak-ng-data" {
		t.Fatal(args)
	}
	if ESpeakArgs("  ") != nil {
		t.Fatal("empty path")
	}
}

func TestESpeakCacheKey(t *testing.T) {
	reset()
	t.Cleanup(reset)
	base := Key{Model: "/voices/lessac.onnx", Text: "Hello"}
	with := base
	with.ESpeak = "/share/espeak-ng-data"
	if _, _, _, err := Flight(base, func() ([]byte, string, error) {
		return []byte("plain"), "audio/wav", nil
	}); err != nil {
		t.Fatal(err)
	}
	data, _, cached, err := Flight(with, func() ([]byte, string, error) {
		return []byte("spoken"), "audio/wav", nil
	})
	if err != nil || cached || string(data) != "spoken" {
		t.Fatalf("cached=%v data=%q err=%v", cached, data, err)
	}
}

func TestTashkeelCacheKey(t *testing.T) {
	reset()
	t.Cleanup(reset)
	base := Key{Model: "/voices/ar.onnx", Text: "مرحبا"}
	with := base
	with.Tashkeel = "/models/libtashkeel.ort"
	if _, _, _, err := Flight(base, func() ([]byte, string, error) {
		return []byte("plain"), "audio/wav", nil
	}); err != nil {
		t.Fatal(err)
	}
	data, _, cached, err := Flight(with, func() ([]byte, string, error) {
		return []byte("marked"), "audio/wav", nil
	})
	if err != nil || cached || string(data) != "marked" {
		t.Fatalf("cached=%v data=%q err=%v", cached, data, err)
	}
}

func TestPhraseCache(t *testing.T) {
	reset()
	t.Cleanup(reset)
	prevN, prevB := cacheMaxEntries, cacheMaxBytes
	cacheMaxEntries = 32
	cacheMaxBytes = 4096
	t.Cleanup(func() {
		cacheMaxEntries = prevN
		cacheMaxBytes = prevB
	})

	key := Key{Model: "/voices/lessac.onnx", Text: "Okay."}
	var calls atomic.Int32
	synth := func() ([]byte, string, error) {
		calls.Add(1)
		return []byte("wav-bytes"), "audio/wav", nil
	}
	data, ct, cached, err := Flight(key, synth)
	if err != nil || cached || ct != "audio/wav" || string(data) != "wav-bytes" || calls.Load() != 1 {
		t.Fatalf("miss data=%q cached=%v calls=%d err=%v", data, cached, calls.Load(), err)
	}
	data, _, cached, err = Flight(key, synth)
	if err != nil || !cached || string(data) != "wav-bytes" || calls.Load() != 1 {
		t.Fatalf("hit cached=%v calls=%d err=%v", cached, calls.Load(), err)
	}

	long := Key{Model: key.Model, Text: repeat("a", phraseMaxRunes+1)}
	if Cacheable(long.Text) {
		t.Fatal("long phrase cached")
	}
	_, _, cached, err = Flight(long, synth)
	if err != nil || cached || calls.Load() != 2 {
		t.Fatalf("long cached=%v calls=%d err=%v", cached, calls.Load(), err)
	}
	_, _, cached, err = Flight(long, synth)
	if err != nil || cached || calls.Load() != 3 {
		t.Fatalf("long repeat cached=%v calls=%d err=%v", cached, calls.Load(), err)
	}

	release := make(chan struct{})
	blocking := func() ([]byte, string, error) {
		calls.Add(1)
		<-release
		return []byte("wav-bytes"), "audio/wav", nil
	}
	sure := Key{Model: key.Model, Text: "Sure."}
	base := calls.Load()
	var wg sync.WaitGroup
	wg.Add(2)
	var hits atomic.Int32
	go func() {
		defer wg.Done()
		_, _, hit, err := Flight(sure, blocking)
		if err != nil {
			t.Error(err)
		}
		if hit {
			hits.Add(1)
		}
	}()
	waitFor(t, func() bool { return calls.Load() == base+1 })
	go func() {
		defer wg.Done()
		_, _, hit, err := Flight(sure, blocking)
		if err != nil {
			t.Error(err)
		}
		if hit {
			hits.Add(1)
		}
	}()
	waitFor(t, func() bool { return waiters.Load() == 1 })
	if calls.Load() != base+1 {
		t.Fatalf("flight started twice: %d", calls.Load())
	}
	close(release)
	wg.Wait()
	if hits.Load() != 1 {
		t.Fatalf("flight hits=%d", hits.Load())
	}

	for _, phrase := range Fillers() {
		if _, ok := Get(Key{Model: key.Model, Text: phrase}); ok {
			continue
		}
		if _, _, _, err := Flight(Key{Model: key.Model, Text: phrase}, synth); err != nil {
			t.Fatal(err)
		}
	}
	if !FillersReady(key.Model, "", "", "", "") {
		t.Fatal("fillers not ready")
	}
	before := calls.Load()
	for _, phrase := range Fillers() {
		if _, _, cached, err := Flight(Key{Model: key.Model, Text: phrase}, synth); err != nil || !cached {
			t.Fatalf("filler %q cached=%v err=%v", phrase, cached, err)
		}
	}
	if calls.Load() != before {
		t.Fatal("fillers synthesized twice")
	}

	reset()
	cacheMaxEntries = 2
	cacheMaxBytes = 1024
	put(Key{Text: "a"}, []byte("1"))
	put(Key{Text: "b"}, []byte("2"))
	put(Key{Text: "c"}, []byte("3"))
	if _, ok := Get(Key{Text: "a"}); ok {
		t.Fatal("oldest entry stayed")
	}
	if _, ok := Get(Key{Text: "c"}); !ok {
		t.Fatal("newest entry dropped")
	}

	if LengthScale(ptr(2)) != "0.5000" || LengthScale(nil) != "" || LengthScale(ptr(0)) != "" {
		t.Fatal(LengthScale(ptr(2)))
	}
	bad := errors.New("piper down")
	if _, _, _, err := Flight(Key{Model: key.Model, Text: "nope"}, func() ([]byte, string, error) {
		return nil, "", bad
	}); !errors.Is(err, bad) {
		t.Fatal(err)
	}
	if _, ok := Get(Key{Model: key.Model, Text: "nope"}); ok {
		t.Fatal("error was cached")
	}
}

func TestPiperStreamWAV(t *testing.T) {
	rate, ok := SampleRate([]byte(`{"audio":{"sample_rate":22050}}`))
	if !ok || rate != 22050 {
		t.Fatal(rate, ok)
	}
	if _, ok := SampleRate([]byte(`{"audio":{"sample_rate":0}}`)); ok {
		t.Fatal("zero rate")
	}
	pcm := bytes.Repeat([]byte{0, 1}, 22050)
	header := StreamHeader(22050)
	meter := whispercli.NewStreamMeter("audio/wav", 0)
	meter.Add(header)
	meter.Add(pcm)
	sec, ok := meter.Seconds()
	if !ok || math.Abs(sec-1) > 1e-9 {
		t.Fatalf("seconds %v %v", sec, ok)
	}
	wav := PCM16WAV(22050, pcm)
	if len(wav) != 44+len(pcm) || string(wav[0:4]) != "RIFF" {
		t.Fatal(len(wav))
	}
	got, ok := whispercli.WavSeconds(wav)
	if !ok || math.Abs(got-1) > 1e-9 {
		t.Fatalf("wav %v %v", got, ok)
	}
	if PCM16WAV(22050, []byte{0}) != nil || StreamHeader(0) != nil {
		t.Fatal("bad wav")
	}
}

func ptr(v float64) *float64 { return &v }

func repeat(s string, n int) string {
	out := make([]byte, 0, len(s)*n)
	for range n {
		out = append(out, s...)
	}
	return string(out)
}

func waitFor(t *testing.T, ready func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for !ready() {
		if time.Now().After(deadline) {
			t.Fatal("timed out")
		}
		time.Sleep(time.Millisecond)
	}
}
