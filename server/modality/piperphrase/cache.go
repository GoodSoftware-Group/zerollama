// Package piperphrase keeps short Piper and Kokoro clips in memory so a repeat
// does not start synthesis again.
package piperphrase

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"unicode/utf8"
)

// Fillers are the short acknowledgements synthesized ahead of the first listen.
func Fillers() []string {
	return []string{
		"Okay.",
		"Sure.",
		"One moment.",
		"Let me check.",
		"Got it.",
	}
}

const phraseMaxRunes = 48

var (
	cacheMaxEntries = 32
	cacheMaxBytes   = 8 << 20
)

// Key identifies one clip. voice is the sanitized voice name. scale is empty
// when the request left Piper's length default in place. silence is empty when
// the request left the 0.2 second sentence pause in place. noise is empty when
// the request left the voice JSON generator noise in place. width is empty when
// the request left the voice JSON phoneme-width noise in place. speaker is
// empty when the request left speaker 0 in place. phonemes is empty when the
// request added no per-character pauses. tashkeel is empty when no libtashkeel
// model path was configured. espeak is empty when no espeak-ng data directory
// was configured. Lang and Emotion are set for Kokoro. Piper leaves them empty.
type Key struct {
	Model, Config, Voice, Scale, Silence, Noise, Width, Speaker, Phonemes, Tashkeel, ESpeak, Text string
	Lang, Emotion                                                                                 string
}

type call struct {
	ch   chan struct{}
	data []byte
	ct   string
	err  error
}

var (
	mu       sync.Mutex
	wavs     = map[Key][]byte{}
	order    []Key
	wavBytes int
	inflight = map[Key]*call{}
	waiters  atomic.Int32
)

// Cacheable reports whether text is short enough to keep.
func Cacheable(text string) bool {
	return text != "" && utf8.RuneCountInString(text) <= phraseMaxRunes
}

// LengthScale converts an OpenAI speed (0.25–4, default 1) into Piper's
// length_scale string. Higher speed is a smaller scale. An omitted or
// non-positive speed leaves the Piper default in place.
func LengthScale(speed *float64) string {
	if speed == nil || *speed <= 0 {
		return ""
	}
	scale := 1.0 / *speed
	scale = max(0.25, min(4.0, scale))
	return fmt.Sprintf("%.4f", scale)
}

// SentenceSilence formats the seconds of silence Piper adds after each
// sentence. Nil leaves the 0.2 second default. A set value is from 0 to 5.
func SentenceSilence(v *float64) (string, error) {
	if v == nil {
		return "", nil
	}
	if math.IsNaN(*v) || math.IsInf(*v, 0) || *v < 0 || *v > 5 {
		return "", fmt.Errorf("sentence_silence must be between 0 and 5")
	}
	return strconv.FormatFloat(*v, 'f', -1, 64), nil
}

// NoiseScale formats the generator noise Piper adds while synthesizing.
// Nil leaves the voice JSON value, or 0.667 when that JSON omits it.
// A set value is from 0 to 2. 0 keeps the generator steady.
func NoiseScale(v *float64) (string, error) {
	if v == nil {
		return "", nil
	}
	if math.IsNaN(*v) || math.IsInf(*v, 0) || *v < 0 || *v > 2 {
		return "", fmt.Errorf("noise_scale must be between 0 and 2")
	}
	return strconv.FormatFloat(*v, 'f', -1, 64), nil
}

// NoiseW formats the phoneme-width noise Piper adds while synthesizing.
// Nil leaves the voice JSON value, or 0.8 when that JSON omits it.
// A set value is from 0 to 2. 0 keeps each phoneme's length steady.
func NoiseW(v *float64) (string, error) {
	if v == nil {
		return "", nil
	}
	if math.IsNaN(*v) || math.IsInf(*v, 0) || *v < 0 || *v > 2 {
		return "", fmt.Errorf("noise_w must be between 0 and 2")
	}
	return strconv.FormatFloat(*v, 'f', -1, 64), nil
}

// TashkeelArgs is the Piper --tashkeel_model flag. An empty path omits it.
// An Arabic voice then looks for libtashkeel_model.ort next to the Piper executable.
func TashkeelArgs(path string) []string {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	return []string{"--tashkeel_model", path}
}

// ESpeakArgs is the Piper --espeak_data flag. An empty path omits it.
// An eSpeak voice then looks for espeak-ng-data next to the Piper executable.
func ESpeakArgs(path string) []string {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	return []string{"--espeak_data", path}
}

// SpeakerID formats the Piper --speaker index. Nil, or 0, leaves the default
// speaker. A single-speaker voice leaves the value unused. A set value is
// from 1 to 4095.
func SpeakerID(v *int) (string, error) {
	if v == nil {
		return "", nil
	}
	if *v < 0 || *v > 4095 {
		return "", fmt.Errorf("speaker_id must be between 0 and 4095")
	}
	if *v == 0 {
		return "", nil
	}
	return strconv.Itoa(*v), nil
}

// PhonemeSilence formats Piper --phoneme_silence pairs. An empty map adds
// nothing. Each key is one character. Each value is seconds from 0 to 5.
// The result is a stable JSON array of [character, seconds] pairs.
func PhonemeSilence(m map[string]float64) (string, error) {
	if len(m) == 0 {
		return "", nil
	}
	if len(m) > 32 {
		return "", fmt.Errorf("phoneme_silence accepts at most 32 phonemes")
	}
	keys := make([]string, 0, len(m))
	for k, v := range m {
		r, size := utf8.DecodeRuneInString(k)
		if r == utf8.RuneError || size != len(k) {
			return "", fmt.Errorf("phoneme_silence keys must be a single character")
		}
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > 5 {
			return "", fmt.Errorf("phoneme_silence must be between 0 and 5")
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)
	pairs := make([][2]string, len(keys))
	for i, k := range keys {
		pairs[i] = [2]string{k, strconv.FormatFloat(m[k], 'f', -1, 64)}
	}
	raw, err := json.Marshal(pairs)
	if err != nil {
		return "", err
	}
	return string(raw), nil
}

// PhonemeSilenceArgs turns a PhonemeSilence value into --phoneme_silence flags.
func PhonemeSilenceArgs(canonical string) ([]string, error) {
	if canonical == "" {
		return nil, nil
	}
	var pairs [][2]string
	if err := json.Unmarshal([]byte(canonical), &pairs); err != nil {
		return nil, err
	}
	args := make([]string, 0, len(pairs)*3)
	for _, pair := range pairs {
		args = append(args, "--phoneme_silence", pair[0], pair[1])
	}
	return args, nil
}

// Get returns a copy of a cached clip.
func Get(key Key) ([]byte, bool) {
	mu.Lock()
	defer mu.Unlock()
	return loadLocked(key)
}

func loadLocked(key Key) ([]byte, bool) {
	wav, ok := wavs[key]
	if !ok {
		return nil, false
	}
	return cloneBytes(wav), true
}

func put(key Key, wav []byte) {
	if !Cacheable(key.Text) || len(wav) == 0 {
		return
	}
	wav = cloneBytes(wav)
	mu.Lock()
	defer mu.Unlock()
	if old, ok := wavs[key]; ok {
		wavBytes -= len(old)
		wavs[key] = wav
		wavBytes += len(wav)
		evictLocked()
		return
	}
	wavs[key] = wav
	order = append(order, key)
	wavBytes += len(wav)
	evictLocked()
}

func evictLocked() {
	for (len(order) > cacheMaxEntries || wavBytes > cacheMaxBytes) && len(order) > 0 {
		drop := order[0]
		order = order[1:]
		wavBytes -= len(wavs[drop])
		delete(wavs, drop)
	}
}

// Flight runs fn once for key. A caller that waits on an in-progress clip
// gets cached=true. The caller that ran fn gets cached=false.
func Flight(key Key, fn func() ([]byte, string, error)) ([]byte, string, bool, error) {
	mu.Lock()
	if wav, ok := loadLocked(key); ok {
		mu.Unlock()
		return wav, "audio/wav", true, nil
	}
	if pending, ok := inflight[key]; ok {
		mu.Unlock()
		waiters.Add(1)
		<-pending.ch
		if pending.err != nil {
			return nil, "", false, pending.err
		}
		return cloneBytes(pending.data), pending.ct, true, nil
	}
	pending := &call{ch: make(chan struct{})}
	inflight[key] = pending
	mu.Unlock()

	data, ct, err := fn()
	if err == nil {
		put(key, data)
	}
	pending.data, pending.ct, pending.err = data, ct, err
	close(pending.ch)

	mu.Lock()
	delete(inflight, key)
	mu.Unlock()
	if err != nil {
		return nil, "", false, err
	}
	return cloneBytes(data), ct, false, nil
}

// FillersReady reports whether every filler for this voice is already stored.
// model is the resolved ONNX path. scale is empty for Piper's default length.
// lang is empty for Piper and en-us or en-gb for Kokoro.
func FillersReady(model, config, voice, scale, lang string) bool {
	if model == "" {
		return false
	}
	for _, phrase := range Fillers() {
		if _, ok := Get(Key{Model: model, Config: config, Voice: voice, Scale: scale, Lang: lang, Text: phrase}); !ok {
			return false
		}
	}
	return true
}

// SampleRate reads audio.sample_rate from a Piper voice JSON file.
func SampleRate(raw []byte) (int, bool) {
	var doc struct {
		Audio struct {
			SampleRate float64 `json:"sample_rate"`
		} `json:"audio"`
	}
	if json.Unmarshal(raw, &doc) != nil {
		return 0, false
	}
	n := int(doc.Audio.SampleRate)
	if n <= 0 || n > 192000 || float64(n) != doc.Audio.SampleRate {
		return 0, false
	}
	return n, true
}

// StreamHeader is a mono 16-bit WAV header whose data length is unknown.
// PCM bytes that follow are little-endian int16 at rate Hz.
func StreamHeader(rate int) []byte {
	if rate <= 0 {
		return nil
	}
	return wavHeader(rate, ^uint32(0))
}

// PCM16WAV is a finite mono 16-bit WAV. An odd length or a rate of 0 returns nil.
func PCM16WAV(rate int, pcm []byte) []byte {
	if rate <= 0 || len(pcm) == 0 || len(pcm)%2 != 0 || len(pcm) > int(^uint32(0))-36 {
		return nil
	}
	h := wavHeader(rate, uint32(len(pcm)))
	out := make([]byte, 0, len(h)+len(pcm))
	out = append(out, h...)
	out = append(out, pcm...)
	return out
}

func wavHeader(rate int, dataLen uint32) []byte {
	b := make([]byte, 44)
	copy(b[0:], "RIFF")
	riff := uint32(36) + dataLen
	if dataLen == ^uint32(0) {
		riff = ^uint32(0)
	}
	binary.LittleEndian.PutUint32(b[4:], riff)
	copy(b[8:], "WAVE")
	copy(b[12:], "fmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], 1)
	binary.LittleEndian.PutUint16(b[22:], 1)
	binary.LittleEndian.PutUint32(b[24:], uint32(rate))
	binary.LittleEndian.PutUint32(b[28:], uint32(rate)*2)
	binary.LittleEndian.PutUint16(b[32:], 2)
	binary.LittleEndian.PutUint16(b[34:], 16)
	copy(b[36:], "data")
	binary.LittleEndian.PutUint32(b[40:], dataLen)
	return b
}

// Store keeps a finished WAV when the phrase is short enough.
func Store(key Key, wav []byte) {
	put(key, wav)
}

func cloneBytes(b []byte) []byte {
	if len(b) == 0 {
		return nil
	}
	out := make([]byte, len(b))
	copy(out, b)
	return out
}

func reset() {
	mu.Lock()
	defer mu.Unlock()
	wavs = map[Key][]byte{}
	order = nil
	wavBytes = 0
	inflight = map[Key]*call{}
	waiters.Store(0)
}
