package voicepipe

import (
	"encoding/binary"
	"errors"
	"strings"
)

// Turn is turn-taking for a duplex socket. With amplitude, a sample is speech
// when its absolute value divided by 32768 is at least Threshold. With a
// neural scorer, Threshold is a speech probability cutoff (default 0.5 when
// the client omitted it). AmplitudeScale is always the PCM amplitude cutoff
// used for trim, barge peeks, and Score fallback (default 0.02). After speech,
// SilenceMS of quieter audio ends the turn. Quiet samples at the end of the
// packet that held the last speech count toward that silence. PrefixMS of
// audio from before the speech is kept.
type Turn struct {
	Threshold float64
	// ThresholdSet is true when the client sent threshold. Neural mode uses
	// 0.5 when it is false. Amplitude mode still uses DefaultTurn's 0.02.
	ThresholdSet bool
	// AmplitudeScale is the PCM amplitude cutoff. Neural mode keeps this at
	// DefaultAmplitude even when Threshold is a probability.
	AmplitudeScale float64
	SilenceMS      int
	PrefixMS       int
}

// DefaultAmplitude is the PCM amplitude cutoff for amplitude VAD and neural fallback.
const DefaultAmplitude = 0.02

// DefaultTurn is the server_vad setting used when the fields are omitted.
func DefaultTurn() Turn {
	return Turn{Threshold: DefaultAmplitude, AmplitudeScale: DefaultAmplitude, SilenceMS: 500, PrefixMS: 300}
}

// NeuralDefaultThreshold is the speech probability cutoff when the client
// omitted threshold and a neural scorer is active.
const NeuralDefaultThreshold = 0.5

// ApplyNeuralThreshold sets Threshold to NeuralDefaultThreshold when the
// client omitted it, and pins AmplitudeScale to DefaultAmplitude so Score
// misses and trim/barge peeks stay on a real amplitude cutoff.
func (t *Turn) ApplyNeuralThreshold() {
	if t == nil {
		return
	}
	if !t.ThresholdSet {
		t.Threshold = NeuralDefaultThreshold
	}
	t.AmplitudeScale = DefaultAmplitude
}

// Amp returns the PCM amplitude cutoff for this turn.
func (t Turn) Amp() float64 {
	return t.AmplitudeScale
}

// ParseTurn reads session.audio.input.turn_detection. Null and type none
// turn it off. server_vad turns it on. Omitted fields use DefaultTurn.
func ParseTurn(v any) (on bool, turn Turn, err error) {
	if v == nil {
		return false, Turn{}, nil
	}
	m, ok := v.(map[string]any)
	if !ok {
		return false, Turn{}, errors.New("turn_detection must be an object")
	}
	switch strings.ToLower(strings.TrimSpace(stringField(m, "type"))) {
	case "", "none":
		return false, Turn{}, nil
	case "server_vad":
	default:
		return false, Turn{}, errors.New("turn_detection type must be server_vad or none")
	}
	turn = DefaultTurn()
	if raw, ok := m["threshold"]; ok {
		f, ok := number(raw)
		if !ok || f < 0 || f > 1 {
			return false, Turn{}, errors.New("turn_detection threshold must be between 0 and 1")
		}
		turn.Threshold = f
		turn.AmplitudeScale = f
		turn.ThresholdSet = true
	}
	if raw, ok := m["silence_duration_ms"]; ok {
		n, ok := whole(raw)
		if !ok || n < 0 || n > 10000 {
			return false, Turn{}, errors.New("turn_detection silence_duration_ms must be between 0 and 10000")
		}
		turn.SilenceMS = n
	}
	if raw, ok := m["prefix_padding_ms"]; ok {
		n, ok := whole(raw)
		if !ok || n < 0 || n > 5000 {
			return false, Turn{}, errors.New("turn_detection prefix_padding_ms must be between 0 and 5000")
		}
		turn.PrefixMS = n
	}
	return true, turn, nil
}

func stringField(m map[string]any, key string) string {
	s, _ := m[key].(string)
	return s
}

func number(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	default:
		return 0, false
	}
}

func whole(v any) (int, bool) {
	f, ok := number(v)
	if !ok || f != float64(int(f)) {
		return 0, false
	}
	return int(f), true
}

// Loud reports whether any PCM16 sample reaches the amplitude threshold.
// Threshold 0 treats any non-zero sample as speech.
func Loud(pcm []byte, threshold float64) bool {
	for i := 0; i+1 < len(pcm); i += 2 {
		if sampleLoud(int(int16(binary.LittleEndian.Uint16(pcm[i:i+2]))), threshold) {
			return true
		}
	}
	return false
}

func sampleLoud(v int, threshold float64) bool {
	if v < 0 {
		v = -v
	}
	if threshold <= 0 {
		return v > 0
	}
	return float64(v) >= threshold*32768
}

// hearTailMS is how much quiet audio after the last loud sample the
// transcriber still receives. A longer silence is left out of that clip.
const hearTailMS = 200

// loudSpeculateMS is how long a loud packet waits, with no further audio,
// before transcription starts. The turn still waits out its silence.
const loudSpeculateMS = 100

// quietSuffix is the number of samples after the last loud sample.
// The whole chunk is quiet when no sample is loud.
func quietSuffix(pcm []byte, threshold float64) int {
	last := lastLoudSample(pcm, threshold)
	samples := len(pcm) / 2
	if last < 0 {
		return samples
	}
	return samples - last - 1
}

// lastLoudSample is the index of the last amplitude-loud sample, or -1.
func lastLoudSample(pcm []byte, threshold float64) int {
	lastLoud := -1
	samples := 0
	for i := 0; i+1 < len(pcm); i += 2 {
		if sampleLoud(int(int16(binary.LittleEndian.Uint16(pcm[i:i+2]))), threshold) {
			lastLoud = samples
		}
		samples++
	}
	return lastLoud
}

// SpeechScorer returns the index of the last speech sample in pcm, or -1
// when the chunk is silence. Rate is PCM16 samples per second. A timeout or
// other error leaves the gate on amplitude for that packet.
type SpeechScorer func(pcm []byte, rate int, threshold float64) (lastSpeech int, err error)

// TrimQuietTail keeps keepMS of quiet audio after the last loud sample.
// A clip with no loud sample is returned as it is. Leading audio stays.
func TrimQuietTail(pcm []byte, rate int, threshold float64, keepMS int) []byte {
	if rate <= 0 || keepMS < 0 || len(pcm) < 4 {
		return pcm
	}
	tail := quietSuffix(pcm, threshold)
	samples := len(pcm) / 2
	if tail == samples {
		return pcm
	}
	keep := rate * keepMS / 1000
	if tail <= keep {
		return pcm
	}
	end := (samples - (tail - keep)) * 2
	if end < 2 || end > len(pcm) {
		return pcm
	}
	return pcm[:end]
}

func prefixBytes(rate, ms int) int {
	if rate <= 0 || ms <= 0 {
		return 0
	}
	n := rate * ms / 1000 * 2
	if n < 0 {
		return 0
	}
	return n
}

// GateStep is one PCM chunk through a Gate.
// Drop means the chunk is held or ignored. Lead is quiet audio from before
// speech and has not been forwarded yet. Draft is the utterance when it first
// goes quiet, before SilenceMS has elapsed. Commit is the whole utterance,
// including the silence that ended it. Draft is empty when the same chunk commits.
type GateStep struct {
	Drop    bool
	Barge   bool
	Started bool
	Lead    []byte
	Draft   []byte
	Commit  []byte
	Err     error
	// Quiet means this chunk ended in silence after speech had started.
	Quiet bool
}

// Gate is the turn detector. Rate is PCM16 samples per second.
// Max is the stored utterance cap in bytes. Zero means no cap. Rate zero
// leaves the chunk untouched. Score is optional; a nil or failed score uses
// amplitude. OnReset runs when the gate clears an utterance (commit or barge).
type Gate struct {
	Turn     Turn
	Rate     int
	Max      int
	Score    SpeechScorer
	OnReset  func()
	heard    bool
	quiet    int
	pre, pcm []byte
}

// Reset drops any audio the gate is holding.
func (g *Gate) Reset() {
	g.pcm = nil
	g.pre = nil
	g.heard = false
	g.quiet = 0
	if g.OnReset != nil {
		g.OnReset()
	}
}

// Heard reports whether speech in this utterance has started.
func (g *Gate) Heard() bool {
	return g.heard
}

// Peek copies the open utterance. Empty when speech has not started.
func (g *Gate) Peek() []byte {
	if !g.heard {
		return nil
	}
	return append([]byte(nil), g.pcm...)
}

// SilenceRemainMS is the silence still missing from an open utterance.
// A packet that ends loud has the full SilenceMS left. The second result
// is false when the clock should stay stopped.
func (g *Gate) SilenceRemainMS() (int, bool) {
	if !g.heard || g.Turn.SilenceMS <= 0 || g.Rate <= 0 {
		return 0, false
	}
	if g.quiet <= 0 {
		return g.Turn.SilenceMS, true
	}
	remain := g.Turn.SilenceMS - g.quiet*1000/g.Rate
	if remain < 1 {
		remain = 1
	}
	return remain, true
}

// Take returns held audio and clears the gate. A manual commit uses it.
func (g *Gate) Take() []byte {
	out := append(append([]byte{}, g.pre...), g.pcm...)
	g.Reset()
	return out
}

// Unsent is the quiet prefix that has not been forwarded. Empty once speech
// has started, because those chunks were already sent.
func (g *Gate) Unsent() []byte {
	if g.heard {
		return nil
	}
	return append([]byte(nil), g.pre...)
}

// Push classifies one PCM16 chunk. replying is true while a reply is playing.
// Quiet audio then is ignored. Loud audio is a barge and starts a new utterance.
func (g *Gate) Push(pcm []byte, replying bool) GateStep {
	if g.Rate <= 0 || len(pcm) == 0 {
		return GateStep{}
	}
	samples := len(pcm) / 2
	// While a reply plays, quiet audio is ignored. Amplitude peeks first so
	// neural state is not reset on every quiet packet. A loud packet is a
	// barge: clear recurrent state before scoring that packet.
	if replying {
		if quietSuffix(pcm, g.Turn.Amp()) >= samples {
			return GateStep{Drop: true}
		}
		if g.OnReset != nil {
			g.OnReset()
		}
	}
	last, ok := g.lastSpeech(pcm)
	tail := samples
	if ok && last >= 0 {
		tail = samples - last - 1
	} else if !ok {
		tail = quietSuffix(pcm, g.Turn.Amp())
	}
	loud := tail < samples
	if replying && !loud {
		return GateStep{Drop: true}
	}
	var step GateStep
	if replying && loud {
		step.Barge = true
	}
	var openedQuiet bool
	if !g.heard {
		if !loud {
			g.keepPre(pcm)
			step.Drop = true
			return step
		}
		step.Lead = append([]byte(nil), g.pre...)
		g.pcm = append(append([]byte{}, g.pre...), pcm...)
		g.pre = nil
		g.heard = true
		g.quiet = tail
		step.Started = true
		openedQuiet = tail > 0
	} else {
		if g.Max > 0 && len(g.pcm)+len(pcm) > g.Max {
			g.Reset()
			step.Err = errors.New("input audio is too large")
			return step
		}
		g.pcm = append(g.pcm, pcm...)
		prev := g.quiet
		if loud {
			g.quiet = tail
		} else {
			g.quiet += samples
		}
		openedQuiet = prev == 0 && g.quiet > 0
	}
	need := g.Rate * g.Turn.SilenceMS / 1000
	if g.heard && g.quiet > 0 && g.quiet >= need {
		step.Commit = append([]byte(nil), g.pcm...)
		g.Reset()
		return step
	}
	if openedQuiet {
		step.Draft = append([]byte(nil), g.pcm...)
	}
	step.Quiet = g.quiet > 0
	return step
}

// lastSpeech uses Score when set. ok is false when amplitude should be used.
func (g *Gate) lastSpeech(pcm []byte) (last int, ok bool) {
	if g.Score == nil {
		return -1, false
	}
	last, err := g.Score(pcm, g.Rate, g.Turn.Threshold)
	if err != nil {
		return -1, false
	}
	samples := len(pcm) / 2
	if last < -1 || last >= samples {
		return -1, false
	}
	return last, true
}

func (g *Gate) keepPre(pcm []byte) {
	keep := prefixBytes(g.Rate, g.Turn.PrefixMS)
	g.pre = append(g.pre, pcm...)
	if keep == 0 {
		g.pre = nil
	} else if len(g.pre) > keep {
		g.pre = append([]byte(nil), g.pre[len(g.pre)-keep:]...)
	}
}
