package model

import (
	"sort"
	"strings"
)

// Projector type strings from this pin's llama.cpp mtmd
// (llama/llama.cpp/tools/mtmd/clip-impl.h). The text GGUF architecture is often
// mistral3 or lfm3; the mmproj carries the type that selects the graph.
const (
	ProjVoxtralRT = "voxtral_rt" // live socket only
	ProjLFM3A     = "lfm3a"      // live socket and file audio
	ProjLFM3VL    = "lfm3vl"     // LFM3-Omni / LFM3-VL vision tower
	ProjVoxtral   = "voxtral"    // Voxtral Small / Mini offline
	ProjQwen25O   = "qwen2.5o"
	ProjUltravox  = "ultravox"
	ProjQwen2A    = "qwen2a"
	ProjQwen3A    = "qwen3a"
	ProjGemma4A   = "gemma4a"
	ProjLFM2A     = "lfm2a"
	ProjPhi4      = "phi4" // Phi-4 multimodal: vision and audio
	// Vision mergers that sit on the same mmproj as an audio tower.
	// Qwen3-Omni publishes clip.vision.projector_type=qwen3vl_merger
	// and clip.audio.projector_type=qwen3a.
	ProjQwen2VL  = "qwen2vl_merger"
	ProjQwen25VL = "qwen2.5vl_merger"
	ProjQwen3VL  = "qwen3vl_merger"
	// Speech projectors. llama-server POST /tts speaks these. The strings are
	// clip projector types, including the GGUF chatterbox tower.
	ProjLFM3AGen     = "lfm3a_gen"
	ProjQwen3TTSGen  = "qwen3tts_gen"
	ProjPocketTTSGen = "pockettts_gen"
	ProjSoprano      = "soprano"
	ProjDrama3Gen    = "drama3_gen"
	ProjChatterbox   = "chatterbox"
)

// MTMDRole is what one GGUF metadata value says about llama.cpp mtmd audio.
// Merge values from the text file and the mmproj before reading ReasoningOff.
type MTMDRole struct {
	Audio      bool // encoder llama-server must load with --mmproj
	Stream     bool // GET /v1/realtime
	StreamOnly bool // voxtral_rt: no file transcription and no text chat
	FileAudio  bool // POST /v1/audio/transcriptions via llama-server
	Vision     bool
	LFM3A      bool
	LFM3VL     bool
	// QwenVL is a Qwen vision merger. The text GGUF may be a MoE, so the
	// image-token floor cannot be inferred from general.architecture alone.
	QwenVL bool
	// Speech is a gen-audio projector (lfm3a_gen, qwen3tts_gen, and the other
	// towers this pin's llama-server speaks). POST /v1/audio/speech uses it.
	Speech bool
	// SingleGPU keeps a realtime model on one device (--split-mode none).
	// Offline 24B audio stays on llama-server's default layer split so a
	// quant that does not fit in 24 GiB can use both GPUs. PARAMETER main_gpu
	// still pins that model to one card.
	SingleGPU bool
}

// Merge unions two observations. Call ReasoningOff on the result.
func (r MTMDRole) Merge(o MTMDRole) MTMDRole {
	r.Audio = r.Audio || o.Audio
	r.Stream = r.Stream || o.Stream
	r.StreamOnly = r.StreamOnly || o.StreamOnly
	r.FileAudio = r.FileAudio || o.FileAudio
	r.Vision = r.Vision || o.Vision
	r.LFM3A = r.LFM3A || o.LFM3A
	r.LFM3VL = r.LFM3VL || o.LFM3VL
	r.QwenVL = r.QwenVL || o.QwenVL
	r.Speech = r.Speech || o.Speech
	r.SingleGPU = r.SingleGPU || o.SingleGPU
	return r
}

// SpeechOnly reports a gen projector with no live or file-audio path.
// Chat, generate, and transcription should use POST /v1/audio/speech instead.
func (r MTMDRole) SpeechOnly() bool {
	return r.Speech && !r.Stream && !r.FileAudio
}

// HostSpeechOnly reports a config-only speech tag: Piper, kokoro, remote-tts, or
// Music 3, with no other capability. Those tags have no mtmd projector, so
// SpeechOnly on an empty role is false. A completion or audio capability
// keeps the text and transcription routes open.
func HostSpeechOnly(speechBackend string, caps []Capability) bool {
	switch speechBackend {
	case BackendPiper, BackendKokoro, BackendRemoteTTS, BackendMusic3:
	default:
		return false
	}
	for _, c := range caps {
		if c != CapabilitySpeech {
			return false
		}
	}
	return true
}

// HostTranscribeOnly reports a config-only Whisper or Qwen3-ASR tag. An audio
// capability is the transcribe capability. Any other capability keeps chat open.
func HostTranscribeOnly(transcribeBackend string, caps []Capability) bool {
	if transcribeBackend != BackendWhisper && transcribeBackend != BackendQwen3ASR {
		return false
	}
	for _, c := range caps {
		if c != CapabilityAudio {
			return false
		}
	}
	return true
}

// SpeechBlock is the 400 reason for a model that cannot speak on
// POST /v1/audio/speech. An empty string means Piper, remote-tts, Music 3,
// or a gen projector may continue. A live-only model stays on the socket.
// A file-audio model without a gen tower stays on transcription.
func SpeechBlock(role MTMDRole) string {
	if role.StreamOnly {
		return "this model serves live audio on GET /v1/realtime"
	}
	if role.FileAudio && !role.Speech {
		return "this model takes a whole clip; use POST /v1/audio/transcriptions"
	}
	return ""
}

// RealtimeBlock is the 400 reason for a model that cannot open GET /v1/realtime.
// An empty string means the socket is allowed. A file-audio model is sent to
// POST /v1/audio/transcriptions. A gen-only projector speaks on this socket.
func RealtimeBlock(role MTMDRole) string {
	if role.Stream {
		return ""
	}
	if role.FileAudio {
		return "this model takes a whole clip; use POST /v1/audio/transcriptions"
	}
	return ""
}

// FileTranscriptionOnLlama reports the llama-server file transcription route.
// A translation of a file-audio chat model stays on chat. That route does not translate.
// A dedicated llama-asr family is never asked to translate.
func FileTranscriptionOnLlama(role MTMDRole, llamaASR, translate bool) bool {
	if llamaASR {
		return !translate
	}
	return role.FileAudio && !translate
}

// TranslationRejected reports a dedicated llama-asr model on POST /v1/audio/translations.
func TranslationRejected(llamaASR, translate bool) bool {
	return translate && llamaASR
}

// ReasoningOff reports LFM3-Omni: the audio and vision towers together.
// The chat template has enable_thinking, and that checkpoint is not trained
// to fill it. llama-server needs --reasoning off.
func (r MTMDRole) ReasoningOff() bool {
	return r.LFM3A && r.LFM3VL
}

// ClassifyMTMDName maps a projector type or architecture string.
func ClassifyMTMDName(name string) MTMDRole {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case ProjVoxtralRT:
		return MTMDRole{Audio: true, Stream: true, StreamOnly: true, SingleGPU: true}
	case ProjLFM3A:
		return MTMDRole{Audio: true, Stream: true, FileAudio: true, LFM3A: true, SingleGPU: true}
	case ProjLFM3VL:
		return MTMDRole{Vision: true, LFM3VL: true}
	case ProjVoxtral, ProjQwen25O, ProjUltravox, ProjQwen2A, ProjQwen3A, ProjGemma4A, ProjLFM2A:
		return MTMDRole{Audio: true, FileAudio: true}
	case ProjPhi4:
		return MTMDRole{Audio: true, FileAudio: true, Vision: true}
	case ProjQwen2VL, ProjQwen25VL, ProjQwen3VL:
		return MTMDRole{Vision: true, QwenVL: true}
	case ProjLFM3AGen, ProjQwen3TTSGen, ProjPocketTTSGen, ProjSoprano, ProjDrama3Gen, ProjChatterbox:
		return MTMDRole{Speech: true}
	default:
		return MTMDRole{}
	}
}

// GenVoiceNames lists the voice ids llama-server will accept for a gen projector.
// LFM3-Audio hardcodes female and male. Other towers store voice.<id> tensors,
// with chatterbox-style voice.<id>.ref_feat / .ref_spk / .ref_tokens suffixes.
func GenVoiceNames(projector string, tensorNames []string) []string {
	seen := map[string]struct{}{}
	var out []string
	add := func(id string) {
		id = strings.TrimSpace(id)
		if id == "" {
			return
		}
		if _, ok := seen[id]; ok {
			return
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	if strings.EqualFold(strings.TrimSpace(projector), ProjLFM3AGen) {
		add("female")
		add("male")
	}
	const prefix = "voice."
	suffixes := []string{".ref_tokens", ".ref_feat", ".ref_spk"}
	for _, name := range tensorNames {
		if !strings.HasPrefix(name, prefix) {
			continue
		}
		id := strings.TrimPrefix(name, prefix)
		for _, suf := range suffixes {
			if strings.HasSuffix(id, suf) && len(id) > len(suf) {
				id = strings.TrimSuffix(id, suf)
				break
			}
		}
		add(id)
	}
	sort.Strings(out)
	return out
}

// GenAudioSampleRate is the PCM rate llama-server uses for a gen projector.
// qwen3tts, pockettts, lfm3a_gen, and chatterbox are 24000. soprano is 32000.
// drama3 is 44100. Other names return 0. The bytes are little-endian float32.
func GenAudioSampleRate(projector string) int {
	switch strings.ToLower(strings.TrimSpace(projector)) {
	case ProjQwen3TTSGen, ProjPocketTTSGen, ProjLFM3AGen, ProjChatterbox:
		return 24000
	case ProjSoprano:
		return 32000
	case ProjDrama3Gen:
		return 44100
	default:
		return 0
	}
}

// IsRealtimeStreamOnly reports a model that only serves GET /v1/realtime.
func IsRealtimeStreamOnly(name string) bool {
	return ClassifyMTMDName(name).StreamOnly
}

// ClassifyMTMDKV maps one GGUF metadata pair. Keys are the literal mtmd
// strings (clip.projector_type, clip.audio.projector_type,
// clip.vision.projector_type) plus general.architecture.
func ClassifyMTMDKV(key, value string) MTMDRole {
	k := strings.ToLower(strings.TrimSpace(key))
	switch {
	case k == "general.architecture",
		k == "clip.projector_type",
		k == "clip.audio.projector_type",
		k == "clip.vision.projector_type",
		strings.HasSuffix(k, ".projector_type"):
		return ClassifyMTMDName(value)
	default:
		return MTMDRole{}
	}
}
