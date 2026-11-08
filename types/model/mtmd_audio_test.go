package model

import "testing"

func TestClassifyMTMDName(t *testing.T) {
	rt := ClassifyMTMDName("voxtral_rt")
	if !rt.Stream || !rt.StreamOnly || !rt.SingleGPU || rt.FileAudio {
		t.Fatalf("voxtral_rt: %+v", rt)
	}
	lfm := ClassifyMTMDName("lfm3a")
	if !lfm.Stream || !lfm.FileAudio || lfm.StreamOnly || !lfm.SingleGPU {
		t.Fatalf("lfm3a: %+v", lfm)
	}
	if got := ClassifyMTMDName("voxtral"); !got.FileAudio || got.Stream || got.SingleGPU {
		t.Fatalf("voxtral offline: %+v", got)
	}
	for _, name := range []string{"qwen2.5o", "ultravox", "qwen2a", "qwen3a", "gemma4a", "lfm2a"} {
		got := ClassifyMTMDName(name)
		if !got.FileAudio || !got.Audio || got.Stream {
			t.Fatalf("%s: %+v", name, got)
		}
	}
	phi := ClassifyMTMDName("phi4")
	if !phi.FileAudio || !phi.Vision {
		t.Fatalf("phi4: %+v", phi)
	}
	if ClassifyMTMDName("mistral3").Audio || ClassifyMTMDName("lfm3").Stream {
		t.Fatal("text architectures are not projectors")
	}
	spoken := ClassifyMTMDName("lfm3a").Merge(ClassifyMTMDName("lfm3a_gen"))
	if !spoken.Speech || !spoken.Stream || !spoken.FileAudio || spoken.StreamOnly {
		t.Fatalf("lfm3 speech: %+v", spoken)
	}
	for _, name := range []string{"qwen3tts_gen", "pockettts_gen", "soprano", "drama3_gen", "chatterbox"} {
		if !ClassifyMTMDName(name).Speech {
			t.Fatalf("%s should speak", name)
		}
	}
	omni := ClassifyMTMDName("qwen3a").Merge(ClassifyMTMDName("qwen3vl_merger"))
	if !omni.FileAudio || !omni.Vision || !omni.QwenVL || omni.SingleGPU || omni.Stream || omni.ReasoningOff() {
		t.Fatalf("qwen3-omni: %+v", omni)
	}
}

func TestSpeechBlock(t *testing.T) {
	if got := SpeechBlock(ClassifyMTMDName("voxtral_rt")); got != "this model serves live audio on GET /v1/realtime" {
		t.Fatal(got)
	}
	if got := SpeechBlock(ClassifyMTMDName("voxtral")); got != "this model takes a whole clip; use POST /v1/audio/transcriptions" {
		t.Fatal(got)
	}
	if got := SpeechBlock(ClassifyMTMDName("qwen3a")); got != "this model takes a whole clip; use POST /v1/audio/transcriptions" {
		t.Fatal(got)
	}
	if got := SpeechBlock(ClassifyMTMDName("soprano")); got != "" {
		t.Fatal(got)
	}
	if got := SpeechBlock(ClassifyMTMDName("lfm3a").Merge(ClassifyMTMDName("lfm3a_gen"))); got != "" {
		t.Fatal(got)
	}
	if got := SpeechBlock(MTMDRole{}); got != "" {
		t.Fatal(got)
	}
	if !HostSpeechOnly(BackendPiper, []Capability{CapabilitySpeech}) || !HostSpeechOnly(BackendKokoro, []Capability{CapabilitySpeech}) || !HostSpeechOnly(BackendRemoteTTS, nil) || !HostSpeechOnly(BackendMusic3, []Capability{CapabilitySpeech}) {
		t.Fatal("host speech")
	}
	if HostSpeechOnly(BackendPiper, []Capability{CapabilitySpeech, CapabilityCompletion}) || HostSpeechOnly(BackendWhisper, []Capability{CapabilityAudio}) || HostSpeechOnly("", []Capability{CapabilitySpeech}) {
		t.Fatal("not host speech only")
	}
	if !HostTranscribeOnly(BackendWhisper, nil) || !HostTranscribeOnly(BackendWhisper, []Capability{CapabilityAudio}) {
		t.Fatal("host transcribe")
	}
	if !HostTranscribeOnly(BackendQwen3ASR, nil) || !HostTranscribeOnly(BackendQwen3ASR, []Capability{CapabilityAudio}) {
		t.Fatal("host qwen3-asr")
	}
	if HostTranscribeOnly(BackendWhisper, []Capability{CapabilityCompletion}) || HostTranscribeOnly(BackendPiper, nil) {
		t.Fatal("not host transcribe only")
	}
}

func TestRealtimeBlock(t *testing.T) {
	if got := RealtimeBlock(ClassifyMTMDName("voxtral_rt")); got != "" {
		t.Fatal(got)
	}
	if got := RealtimeBlock(ClassifyMTMDName("lfm3a").Merge(ClassifyMTMDName("lfm3a_gen"))); got != "" {
		t.Fatal(got)
	}
	if got := RealtimeBlock(ClassifyMTMDName("voxtral")); got != "this model takes a whole clip; use POST /v1/audio/transcriptions" {
		t.Fatal(got)
	}
	if got := RealtimeBlock(ClassifyMTMDName("qwen3tts_gen")); got != "" {
		t.Fatal(got)
	}
	if !ClassifyMTMDName("soprano").SpeechOnly() || ClassifyMTMDName("lfm3a").Merge(ClassifyMTMDName("lfm3a_gen")).SpeechOnly() {
		t.Fatal("speech only")
	}
	if got := RealtimeBlock(ClassifyMTMDName("chatterbox")); got != "" {
		t.Fatal(got)
	}
}

func TestTranslationRoute(t *testing.T) {
	vox := ClassifyMTMDName("voxtral")
	if !FileTranscriptionOnLlama(vox, false, false) || FileTranscriptionOnLlama(vox, false, true) {
		t.Fatal("voxtral")
	}
	if TranslationRejected(false, true) || !TranslationRejected(true, true) || TranslationRejected(true, false) {
		t.Fatal("reject")
	}
	if FileTranscriptionOnLlama(vox, true, true) || !FileTranscriptionOnLlama(ClassifyMTMDName("lfm3a"), false, false) {
		t.Fatal("asr")
	}
}

func TestMTMDReasoningOff(t *testing.T) {
	role := ClassifyMTMDName("lfm3a").Merge(ClassifyMTMDName("lfm3vl"))
	if !role.ReasoningOff() || !role.Stream || !role.Vision {
		t.Fatalf("omni: %+v", role)
	}
	if ClassifyMTMDName("lfm3vl").ReasoningOff() || ClassifyMTMDName("lfm3a").ReasoningOff() {
		t.Fatal("one tower is not omni")
	}
}

func TestClassifyMTMDKV(t *testing.T) {
	if !ClassifyMTMDKV("clip.projector_type", "voxtral_rt").StreamOnly {
		t.Fatal("clip projector")
	}
	if !ClassifyMTMDKV("clip.audio.projector_type", "lfm3a").LFM3A {
		t.Fatal("audio projector")
	}
	if !ClassifyMTMDKV("clip.vision.projector_type", "lfm3vl").LFM3VL {
		t.Fatal("vision projector")
	}
	if !ClassifyMTMDKV("clip.gen.audio.projector_type", "lfm3a_gen").Speech {
		t.Fatal("gen projector")
	}
	if ClassifyMTMDKV("general.description", "voxtral").Audio {
		t.Fatal("unrelated key")
	}
}

func TestGenVoiceNames(t *testing.T) {
	lfm := GenVoiceNames("lfm3a_gen", nil)
	if len(lfm) != 2 || lfm[0] != "female" || lfm[1] != "male" {
		t.Fatalf("lfm3 voices %v", lfm)
	}
	got := GenVoiceNames("chatterbox", []string{
		"a.gen.code.out_embd.weight",
		"voice.luna.ref_feat",
		"voice.luna.ref_spk",
		"voice.luna.ref_tokens",
		"voice.sol",
	})
	if len(got) != 2 || got[0] != "luna" || got[1] != "sol" {
		t.Fatalf("voices %v", got)
	}
	one := GenVoiceNames("voxtral", []string{"voice.x"})
	if len(one) != 1 || one[0] != "x" {
		t.Fatalf("voice tensor %v", one)
	}
}

func TestGenAudioSampleRate(t *testing.T) {
	if GenAudioSampleRate("lfm3a_gen") != 24000 || GenAudioSampleRate("Qwen3TTS_Gen") != 24000 {
		t.Fatal("expected 24000")
	}
	if GenAudioSampleRate("soprano") != 32000 || GenAudioSampleRate("drama3_gen") != 44100 {
		t.Fatal("expected soprano 32000 and drama3 44100")
	}
	if GenAudioSampleRate("voxtral") != 0 {
		t.Fatal("expected no rate")
	}
}

func TestIsRealtimeStreamOnly(t *testing.T) {
	if !IsRealtimeStreamOnly(" Voxtral_RT ") {
		t.Fatal("expected stream only")
	}
	if IsRealtimeStreamOnly("voxtral") || IsRealtimeStreamOnly("lfm3a") {
		t.Fatal("file-capable models are not stream-only")
	}
}
