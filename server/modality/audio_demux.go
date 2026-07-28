package modality

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
)

// SGLang #31832: browser MediaRecorder often sends WebM/MP4 as input_audio.
const invalidAudioContainerMessage = "Invalid input_audio: no decodable audio stream was found in the media container."

// audioContainerSignatures match SGLang is_audio_container (MP4/AVI/AMR/WebM).
// RIFF/WAVE is intentionally excluded — already valid AudioFormat wav.
var audioContainerSignatures = [][]struct {
	offset int
	magic  []byte
}{
	{{offset: 4, magic: []byte("ftyp")}},
	{{offset: 0, magic: []byte("RIFF")}, {offset: 8, magic: []byte("AVI ")}},
	{{offset: 0, magic: []byte("#!AMR\n")}},
	{{offset: 0, magic: []byte("#!AMR-WB\n")}},
	{{offset: 0, magic: []byte{0x1a, 0x45, 0xdf, 0xa3}}}, // EBML: WebM / Matroska
}

// IsAudioContainer reports whether data looks like a media container that may
// hold an audio stream (not raw WAV/MP3).
func IsAudioContainer(data []byte) bool {
	for _, sig := range audioContainerSignatures {
		ok := true
		for _, part := range sig {
			end := part.offset + len(part.magic)
			if end > len(data) || !bytes.Equal(data[part.offset:end], part.magic) {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

// ExpandAudioClipsInChatRequest demuxes container-backed AudioClips to WAV.
// Already-sniffed wav/mp3 clips are left unchanged. Native /api/chat and OpenAI
// paths both benefit (OpenAI maps input_audio → AudioClips before this runs).
func ExpandAudioClipsInChatRequest(ctx context.Context, req *api.ChatRequest) error {
	if req == nil {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	for i := range req.Messages {
		clips := req.Messages[i].AudioClips
		if len(clips) == 0 {
			continue
		}
		out := make([]api.AudioData, len(clips))
		for j, clip := range clips {
			decoded, err := normalizeAudioClip(ctx, clip)
			if err != nil {
				return err
			}
			out[j] = decoded
		}
		req.Messages[i].AudioClips = out
	}
	return nil
}

func normalizeAudioClip(ctx context.Context, data []byte) ([]byte, error) {
	if len(data) == 0 {
		return nil, ClientMedia("empty input_audio data")
	}
	if _, ok := llm.AudioFormat(data); ok {
		return data, nil
	}
	if !IsAudioContainer(data) {
		// Leave opaque bytes for runner sniff; not a known container.
		return data, nil
	}
	return demuxAudioContainerToWAV(ctx, data)
}

func demuxAudioContainerToWAV(ctx context.Context, data []byte) ([]byte, error) {
	ffmpeg := envconfig.FFmpegBin()
	if _, err := exec.LookPath(ffmpeg); err != nil {
		return nil, WrapServerMedia(err, "ffmpeg not found for input_audio demux (is ffmpeg installed and on PATH?)")
	}

	in, err := os.CreateTemp("", "ollama-audio-in-*."+sniffVideoExt(data))
	if err != nil {
		return nil, WrapServerMedia(err, "temp input for audio demux")
	}
	inPath := in.Name()
	defer os.Remove(inPath)
	if _, err := in.Write(data); err != nil {
		in.Close()
		return nil, WrapServerMedia(err, "write audio demux input")
	}
	if err := in.Close(); err != nil {
		return nil, WrapServerMedia(err, "close audio demux input")
	}

	out, err := os.CreateTemp("", "ollama-audio-out-*.wav")
	if err != nil {
		return nil, WrapServerMedia(err, "temp output for audio demux")
	}
	outPath := out.Name()
	out.Close()
	defer os.Remove(outPath)

	timeout := envconfig.VideoFFmpegTimeout()
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Mono 16 kHz PCM WAV — matches common ASR / llm.AudioFormat wav sniff.
	args := []string{
		"-hide_banner", "-loglevel", "error", "-y",
		"-i", inPath,
		"-vn",
		"-acodec", "pcm_s16le",
		"-ar", "16000",
		"-ac", "1",
		outPath,
	}
	cmd := exec.CommandContext(ctx, ffmpeg, args...)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg != "" {
			return nil, WrapClientMedia(err, invalidAudioContainerMessage+": "+msg)
		}
		return nil, WrapClientMedia(err, invalidAudioContainerMessage)
	}

	wav, err := os.ReadFile(outPath)
	if err != nil {
		return nil, WrapServerMedia(err, "read demuxed wav")
	}
	if _, ok := llm.AudioFormat(wav); !ok || len(wav) < 44 {
		return nil, ClientMedia(invalidAudioContainerMessage)
	}
	return wav, nil
}
