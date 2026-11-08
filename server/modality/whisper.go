package modality

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/modality/whispercli"
	"github.com/ollama/ollama/types/model"
)

// TranscribeWhisper runs a whisper.cpp-compatible CLI. Expected invocation shape:
//
//	<bin> -m <model.ggml> -f <audio.ext> -otxt -of <outbase> [-l <lang>] [--prompt <text>] [--temperature <0..1>] [-oj]
//
// The input file extension is chosen from the original upload name and/or magic bytes.
// An empty prompt or temperature is left off the command so the CLI keeps its defaults.
// verbose asks for segment offsets from the JSON file.
// words, and verbose_json from the handler, ask for -ojf so token probabilities are present.
// words also passes --split-on-word so segment text breaks on words.
// translate asks whisper.cpp for English text.
// diarize asks whisper.cpp to label a stereo clip by channel energy.
// suppressNST asks whisper.cpp to keep non-speech marks out of the text.
// noFallback asks whisper.cpp to keep a failed segment at the requested temperature.
// temperatureInc is the step added after a failed segment. Empty leaves 0.2.
// entropyThold is the token-entropy cutoff. A segment below it fails. Empty leaves 2.4.
// logprobThold is the average log-probability cutoff. A segment below it fails when the clip still looks like speech. Empty leaves -1.
// noSpeechThold is the silence cutoff. A higher no-speech probability marks the segment as silence when the log probability is still low. Empty leaves 0.6.
// beamSize is the number of beam-search candidates when the temperature is 0. Empty leaves 5. A value of 1 decodes greedily.
// bestOf is the number of candidates when the temperature is above 0. Empty leaves 5.
// carryInitial keeps the initial prompt in front of every window.
// maxLen is the maximum characters in a segment. Empty leaves the cap off.
// wordThold is the token-timestamp probability. A timestamp above it is kept. Empty leaves 0.01.
// maxContext is the cap on earlier text tokens for the next window. Empty leaves the library cap. 0 leaves that text out.
// audioCtx is the number of encoder frames in each window. Empty uses the full context.
// offsetT is milliseconds into the clip before transcription starts. Empty starts at the beginning.
// durationMS is milliseconds to transcribe from that start. Empty continues to the end.
// suppressRegex is a regular expression matched against each whole token. Empty leaves every token available.

// VADModelPath is the Silero VAD file for a whisper model.
// backend_paths.vad_model wins. OLLAMA_WHISPER_VAD_MODEL is the fallback.
func VADModelPath(cfg model.ConfigV2) string {
	if p := strings.TrimSpace(PathFor(cfg, "vad_model")); p != "" {
		return p
	}
	return strings.TrimSpace(envconfig.WhisperVADModel())
}

func TranscribeWhisper(ctx context.Context, cfg model.ConfigV2, audio []byte, originalFilename, language, prompt, temperature, temperatureInc, entropyThold, logprobThold, noSpeechThold, beamSize, bestOf, maxLen, wordThold, maxContext, audioCtx, offsetT, durationMS, suppressRegex, grammar, grammarRule, grammarPenalty, dtw, vadModel, vadThreshold, vadMinSpeechMS, vadMinSilenceMS, vadMaxSpeechS, vadSpeechPadMS, vadSamplesOverlap string, verbose, words, translate, diarize, suppressNST, noFallback, carryInitial, noTimestamps, printSpecial, detectLanguage, tinyDiarize, splitOnWord bool) (whispercli.Transcript, error) {
	ctx, cancel := whisperCtx(ctx)
	defer cancel()

	bin := envconfig.WhisperBin()
	modelPath := PathFor(cfg, "whisper_model")
	if modelPath == "" {
		modelPath = envconfig.WhisperModelPath()
	}
	if modelPath == "" {
		return whispercli.Transcript{}, fmt.Errorf("set backend_paths.whisper_model in the model config or OLLAMA_WHISPER_MODEL")
	}

	ext := AudioInputExt(originalFilename, audio)

	tmpDir, err := os.MkdirTemp("", "ollama-whisper-*")
	if err != nil {
		return whispercli.Transcript{}, err
	}
	defer os.RemoveAll(tmpDir)

	inPath := filepath.Join(tmpDir, "input."+ext)
	if err := os.WriteFile(inPath, audio, 0o600); err != nil {
		return whispercli.Transcript{}, err
	}
	outBase := filepath.Join(tmpDir, "out")
	args := whispercli.Args(modelPath, inPath, outBase, language, prompt, temperature, temperatureInc, entropyThold, logprobThold, noSpeechThold, beamSize, bestOf, maxLen, wordThold, maxContext, audioCtx, offsetT, durationMS, suppressRegex, grammar, grammarRule, grammarPenalty, dtw, vadModel, vadThreshold, vadMinSpeechMS, vadMinSilenceMS, vadMaxSpeechS, vadSpeechPadMS, vadSamplesOverlap, verbose, words, translate, diarize, suppressNST, noFallback, carryInitial, noTimestamps, printSpecial, detectLanguage, tinyDiarize, splitOnWord)
	if extra := envconfig.WhisperExtraArgs(); extra != "" {
		args = append(args, strings.Fields(extra)...)
	}
	cmd := exec.CommandContext(ctx, bin, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return whispercli.Transcript{}, fmt.Errorf("whisper: deadline exceeded or cancelled: %w", ctx.Err())
		}
		return whispercli.Transcript{}, fmt.Errorf("whisper (%s): %w: %s", bin, err, stderr.String())
	}
	txtData, err := os.ReadFile(outBase + ".txt")
	if err != nil {
		return whispercli.Transcript{}, fmt.Errorf("read whisper output: %w", err)
	}
	out := whispercli.Transcript{Text: strings.TrimSpace(string(txtData))}
	if !verbose && !words {
		applyDetectedLanguage(&out, stderr.String())
		return out, nil
	}
	raw, err := os.ReadFile(outBase + ".json")
	if err != nil {
		return whispercli.Transcript{}, fmt.Errorf("read whisper json: %w", err)
	}
	parsed, err := whispercli.ParseJSON(raw)
	if err != nil {
		return whispercli.Transcript{}, fmt.Errorf("whisper json: %w", err)
	}
	if parsed.Text != "" {
		out.Text = parsed.Text
	}
	out.Language = parsed.Language
	out.Segments = parsed.Segments
	out.Words = parsed.Words
	out.Logprobs = parsed.Logprobs
	applyDetectedLanguage(&out, stderr.String())
	return out, nil
}

func applyDetectedLanguage(out *whispercli.Transcript, stderr string) {
	code, p := whispercli.DetectedLanguage(stderr)
	out.DetectedLanguage = code
	if p != nil && (out.Language == "" || out.Language == code) {
		out.LanguageProbability = p
	}
	if out.Language == "" {
		out.Language = code
	}
}
