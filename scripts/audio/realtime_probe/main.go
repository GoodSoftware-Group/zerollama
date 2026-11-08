// Command realtime_probe drives zerollama's audio routes and prints how late
// the first words or speech samples were.
//
// Realtime sends a wav to GET /v1/realtime. The wav is resampled to the rate
// in session.created before any input_audio_buffer.append. File mode posts
// the same wav to POST /v1/audio/transcriptions.
//
//	go run ./scripts/audio/realtime_probe -model voxtral-realtime -wav clip.wav
//	go run ./scripts/audio/realtime_probe -model voxtral-realtime -wav clip.wav -turn server_vad -lang en
//	go run ./scripts/audio/realtime_probe -model llama3.2 -pipeline duplex -tts piper -asr whisper -wav clip.wav
//	go run ./scripts/audio/realtime_probe -model llama3.2 -pipeline duplex -tts piper -asr whisper -wav clip.wav -turn server_vad
//	go run ./scripts/audio/realtime_probe -model lfm3-audio -intent "" -text "hello" -modalities audio -out reply.wav
//	go run ./scripts/audio/realtime_probe -mode transcribe -model voxtral-small -wav clip.wav
//
// The probe only talks to a server that is already running. It does not start one.
package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	host := flag.String("host", "http://127.0.0.1:11434", "zerollama base URL")
	model := flag.String("model", "", "model tag")
	wavPath := flag.String("wav", "", "PCM wav to send")
	textIn := flag.String("text", "", "conversation text turn, or the speech input")
	lang := flag.String("lang", "", "language tag: TTS template, the live transcription language query, and file transcription")
	voice := flag.String("voice", "", "speech voice id, or the duplex speaker id")
	speakerRef := flag.String("speaker-ref", "", "wav or other audio to clone, for a speech model that takes a reference")
	streamSpeech := flag.Bool("stream", false, "flush speech audio, or transcription events, as they arrive")
	seedFlag := flag.String("seed", "", "speech RNG seed; omit or use a negative value for the server default")
	maxTokensFlag := flag.String("max-tokens", "", "speech max audio frames, forwarded as max_new_tokens (-1 means no cap); transcribe or translate token cap, a positive integer")
	topKFlag := flag.String("top-k", "", "speech acoustic top_k; 0 uses the model default")
	topPFlag := flag.String("top-p", "", "speech acoustic top_p, from 0 to 1; 0 uses the model default")
	repeatFlag := flag.String("repeat-penalty", "", "speech backbone repeat_penalty; omit to leave 1.05")
	sentenceSilenceFlag := flag.String("sentence-silence", "", "piper seconds of silence after each sentence, from 0 to 5; omitted stays 0.2")
	noiseScaleFlag := flag.String("noise-scale", "", "piper generator noise, from 0 to 2; 0 is steady; omitted leaves the voice JSON")
	noiseWFlag := flag.String("noise-w", "", "piper phoneme-width noise, from 0 to 2; 0 keeps phoneme lengths steady; omitted leaves the voice JSON")
	speakerIDFlag := flag.String("speaker-id", "", "piper speaker index, from 0 to 4095; 0 or omitted leaves the default speaker")
	phonemeSilenceFlag := flag.String("phoneme-silence", "", "piper JSON object of one character to seconds of silence, from 0 to 5")
	mode := flag.String("mode", "realtime", "realtime, transcribe, translate, or speech")
	intent := flag.String("intent", "transcription", "realtime intent; empty opens a conversation")
	turnFlag := flag.String("turn", "", "realtime turn taking: server_vad commits on silence on the server; empty or none commits from this client")
	pipelineFlag := flag.String("pipeline", "", "realtime pipeline: duplex chats with -model, speaks with -tts, and transcribes with -asr")
	asrFlag := flag.String("asr", "", "duplex transcriber model tag")
	ttsFlag := flag.String("tts", "", "duplex speech model tag")
	rateFlag := flag.String("rate", "", "duplex input pcm16 rate; omitted uses the wav rate")
	modalities := flag.String("modalities", "", "conversation output: text, audio, or text,audio; empty lets the model choose")
	barge := flag.Bool("barge", false, "send response.cancel when the first reply byte arrives")
	instructions := flag.String("instructions", "", "realtime session.update system instructions, sent before the first turn")
	toolsFlag := flag.String("tools", "", "realtime tools: a JSON array, or a path to one, in the Responses API shape")
	toolChoice := flag.String("tool-choice", "", "realtime tool_choice: auto, none, or required")
	maxOut := flag.String("max-output-tokens", "", "realtime max_output_tokens: a positive integer or inf")
	toolOutput := flag.String("tool-output", "", "realtime text sent as function_call_output after a tool call, then response.create")
	follow := flag.String("follow", "", "realtime user text sent after the reply, then another response.create")
	followWav := flag.String("follow-wav", "", "realtime PCM wav spoken after the reply; with -follow, the text is sent first")
	promptFlag := flag.String("prompt", "", "file transcription prompt forwarded with the audio")
	transcribeTemp := flag.String("temperature", "", "file transcription temperature, from 0 to 1; omitted stays 0")
	formatFlag := flag.String("format", "json", "transcribe response_format (json, text, srt, verbose_json, vtt, diarized_json), or speech response_format (wav, pcm, mp3, opus, aac, flac); the speech default omits it")
	wordsFlag := flag.Bool("words", false, "transcribe or translate verbose_json word times")
	logprobsFlag := flag.Bool("logprobs", false, "transcribe or translate: ask for whisper token logprobs")
	diarizeFlag := flag.Bool("diarize", false, "transcribe or translate: ask whisper.cpp to label a stereo clip")
	suppressNSTFlag := flag.Bool("suppress-nst", false, "transcribe or translate: ask whisper.cpp to drop non-speech marks")
	noFallbackFlag := flag.Bool("no-fallback", false, "transcribe or translate: keep whisper.cpp at the requested temperature")
	tempIncFlag := flag.String("temperature-inc", "", "transcribe or translate: whisper.cpp temperature step after a failed segment, from 0 to 1; omitted stays 0.2")
	entropyFlag := flag.String("entropy-thold", "", "transcribe or translate: whisper.cpp token-entropy cutoff, from 0 to 4; a segment below it fails; omitted stays 2.4")
	logprobFlag := flag.String("logprob-thold", "", "transcribe or translate: whisper.cpp average log-probability cutoff, from -10 to 0; a segment below it fails when speech is still likely; omitted stays -1")
	noSpeechFlag := flag.String("no-speech-thold", "", "transcribe or translate: whisper.cpp silence cutoff, from 0 to 1; a higher no-speech probability marks silence when the log probability is still low; omitted stays 0.6")
	beamFlag := flag.String("beam-size", "", "transcribe or translate: whisper.cpp beam width at temperature 0, from 1 to 16; 1 decodes greedily; omitted stays 5")
	bestOfFlag := flag.String("best-of", "", "transcribe or translate: whisper.cpp candidate count when the temperature is above 0, from 1 to 16; omitted stays 5")
	carryFlag := flag.Bool("carry-initial-prompt", false, "transcribe or translate: keep the whisper.cpp prompt in front of every window")
	maxLenFlag := flag.String("max-len", "", "transcribe or translate: whisper.cpp maximum segment length in characters, from 1 to 1000; 0 or omitted leaves the cap off")
	wordTholdFlag := flag.String("word-thold", "", "transcribe or translate: whisper.cpp token-timestamp probability, from 0 to 1; a timestamp above it is kept; omitted stays 0.01")
	maxContextFlag := flag.String("max-context", "", "transcribe or translate: whisper.cpp cap on earlier text tokens, from -1 to 16384; -1 or omitted leaves the library cap; 0 leaves earlier text out")
	audioCtxFlag := flag.String("audio-ctx", "", "transcribe or translate: whisper.cpp encoder frames per window, from 1 to 1500; 0 or omitted uses the full context")
	offsetTFlag := flag.String("offset-t", "", "transcribe or translate: whisper.cpp start offset in milliseconds, from 1 to 86400000; 0 or omitted starts at the beginning")
	durationMSFlag := flag.String("duration-ms", "", "transcribe or translate: whisper.cpp milliseconds to transcribe from the start offset; 0 or omitted continues to the end")
	suppressRegexFlag := flag.String("suppress-regex", "", "transcribe or translate: whisper.cpp regular expression matched against each whole token; a match is left out")
	offsetNFlag := flag.String("offset-n", "", "transcribe or translate: added to each SRT cue number; 0 or omitted starts at 1")
	noTimestampsFlag := flag.Bool("no-timestamps", false, "transcribe or translate: whisper.cpp decodes without timestamp tokens; each segment advances 30 seconds")
	grammarFlag := flag.String("grammar", "", "transcribe or translate: whisper.cpp GBNF grammar; requires -grammar-rule")
	grammarRuleFlag := flag.String("grammar-rule", "", "transcribe or translate: GBNF rule decoding starts from")
	grammarPenaltyFlag := flag.String("grammar-penalty", "", "transcribe or translate: logit points subtracted from tokens the grammar rejects, from 0 to 1000; omitted stays at 100; 0 leaves those logits unchanged")
	printSpecialFlag := flag.Bool("print-special", false, "transcribe or translate: whisper.cpp keeps special tokens in the segment text")
	detectLanguageFlag := flag.Bool("detect-language", false, "transcribe or translate: whisper.cpp reports the spoken language and skips the transcript")
	dtwFlag := flag.String("dtw", "", "transcribe or translate: whisper.cpp DTW alignment preset matching the model; turns flash attention off so word starts use DTW times")
	vadFlag := flag.Bool("vad", false, "transcribe or translate: whisper.cpp keeps speech and drops the rest, using the server Silero VAD model")
	vadThresholdFlag := flag.String("vad-threshold", "", "transcribe or translate: Silero probability that starts a speech region, from 0 to 1; omitted stays at 0.5; requires -vad")
	vadMinSpeechFlag := flag.String("vad-min-speech-duration-ms", "", "transcribe or translate: drop a speech region of this many milliseconds or shorter; omitted stays at 250; 0 keeps any length; requires -vad")
	vadMinSilenceFlag := flag.String("vad-min-silence-duration-ms", "", "transcribe or translate: a shorter pause stays inside the speech region; omitted stays at 100; 0 ends on the first quiet frame; gaps under 200 ms are joined; requires -vad")
	vadMaxSpeechFlag := flag.String("vad-max-speech-duration-s", "", "transcribe or translate: split a speech region longer than this many seconds; omitted leaves the cap off; requires -vad")
	vadSpeechPadFlag := flag.String("vad-speech-pad-ms", "", "transcribe or translate: grow each speech region by this many milliseconds on each side; omitted stays at 30; 0 adds no padding; requires -vad")
	vadOverlapFlag := flag.String("vad-samples-overlap", "", "transcribe or translate: extra seconds the decoder hears after each speech region except the last; omitted stays at 0.1; 0 copies the region as detected; requires -vad")
	tinyDiarizeFlag := flag.Bool("tinydiarize", false, "transcribe or translate: a tinydiarize model may end a segment with \" [SPEAKER_TURN]\"; cannot be combined with -diarize")
	splitOnWordFlag := flag.Bool("split-on-word", false, "transcribe or translate: a max_len wrap waits for a token that starts with a space; requires -max-len")
	outPath := flag.String("out", "", "write spoken PCM to this wav, at the session output rate")
	chunkMs := flag.Int("chunk-ms", 320, "append chunk length at the session rate")
	timeout := flag.Duration("timeout", 2*time.Minute, "wait for the transcript or reply")
	flag.Parse()
	if *model == "" || (*wavPath == "" && strings.TrimSpace(*textIn) == "") {
		fmt.Fprintln(os.Stderr, "usage: realtime_probe -model <tag> (-wav <clip.wav> | -text <message>) [-mode realtime|transcribe|speech]")
		os.Exit(2)
	}
	var samples []int16
	rate := 0
	if *wavPath != "" {
		f, err := os.Open(*wavPath)
		if err != nil {
			fatal(err)
		}
		defer f.Close()
		samples, rate, err = wavPCM16(f)
		if err != nil {
			fatal(err)
		}
	}
	var followSamples []int16
	followRate := 0
	if *followWav != "" {
		f, err := os.Open(*followWav)
		if err != nil {
			fatal(err)
		}
		defer f.Close()
		followSamples, followRate, err = wavPCM16(f)
		if err != nil {
			fatal(err)
		}
	}

	var report probeReport
	var err error
	switch *mode {
	case "transcribe", "translate":
		if *wavPath == "" {
			fatal(fmt.Errorf("%s needs -wav", *mode))
		}
		format, ferr := cleanResponseFormat(*formatFlag)
		if ferr != nil {
			fatal(ferr)
		}
		if *wordsFlag && format != "verbose_json" {
			fatal(fmt.Errorf("-words requires -format verbose_json"))
		}
		if *logprobsFlag && format != "json" && format != "verbose_json" {
			fatal(fmt.Errorf("-logprobs requires -format json or verbose_json"))
		}
		report, err = probeTranscribe(*host, *model, *wavPath, samples, rate, *lang, *promptFlag, *transcribeTemp, *mode == "translate", format, *wordsFlag, *logprobsFlag, *outPath, *streamSpeech, *diarizeFlag, *maxTokensFlag, *suppressNSTFlag, *noFallbackFlag, *tempIncFlag, *entropyFlag, *logprobFlag, *noSpeechFlag, *beamFlag, *bestOfFlag, *maxLenFlag, *wordTholdFlag, *maxContextFlag, *audioCtxFlag, *offsetTFlag, *durationMSFlag, *suppressRegexFlag, *offsetNFlag, *grammarFlag, *grammarRuleFlag, *grammarPenaltyFlag, *dtwFlag, *vadThresholdFlag, *vadMinSpeechFlag, *vadMinSilenceFlag, *vadMaxSpeechFlag, *vadSpeechPadFlag, *vadOverlapFlag, *carryFlag, *noTimestampsFlag, *printSpecialFlag, *detectLanguageFlag, *vadFlag, *tinyDiarizeFlag, *splitOnWordFlag, *timeout)
	case "speech":
		if strings.TrimSpace(*textIn) == "" {
			fatal(fmt.Errorf("speech needs -text"))
		}
		seed, maxTokens, perr := parseSpeechLimits(*seedFlag, *maxTokensFlag)
		if perr != nil {
			fatal(perr)
		}
		sample, serr := parseSpeechSample(*topKFlag, *topPFlag, *repeatFlag)
		if serr != nil {
			fatal(serr)
		}
		format, ferr := cleanSpeechFormat(*formatFlag)
		if ferr != nil {
			fatal(ferr)
		}
		silence, nerr := parseSentenceSilence(*sentenceSilenceFlag)
		if nerr != nil {
			fatal(nerr)
		}
		noise, gerr := parseNoiseScale(*noiseScaleFlag)
		if gerr != nil {
			fatal(gerr)
		}
		width, werr := parseNoiseW(*noiseWFlag)
		if werr != nil {
			fatal(werr)
		}
		speaker, spkerr := parseSpeakerID(*speakerIDFlag)
		if spkerr != nil {
			fatal(spkerr)
		}
		phonemes, pherr := parsePhonemeSilence(*phonemeSilenceFlag)
		if pherr != nil {
			fatal(pherr)
		}
		report, err = probeSpeech(*host, *model, strings.TrimSpace(*textIn), *voice, *lang, *speakerRef, *outPath, *streamSpeech, format, seed, maxTokens, sample, silence, noise, width, speaker, phonemes, *timeout)
	case "realtime":
		mods, perr := parseModalities(*modalities)
		if perr != nil {
			fatal(perr)
		}
		toolsJSON, terr := loadToolsJSON(*toolsFlag)
		if terr != nil {
			fatal(terr)
		}
		pipe, pipeErr := cleanPipeline(*pipelineFlag)
		if pipeErr != nil {
			fatal(pipeErr)
		}
		var extra realtimeExtra
		if pipe == "duplex" {
			if strings.TrimSpace(*ttsFlag) == "" || strings.TrimSpace(*asrFlag) == "" {
				fatal(fmt.Errorf("duplex requires -tts and -asr"))
			}
			inRate, rerr := parseProbeRate(*rateFlag, rate)
			if rerr != nil {
				fatal(rerr)
			}
			extra = realtimeExtra{
				Pipeline: "duplex",
				Voice:    strings.TrimSpace(*ttsFlag),
				ASR:      strings.TrimSpace(*asrFlag),
				Speaker:  strings.TrimSpace(*voice),
				Rate:     inRate,
			}
		}
		if (strings.TrimSpace(*follow) != "" || *followWav != "") && strings.TrimSpace(*intent) != "" && pipe != "duplex" {
			fatal(fmt.Errorf("-follow needs a conversation; leave -intent empty"))
		}
		sessionType := ""
		if pipe == "" && strings.TrimSpace(*intent) == "transcription" {
			sessionType = "transcription"
		}
		langTag, lerr := cleanLangTag(*lang)
		if lerr != nil {
			fatal(lerr)
		}
		update, uerr := sessionUpdate(*instructions, toolsJSON, *toolChoice, *maxOut, langTag, sessionType)
		if uerr != nil {
			fatal(uerr)
		}
		turnMode, turnErr := cleanTurn(*turnFlag)
		if turnErr != nil {
			fatal(turnErr)
		}
		if turnMode == "server_vad" && pipe != "duplex" {
			update = applyModalities(update, mods)
		}
		report, err = probeRealtime(*host, *model, *intent, strings.TrimSpace(*textIn), mods, *barge, *outPath, samples, rate, *chunkMs, *timeout, update, *toolOutput, *follow, followSamples, followRate, turnMode, langTag, extra)
	default:
		fatal(fmt.Errorf("unknown mode %q", *mode))
	}
	if err != nil {
		fatal(err)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		fatal(err)
	}
}

type probeReport struct {
	Mode         string          `json:"mode"`
	Model        string          `json:"model"`
	InputRate    int             `json:"input_rate"`
	SessionRate  int             `json:"session_rate,omitempty"`
	Resampled    bool            `json:"resampled,omitempty"`
	OutputRate   int             `json:"output_rate,omitempty"`
	FirstTextMs  int64           `json:"first_text_ms,omitempty"`
	FirstAudioMs int64           `json:"first_audio_ms,omitempty"`
	CancelMs     int64           `json:"cancel_ms,omitempty"`
	TotalMs      int64           `json:"total_ms"`
	Status       string          `json:"status,omitempty"`
	Cancelled    bool            `json:"cancelled,omitempty"`
	Transcript   string          `json:"transcript,omitempty"`
	Reply        string          `json:"reply,omitempty"`
	ToolCalls    []probeToolCall `json:"tool_calls,omitempty"`
	AudioBytes   int             `json:"audio_bytes,omitempty"`
	AudioMIME    string          `json:"audio_mime,omitempty"`
	Words        int             `json:"words,omitempty"`
	Tokens       int             `json:"tokens,omitempty"`
	Seconds      float64         `json:"seconds,omitempty"`
	Logprobs     int             `json:"logprobs,omitempty"`
	OutFile      string          `json:"out_file,omitempty"`
	Error        string          `json:"error,omitempty"`
}

// probeToolCall is one function call from response.function_call_arguments.done.
type probeToolCall struct {
	Name      string `json:"name,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

func cleanResponseFormat(s string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "json":
		return "json", nil
	case "text", "srt", "verbose_json", "vtt", "diarized_json":
		return strings.ToLower(strings.TrimSpace(s)), nil
	default:
		return "", fmt.Errorf("format must be json, text, srt, verbose_json, vtt, or diarized_json")
	}
}

func cleanGrammar(grammar, rule string) (string, string, error) {
	if strings.TrimSpace(grammar) == "" && strings.TrimSpace(rule) == "" {
		return "", "", nil
	}
	if strings.TrimSpace(grammar) == "" || strings.TrimSpace(rule) == "" {
		return "", "", fmt.Errorf("grammar and grammar_rule are both required")
	}
	if len(grammar) > 8192 || strings.ContainsRune(grammar, 0) || !strings.Contains(grammar, "::=") {
		return "", "", fmt.Errorf("grammar must be GBNF of at most 8192 bytes")
	}
	rule = strings.TrimSpace(rule)
	found := false
	for _, line := range strings.Split(grammar, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, rule) && strings.HasPrefix(strings.TrimSpace(line[len(rule):]), "::=") {
			found = true
			break
		}
	}
	if !found {
		return "", "", fmt.Errorf("grammar_rule was not found in the grammar")
	}
	return grammar, rule, nil
}

func probeTranscribe(host, model, filename string, samples []int16, rate int, language, prompt, temperature string, translate bool, format string, words, logprobs bool, outPath string, stream, diarize bool, maxTokens string, suppressNST, noFallback bool, temperatureInc, entropyThold, logprobThold, noSpeechThold, beamSize, bestOf, maxLen, wordThold, maxContext, audioCtx, offsetT, durationMS, suppressRegex, offsetN, grammar, grammarRule, grammarPenalty, dtw, vadThreshold, vadMinSpeechMS, vadMinSilenceMS, vadMaxSpeechS, vadSpeechPadMS, vadSamplesOverlap string, carryInitial, noTimestamps, printSpecial, detectLanguage, vad, tinyDiarize, splitOnWord bool, timeout time.Duration) (probeReport, error) {
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	part, err := w.CreateFormFile("file", filename)
	if err != nil {
		return probeReport{}, err
	}
	if _, err := part.Write(wavWrap(samples, rate)); err != nil {
		return probeReport{}, err
	}
	if err := w.WriteField("model", model); err != nil {
		return probeReport{}, err
	}
	if lang, err := cleanLangTag(language); err != nil {
		return probeReport{}, err
	} else if lang != "" {
		if err := w.WriteField("language", lang); err != nil {
			return probeReport{}, err
		}
	}
	if p := strings.TrimSpace(prompt); p != "" {
		if err := w.WriteField("prompt", p); err != nil {
			return probeReport{}, err
		}
	}
	if temp := strings.TrimSpace(temperature); temp != "" {
		f, err := strconv.ParseFloat(temp, 64)
		if err != nil || f < 0 || f > 1 {
			return probeReport{}, fmt.Errorf("temperature must be between 0 and 1")
		}
		if err := w.WriteField("temperature", strconv.FormatFloat(f, 'f', -1, 64)); err != nil {
			return probeReport{}, err
		}
	}
	if format == "" {
		format = "json"
	}
	if err := w.WriteField("response_format", format); err != nil {
		return probeReport{}, err
	}
	if words {
		if err := w.WriteField("timestamp_granularities[]", "word"); err != nil {
			return probeReport{}, err
		}
	}
	if logprobs {
		if err := w.WriteField("include[]", "logprobs"); err != nil {
			return probeReport{}, err
		}
	}
	if stream {
		if err := w.WriteField("stream", "true"); err != nil {
			return probeReport{}, err
		}
	}
	if diarize {
		if err := w.WriteField("diarize", "true"); err != nil {
			return probeReport{}, err
		}
	}
	if suppressNST {
		if err := w.WriteField("suppress_nst", "true"); err != nil {
			return probeReport{}, err
		}
	}
	if noFallback {
		if err := w.WriteField("no_fallback", "true"); err != nil {
			return probeReport{}, err
		}
	}
	if noTimestamps {
		if err := w.WriteField("no_timestamps", "true"); err != nil {
			return probeReport{}, err
		}
	}
	if printSpecial {
		if err := w.WriteField("print_special", "true"); err != nil {
			return probeReport{}, err
		}
	}
	if detectLanguage {
		if err := w.WriteField("detect_language", "true"); err != nil {
			return probeReport{}, err
		}
	}
	if vad {
		if err := w.WriteField("vad", "true"); err != nil {
			return probeReport{}, err
		}
	}
	if tinyDiarize {
		if diarize {
			return probeReport{}, fmt.Errorf("tinydiarize cannot be combined with diarize")
		}
		if err := w.WriteField("tinydiarize", "true"); err != nil {
			return probeReport{}, err
		}
	}
	if th := strings.TrimSpace(vadThreshold); th != "" {
		f, err := strconv.ParseFloat(th, 64)
		if err != nil || f != f || f < 0 || f > 1 {
			return probeReport{}, fmt.Errorf("vad_threshold must be between 0 and 1")
		}
		if !vad {
			return probeReport{}, fmt.Errorf("vad_threshold requires vad=true")
		}
		if err := w.WriteField("vad_threshold", strconv.FormatFloat(f, 'f', -1, 64)); err != nil {
			return probeReport{}, err
		}
	}
	if ms := strings.TrimSpace(vadMinSpeechMS); ms != "" {
		n, err := strconv.Atoi(ms)
		if err != nil || n < 0 || n > 86_400_000 {
			return probeReport{}, fmt.Errorf("vad_min_speech_duration_ms must be between 0 and 86400000")
		}
		if !vad {
			return probeReport{}, fmt.Errorf("vad_min_speech_duration_ms requires vad=true")
		}
		if err := w.WriteField("vad_min_speech_duration_ms", strconv.Itoa(n)); err != nil {
			return probeReport{}, err
		}
	}
	if ms := strings.TrimSpace(vadMinSilenceMS); ms != "" {
		n, err := strconv.Atoi(ms)
		if err != nil || n < 0 || n > 86_400_000 {
			return probeReport{}, fmt.Errorf("vad_min_silence_duration_ms must be between 0 and 86400000")
		}
		if !vad {
			return probeReport{}, fmt.Errorf("vad_min_silence_duration_ms requires vad=true")
		}
		if err := w.WriteField("vad_min_silence_duration_ms", strconv.Itoa(n)); err != nil {
			return probeReport{}, err
		}
	}
	if sec := strings.TrimSpace(vadMaxSpeechS); sec != "" {
		f, err := strconv.ParseFloat(sec, 64)
		if err != nil || f != f || f < 1 || f > 100000 {
			return probeReport{}, fmt.Errorf("vad_max_speech_duration_s must be between 1 and 100000")
		}
		if !vad {
			return probeReport{}, fmt.Errorf("vad_max_speech_duration_s requires vad=true")
		}
		if err := w.WriteField("vad_max_speech_duration_s", strconv.FormatFloat(f, 'f', -1, 64)); err != nil {
			return probeReport{}, err
		}
	}
	if ms := strings.TrimSpace(vadSpeechPadMS); ms != "" {
		n, err := strconv.Atoi(ms)
		if err != nil || n < 0 || n > 86_400_000 {
			return probeReport{}, fmt.Errorf("vad_speech_pad_ms must be between 0 and 86400000")
		}
		if !vad {
			return probeReport{}, fmt.Errorf("vad_speech_pad_ms requires vad=true")
		}
		if err := w.WriteField("vad_speech_pad_ms", strconv.Itoa(n)); err != nil {
			return probeReport{}, err
		}
	}
	if sec := strings.TrimSpace(vadSamplesOverlap); sec != "" {
		f, err := strconv.ParseFloat(sec, 64)
		if err != nil || f != f || f < 0 || f > 3600 {
			return probeReport{}, fmt.Errorf("vad_samples_overlap must be between 0 and 3600")
		}
		if !vad {
			return probeReport{}, fmt.Errorf("vad_samples_overlap requires vad=true")
		}
		if err := w.WriteField("vad_samples_overlap", strconv.FormatFloat(f, 'f', -1, 64)); err != nil {
			return probeReport{}, err
		}
	}
	if carryInitial {
		if err := w.WriteField("carry_initial_prompt", "true"); err != nil {
			return probeReport{}, err
		}
	}
	if inc := strings.TrimSpace(temperatureInc); inc != "" {
		f, err := strconv.ParseFloat(inc, 64)
		if err != nil || f < 0 || f > 1 {
			return probeReport{}, fmt.Errorf("temperature_inc must be between 0 and 1")
		}
		if err := w.WriteField("temperature_inc", strconv.FormatFloat(f, 'f', -1, 64)); err != nil {
			return probeReport{}, err
		}
	}
	if thold := strings.TrimSpace(entropyThold); thold != "" {
		f, err := strconv.ParseFloat(thold, 64)
		if err != nil || f < 0 || f > 4 {
			return probeReport{}, fmt.Errorf("entropy_thold must be between 0 and 4")
		}
		if err := w.WriteField("entropy_thold", strconv.FormatFloat(f, 'f', -1, 64)); err != nil {
			return probeReport{}, err
		}
	}
	if lp := strings.TrimSpace(logprobThold); lp != "" {
		f, err := strconv.ParseFloat(lp, 64)
		if err != nil || f < -10 || f > 0 {
			return probeReport{}, fmt.Errorf("logprob_thold must be between -10 and 0")
		}
		if err := w.WriteField("logprob_thold", strconv.FormatFloat(f, 'f', -1, 64)); err != nil {
			return probeReport{}, err
		}
	}
	if ns := strings.TrimSpace(noSpeechThold); ns != "" {
		f, err := strconv.ParseFloat(ns, 64)
		if err != nil || f < 0 || f > 1 {
			return probeReport{}, fmt.Errorf("no_speech_thold must be between 0 and 1")
		}
		if err := w.WriteField("no_speech_thold", strconv.FormatFloat(f, 'f', -1, 64)); err != nil {
			return probeReport{}, err
		}
	}
	if beam := strings.TrimSpace(beamSize); beam != "" {
		n, err := strconv.Atoi(beam)
		if err != nil || n < 1 || n > 16 {
			return probeReport{}, fmt.Errorf("beam_size must be between 1 and 16")
		}
		if err := w.WriteField("beam_size", strconv.Itoa(n)); err != nil {
			return probeReport{}, err
		}
	}
	if best := strings.TrimSpace(bestOf); best != "" {
		n, err := strconv.Atoi(best)
		if err != nil || n < 1 || n > 16 {
			return probeReport{}, fmt.Errorf("best_of must be between 1 and 16")
		}
		if err := w.WriteField("best_of", strconv.Itoa(n)); err != nil {
			return probeReport{}, err
		}
	}
	wroteMaxLen := false
	if length := strings.TrimSpace(maxLen); length != "" {
		n, err := strconv.Atoi(length)
		if err != nil || n < 0 || n > 1000 {
			return probeReport{}, fmt.Errorf("max_len must be between 0 and 1000")
		}
		if n > 0 {
			if err := w.WriteField("max_len", strconv.Itoa(n)); err != nil {
				return probeReport{}, err
			}
			wroteMaxLen = true
		}
	}
	if splitOnWord {
		if !wroteMaxLen {
			return probeReport{}, fmt.Errorf("split_on_word requires max_len")
		}
		if err := w.WriteField("split_on_word", "true"); err != nil {
			return probeReport{}, err
		}
	}
	if thold := strings.TrimSpace(wordThold); thold != "" {
		f, err := strconv.ParseFloat(thold, 64)
		if err != nil || f < 0 || f > 1 {
			return probeReport{}, fmt.Errorf("word_thold must be between 0 and 1")
		}
		if err := w.WriteField("word_thold", strconv.FormatFloat(f, 'f', -1, 64)); err != nil {
			return probeReport{}, err
		}
	}
	if ctxCap := strings.TrimSpace(maxContext); ctxCap != "" {
		n, err := strconv.Atoi(ctxCap)
		if err != nil || n < -1 || n > 16384 {
			return probeReport{}, fmt.Errorf("max_context must be between -1 and 16384")
		}
		if n >= 0 {
			if err := w.WriteField("max_context", strconv.Itoa(n)); err != nil {
				return probeReport{}, err
			}
		}
	}
	if frames := strings.TrimSpace(audioCtx); frames != "" {
		n, err := strconv.Atoi(frames)
		if err != nil || n < 0 || n > 1500 {
			return probeReport{}, fmt.Errorf("audio_ctx must be between 0 and 1500")
		}
		if n > 0 {
			if err := w.WriteField("audio_ctx", strconv.Itoa(n)); err != nil {
				return probeReport{}, err
			}
		}
	}
	if start := strings.TrimSpace(offsetT); start != "" {
		n, err := strconv.Atoi(start)
		if err != nil || n < 0 || n > 86_400_000 {
			return probeReport{}, fmt.Errorf("offset_t must be between 0 and 86400000")
		}
		if n > 0 {
			if err := w.WriteField("offset_t", strconv.Itoa(n)); err != nil {
				return probeReport{}, err
			}
		}
	}
	if window := strings.TrimSpace(durationMS); window != "" {
		n, err := strconv.Atoi(window)
		if err != nil || n < 0 || n > 86_400_000 {
			return probeReport{}, fmt.Errorf("duration_ms must be between 0 and 86400000")
		}
		if n > 0 {
			if err := w.WriteField("duration_ms", strconv.Itoa(n)); err != nil {
				return probeReport{}, err
			}
		}
	}
	if pattern := suppressRegex; strings.TrimSpace(pattern) != "" {
		if len(pattern) > 256 || strings.ContainsRune(pattern, 0) {
			return probeReport{}, fmt.Errorf("suppress_regex must be at most 256 bytes")
		}
		if err := w.WriteField("suppress_regex", pattern); err != nil {
			return probeReport{}, err
		}
	}
	if cue := strings.TrimSpace(offsetN); cue != "" {
		n, err := strconv.Atoi(cue)
		if err != nil || n < -1_000_000 || n > 1_000_000 {
			return probeReport{}, fmt.Errorf("offset_n must be between -1000000 and 1000000")
		}
		if n != 0 {
			if err := w.WriteField("offset_n", strconv.Itoa(n)); err != nil {
				return probeReport{}, err
			}
		}
	}
	grammar, grammarRule, gerr := cleanGrammar(grammar, grammarRule)
	if gerr != nil {
		return probeReport{}, gerr
	}
	if grammar != "" {
		if err := w.WriteField("grammar", grammar); err != nil {
			return probeReport{}, err
		}
		if err := w.WriteField("grammar_rule", grammarRule); err != nil {
			return probeReport{}, err
		}
	}
	if preset := strings.ToLower(strings.TrimSpace(dtw)); preset != "" {
		switch preset {
		case "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large.v1", "large.v2", "large.v3", "large.v3.turbo":
		default:
			return probeReport{}, fmt.Errorf("dtw must be a whisper model preset")
		}
		if err := w.WriteField("dtw", preset); err != nil {
			return probeReport{}, err
		}
	}
	if penalty := strings.TrimSpace(grammarPenalty); penalty != "" {
		f, err := strconv.ParseFloat(penalty, 64)
		if err != nil || f != f || f < 0 || f > 1000 {
			return probeReport{}, fmt.Errorf("grammar_penalty must be between 0 and 1000")
		}
		if err := w.WriteField("grammar_penalty", strconv.FormatFloat(f, 'f', -1, 64)); err != nil {
			return probeReport{}, err
		}
	}
	if capTokens := strings.TrimSpace(maxTokens); capTokens != "" {
		n, err := strconv.Atoi(capTokens)
		if err != nil || n < 1 || n > 1_000_000 {
			return probeReport{}, fmt.Errorf("max_tokens must be a positive integer")
		}
		if err := w.WriteField("max_tokens", strconv.Itoa(n)); err != nil {
			return probeReport{}, err
		}
	}
	if err := w.Close(); err != nil {
		return probeReport{}, err
	}
	path := "/v1/audio/transcriptions"
	if translate {
		path = "/v1/audio/translations"
	}
	req, err := http.NewRequest(http.MethodPost, strings.TrimRight(host, "/")+path, &body)
	if err != nil {
		return probeReport{}, err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	client := &http.Client{Timeout: timeout}
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return probeReport{}, err
	}
	defer resp.Body.Close()
	var raw []byte
	var streamText string
	var streamTokens int
	var firstDelta int64
	var streamSeconds float64
	if stream {
		streamText, streamTokens, streamSeconds, firstDelta, raw, err = readTranscriptStream(resp.Body, start)
	} else {
		raw, err = io.ReadAll(resp.Body)
	}
	elapsed := time.Since(start).Milliseconds()
	if err != nil {
		return probeReport{}, err
	}
	modeName := "transcribe"
	if translate {
		modeName = "translate"
	}
	report := probeReport{Mode: modeName, Model: model, InputRate: rate, TotalMs: elapsed, Status: resp.Header.Get("Content-Type")}
	if resp.StatusCode != http.StatusOK {
		report.Error = strings.TrimSpace(string(raw))
		return report, fmt.Errorf("transcription HTTP %d", resp.StatusCode)
	}
	var parsed struct {
		Text  string `json:"text"`
		Words []struct {
			Word string `json:"word"`
		} `json:"words"`
		Usage struct {
			TotalTokens int     `json:"total_tokens"`
			Seconds     float64 `json:"seconds"`
		} `json:"usage"`
		Logprobs []struct {
			Token string `json:"token"`
		} `json:"logprobs"`
	}
	if stream && streamText != "" {
		report.Transcript = strings.TrimSpace(streamText)
		report.Tokens = streamTokens
		report.Seconds = streamSeconds
		if firstDelta > 0 {
			report.FirstTextMs = firstDelta
		}
	} else if err := json.Unmarshal(raw, &parsed); err != nil || (format != "json" && format != "verbose_json") {
		report.Transcript = strings.TrimSpace(string(raw))
	} else {
		report.Transcript = strings.TrimSpace(parsed.Text)
		report.Words = len(parsed.Words)
		report.Tokens = parsed.Usage.TotalTokens
		report.Seconds = parsed.Usage.Seconds
		report.Logprobs = len(parsed.Logprobs)
	}
	if outPath != "" {
		if err := os.WriteFile(outPath, raw, 0o644); err != nil {
			return report, err
		}
		report.OutFile = outPath
	}
	if report.FirstTextMs == 0 {
		report.FirstTextMs = elapsed
	}
	return report, nil
}

// readTranscriptStream reads llama-server transcription SSE. data lines are
// transcript.text.delta and transcript.text.done, then data: [DONE].
func readTranscriptStream(r io.Reader, start time.Time) (text string, tokens int, seconds float64, firstDelta int64, raw []byte, err error) {
	var buf bytes.Buffer
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	var deltas strings.Builder
	for sc.Scan() {
		line := sc.Text()
		buf.WriteString(line)
		buf.WriteByte('\n')
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var ev struct {
			Type  string `json:"type"`
			Delta string `json:"delta"`
			Text  string `json:"text"`
			Usage struct {
				TotalTokens int     `json:"total_tokens"`
				Seconds     float64 `json:"seconds"`
			} `json:"usage"`
		}
		if json.Unmarshal([]byte(payload), &ev) != nil {
			continue
		}
		if ev.Type == "transcript.text.delta" && ev.Delta != "" {
			if firstDelta == 0 {
				firstDelta = time.Since(start).Milliseconds()
			}
			deltas.WriteString(ev.Delta)
		}
		if ev.Type == "transcript.text.done" {
			text = ev.Text
			tokens = ev.Usage.TotalTokens
			seconds = ev.Usage.Seconds
		}
	}
	if err := sc.Err(); err != nil {
		return "", 0, 0, 0, buf.Bytes(), err
	}
	if strings.TrimSpace(text) == "" {
		text = deltas.String()
	}
	return text, tokens, seconds, firstDelta, buf.Bytes(), nil
}

func realtimeURL(host, model, intent, language, turn string, extra realtimeExtra) (string, error) {
	u, err := url.Parse(strings.TrimRight(host, "/") + "/v1/realtime")
	if err != nil {
		return "", err
	}
	q := u.Query()
	q.Set("model", model)
	if extra.Pipeline == "" && intent != "" {
		q.Set("intent", intent)
	}
	if language != "" {
		q.Set("language", language)
	}
	if turn != "" {
		q.Set("turn", turn)
	}
	if extra.Pipeline != "" {
		q.Set("pipeline", extra.Pipeline)
	}
	if extra.Voice != "" {
		q.Set("voice", extra.Voice)
	}
	if extra.ASR != "" {
		q.Set("asr", extra.ASR)
	}
	if extra.Speaker != "" {
		q.Set("speaker", extra.Speaker)
	}
	if extra.Rate > 0 {
		q.Set("rate", strconv.Itoa(extra.Rate))
	}
	u.RawQuery = q.Encode()
	switch u.Scheme {
	case "http", "ws", "":
		u.Scheme = "ws"
	default:
		return "", fmt.Errorf("probe speaks ws over %s; use an http:// host", u.Scheme)
	}
	return u.String(), nil
}

func probeRealtime(host, model, intent, textIn string, modalities []string, barge bool, outPath string, samples []int16, fileRate, chunkMs int, timeout time.Duration, session map[string]any, toolOutput, follow string, followSamples []int16, followRate int, turn, language string, extra realtimeExtra) (probeReport, error) {
	wsURL, err := realtimeURL(host, model, intent, language, turn, extra)
	if err != nil {
		return probeReport{}, err
	}
	ws, err := dialWebSocket(wsURL)
	if err != nil {
		return probeReport{}, err
	}
	defer ws.Close()
	_ = ws.conn.SetDeadline(time.Now().Add(timeout))

	report := probeReport{Mode: "realtime", Model: model, InputRate: fileRate}
	var (
		text         strings.Builder
		reply        strings.Builder
		spoken       []byte
		spokenMIME   string
		sent         bool
		start        time.Time
		awaitUpdate  bool
		sessionRate  int
		cancelAt     time.Time
		cancelSent   bool
		heardOutput  bool
		replyStarted bool
		toolReplied  bool
		followSent   bool
	)
	finish := func(r probeReport, err error) (probeReport, error) {
		if spokenMIME != "" && spokenMIME != "audio/pcm" {
			r.AudioMIME = spokenMIME
			if spokenMIME == "audio/wav" {
				if sec, ok := wavSeconds(spoken); ok {
					r.Seconds = sec
				}
			}
			if outPath != "" {
				if werr := os.WriteFile(outPath, spoken, 0o644); werr == nil {
					r.OutFile = outPath
				}
			}
		} else if r.OutputRate > 0 && len(spoken) >= 2 {
			n := len(spoken) - len(spoken)%2
			r.Seconds = float64(n/2) / float64(r.OutputRate)
			if outPath != "" {
				samples := make([]int16, n/2)
				for i := range samples {
					samples[i] = int16(spoken[i*2]) | int16(spoken[i*2+1])<<8
				}
				if werr := os.WriteFile(outPath, wavWrap(samples, r.OutputRate), 0o644); werr == nil {
					r.OutFile = outPath
				}
			}
		}
		return r, err
	}
	sendAudio := func(sessionRate int) error {
		pcm := samples
		if fileRate != sessionRate {
			pcm = resampleLinear(samples, fileRate, sessionRate)
			report.Resampled = true
		}
		report.SessionRate = sessionRate
		start = time.Now()
		msgs, err := userTurn(textIn, pcm, sessionRate, chunkMs, intent == "" || extra.Pipeline == "duplex", modalities, turn == "server_vad", extra.Pipeline == "duplex")
		if err != nil {
			return err
		}
		for _, msg := range msgs {
			payload, err := json.Marshal(msg)
			if err != nil {
				return err
			}
			if err := ws.WriteText(payload); err != nil {
				return err
			}
		}
		sent = true
		return nil
	}
	maybeBarge := func() error {
		if !barge || cancelSent || (intent != "" && extra.Pipeline != "duplex") {
			return nil
		}
		if !replyStarted {
			return nil
		}
		cancelSent = true
		cancelAt = time.Now()
		return ws.WriteText([]byte(`{"type":"response.cancel"}`))
	}

	for {
		msg, err := ws.ReadText()
		if err != nil {
			if sent && (report.Transcript != "" || report.AudioBytes > 0 || heardOutput) {
				report.TotalMs = sinceOrZero(start)
				return finish(report, nil)
			}
			return finish(report, err)
		}
		var ev map[string]any
		if err := json.Unmarshal([]byte(msg), &ev); err != nil {
			continue
		}
		kind, _ := ev["type"].(string)
		switch kind {
		case "response.created":
			replyStarted = true
		case "session.created":
			rate, ok := sessionInputRate(ev)
			if !ok {
				return finish(report, fmt.Errorf("session.created did not include audio.input.format.rate"))
			}
			sessionRate = rate
			if out, ok := sessionOutputRate(ev); ok {
				report.OutputRate = out
			}
			if session != nil {
				payload, err := json.Marshal(session)
				if err != nil {
					return finish(report, err)
				}
				if err := ws.WriteText(payload); err != nil {
					return finish(report, err)
				}
				awaitUpdate = true
				continue
			}
			if err := sendAudio(rate); err != nil {
				return finish(report, err)
			}
		case "session.updated":
			if !awaitUpdate || sent {
				continue
			}
			awaitUpdate = false
			if out, ok := sessionOutputRate(ev); ok {
				report.OutputRate = out
			}
			if err := sendAudio(sessionRate); err != nil {
				return finish(report, err)
			}
		case "response.function_call_arguments.done":
			report.ToolCalls = append(report.ToolCalls, probeToolCall{
				Name:      jsonString(ev, "name"),
				CallID:    jsonString(ev, "call_id"),
				Arguments: jsonString(ev, "arguments"),
			})
			if !start.IsZero() && !heardOutput {
				report.FirstTextMs = time.Since(start).Milliseconds()
			}
			heardOutput = true
		case "conversation.item.input_audio_transcription.delta":
			delta, _ := ev["delta"].(string)
			if delta != "" && !start.IsZero() && !heardOutput {
				report.FirstTextMs = time.Since(start).Milliseconds()
				heardOutput = true
			}
			text.WriteString(delta)
		case "response.output_text.delta":
			delta, _ := ev["delta"].(string)
			if delta != "" && !start.IsZero() && !heardOutput {
				report.FirstTextMs = time.Since(start).Milliseconds()
				heardOutput = true
			}
			reply.WriteString(delta)
			if delta != "" {
				replyStarted = true
			}
		case "conversation.item.input_audio_transcription.completed":
			if full, _ := ev["transcript"].(string); full != "" {
				text.Reset()
				text.WriteString(full)
			}
			report.Transcript = strings.TrimSpace(text.String())
			if !heardOutput && !start.IsZero() {
				report.FirstTextMs = time.Since(start).Milliseconds()
				heardOutput = true
			}
		case "response.output_audio.delta":
			audio, _ := ev["delta"].(string)
			if audio != "" {
				decoded, err := base64.StdEncoding.DecodeString(audio)
				if err == nil {
					mime, _ := ev["mime"].(string)
					if mime == "" {
						mime = "audio/pcm"
					}
					if spokenMIME == "" {
						spokenMIME = mime
					}
					if mime == spokenMIME {
						spoken = append(spoken, decoded...)
						report.AudioBytes += len(decoded)
					}
					if r, ok := ev["rate"].(float64); ok && r > 0 && report.OutputRate == 0 {
						report.OutputRate = int(r)
					}
					if !start.IsZero() && report.FirstAudioMs == 0 {
						report.FirstAudioMs = time.Since(start).Milliseconds()
					}
					heardOutput = true
					replyStarted = true
				}
			}
		case "response.done", "error":
			if kind == "error" {
				if errObj, _ := ev["error"].(map[string]any); errObj != nil {
					if m, _ := errObj["message"].(string); m != "" {
						report.Error = m
					}
				}
				if report.Error == "" {
					report.Error = msg
				}
				report.TotalMs = sinceOrZero(start)
				return finish(report, fmt.Errorf("%s", report.Error))
			}
			report.Reply = strings.TrimSpace(reply.String())
			if heard := strings.TrimSpace(text.String()); heard != "" {
				report.Transcript = heard
			} else if report.Reply != "" {
				report.Transcript = report.Reply
			}
			report.Status = responseStatus(ev)
			report.Cancelled = report.Status == "cancelled"
			if report.Cancelled && !cancelAt.IsZero() {
				report.CancelMs = time.Since(cancelAt).Milliseconds()
			}
			report.TotalMs = sinceOrZero(start)
			if strings.TrimSpace(toolOutput) != "" && !toolReplied && !report.Cancelled && len(report.ToolCalls) > 0 {
				toolReplied = true
				text.Reset()
				reply.Reset()
				for _, call := range report.ToolCalls {
					payload, err := json.Marshal(toolResult(call.CallID, toolOutput))
					if err != nil {
						return finish(report, err)
					}
					if err := ws.WriteText(payload); err != nil {
						return finish(report, err)
					}
				}
				create := map[string]any{"type": "response.create"}
				if len(modalities) > 0 {
					create["response"] = map[string]any{"output_modalities": modalities}
				}
				payload, err := json.Marshal(create)
				if err != nil {
					return finish(report, err)
				}
				if err := ws.WriteText(payload); err != nil {
					return finish(report, err)
				}
				continue
			}
			if (strings.TrimSpace(follow) != "" || len(followSamples) > 0) && !followSent && !report.Cancelled && (intent == "" || extra.Pipeline == "duplex") {
				followSent = true
				text.Reset()
				reply.Reset()
				var msgs []map[string]any
				if len(followSamples) > 0 {
					pcm := followSamples
					if followRate > 0 && sessionRate > 0 && followRate != sessionRate {
						pcm = resampleLinear(followSamples, followRate, sessionRate)
					}
					var turnErr error
					msgs, turnErr = userTurn(strings.TrimSpace(follow), pcm, sessionRate, chunkMs, true, modalities, turn == "server_vad", extra.Pipeline == "duplex")
					if turnErr != nil {
						return finish(report, turnErr)
					}
				} else {
					msgs = followTurn(strings.TrimSpace(follow), modalities)
				}
				for _, msg := range msgs {
					payload, err := json.Marshal(msg)
					if err != nil {
						return finish(report, err)
					}
					if err := ws.WriteText(payload); err != nil {
						return finish(report, err)
					}
				}
				continue
			}
			return finish(report, nil)
		}
		if err := maybeBarge(); err != nil {
			return finish(report, err)
		}
		if sent && report.Transcript != "" && kind == "conversation.item.input_audio_transcription.completed" {
			// Live ASR completes the item when the commit drains. Keep reading
			// until the socket closes or response.done, but do not wait forever
			// after the transcript is in hand if the next read would only be EOF.
			report.TotalMs = sinceOrZero(start)
			_ = ws.conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		}
	}
}

func loadToolsJSON(flag string) (string, error) {
	s := strings.TrimSpace(flag)
	if s == "" || strings.HasPrefix(s, "[") {
		return s, nil
	}
	b, err := os.ReadFile(s)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func jsonString(ev map[string]any, key string) string {
	s, _ := ev[key].(string)
	return s
}

func sinceOrZero(start time.Time) int64 {
	if start.IsZero() {
		return 0
	}
	return time.Since(start).Milliseconds()
}

// wavWrapFloat packs little-endian float32 PCM into a WAV (format 3).
func wavWrapFloat(pcm []byte, rate int) []byte {
	var buf bytes.Buffer
	write := func(b []byte) { buf.Write(b) }
	u32 := func(v uint32) {
		var b [4]byte
		b[0] = byte(v)
		b[1] = byte(v >> 8)
		b[2] = byte(v >> 16)
		b[3] = byte(v >> 24)
		write(b[:])
	}
	u16 := func(v uint16) {
		write([]byte{byte(v), byte(v >> 8)})
	}
	write([]byte("RIFF"))
	u32(uint32(36 + len(pcm)))
	write([]byte("WAVE"))
	write([]byte("fmt "))
	u32(16)
	u16(3)
	u16(1)
	u32(uint32(rate))
	u32(uint32(rate * 4))
	u16(4)
	u16(32)
	write([]byte("data"))
	u32(uint32(len(pcm)))
	write(pcm)
	return buf.Bytes()
}

func wavWrap(samples []int16, rate int) []byte {
	data := pcm16Bytes(samples)
	var buf bytes.Buffer
	write := func(b []byte) { buf.Write(b) }
	u32 := func(v uint32) {
		var b [4]byte
		b[0] = byte(v)
		b[1] = byte(v >> 8)
		b[2] = byte(v >> 16)
		b[3] = byte(v >> 24)
		write(b[:])
	}
	u16 := func(v uint16) {
		write([]byte{byte(v), byte(v >> 8)})
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

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "realtime_probe: %v\n", err)
	os.Exit(1)
}
