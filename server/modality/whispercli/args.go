// Package whispercli builds the argv for a whisper.cpp-compatible CLI.
package whispercli

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Segment is one whisper.cpp transcription row. Start and End are seconds.
// AvgLogprob is the mean natural log of token probabilities when the JSON
// includes them. NoSpeechProb is the <|nospeech|> token probability.
// Tokens are whisper.cpp token ids when the JSON wrote them.
// Speaker is "0", "1", or "?" when the text starts with whisper.cpp's
// "(speaker N)" diarization prefix.
// SpeakerTurnNext is true when a tinydiarize model emitted the turn token.
// The text then ends with " [SPEAKER_TURN]".
type Segment struct {
	Start           float64
	End             float64
	Text            string
	Speaker         string
	SpeakerTurnNext bool
	Tokens          []int
	AvgLogprob      *float64
	NoSpeechProb    *float64
}

// Word is one word and its time span, in seconds.
type Word struct {
	Word  string
	Start float64
	End   float64
}

// Logprob is one whisper.cpp token and the natural log of its probability.
type Logprob struct {
	Token   string
	Logprob float64
}

// Transcript is the text file plus, when -oj was set, the parsed JSON.
// Words is set when the JSON includes per-token times (-ojf).
type Transcript struct {
	Text                string
	Language            string
	DetectedLanguage    string
	LanguageProbability *float64
	Segments            []Segment
	Words               []Word
	Logprobs            []Logprob
}

// Args is the whisper.cpp argv without the binary. prompt and temperature
// use the long flags from examples/cli/cli.cpp. Empty values are omitted.
// jsonOut adds -oj so the CLI also writes <outBase>.json with segment offsets.
// words adds --split-on-word and -ojf, so segments break on words and the JSON
// includes per-token times.
// translate adds --translate, which asks whisper.cpp for English text.
// diarize adds --diarize, which labels a stereo clip by channel energy.
// suppressNST adds --suppress-nst, which keeps non-speech marks out of the text.
// noFallback adds --no-fallback, so a failed segment stays at the requested temperature.
// temperatureInc adds --temperature-inc. Empty leaves whisper.cpp at 0.2.
// entropyThold adds --entropy-thold. Empty leaves whisper.cpp at 2.4.
// logprobThold adds --logprob-thold. Empty leaves whisper.cpp at -1.
// noSpeechThold adds --no-speech-thold. Empty leaves whisper.cpp at 0.6.
// beamSize adds --beam-size. Empty leaves whisper.cpp at 5. The width applies when the temperature is 0.
// bestOf adds --best-of. Empty leaves whisper.cpp at 5. The count applies when the temperature is above 0.
// carryInitial adds --carry-initial-prompt, so the initial prompt stays in front of every window.
// maxLen adds --max-len. Empty leaves the character cap off. A positive value wraps a longer segment.
// splitOnWord adds --split-on-word when word timestamps did not already add it.
// The wrap then waits for a token that starts with a space.
// wordThold adds --word-thold. Empty leaves whisper.cpp at 0.01. A timestamp is kept when its probability is above this value.
// maxContext adds --max-context. Empty, or -1, leaves the library cap. 0 leaves earlier text out of the next window.
// audioCtx adds --audio-ctx. Empty, or 0, uses the full encoder context.
// offsetT adds --offset-t. Empty, or 0, starts at the beginning of the clip.
// durationMS adds --duration. Empty, or 0, continues to the end of the clip.
// suppressRegex adds --suppress-regex. Empty leaves every token available.
// noTimestamps adds --no-timestamps, so the decoder cannot emit timestamp tokens.
// grammar and grammarRule add --grammar and --grammar-rule. Empty leaves sampling unguided.
// grammarPenalty adds --grammar-penalty. Empty leaves whisper.cpp at 100. 0 leaves rejected-token logits unchanged.
// printSpecial adds --print-special, so special tokens stay in the segment text.
// detectLanguage adds --detect-language. Whisper.cpp reports the language and does not decode a transcript.
// dtw adds --dtw and --no-flash-attn. Empty leaves flash attention on and keeps token-offset word times.
// vadModel adds --vad and --vad-model. Empty transcribes the whole clip.
// vadThreshold adds --vad-threshold. Empty stays at 0.5.
// vadMinSpeechMS adds --vad-min-speech-duration-ms. Empty stays at 250.
// vadMinSilenceMS adds --vad-min-silence-duration-ms. Empty stays at 100.
// vadMaxSpeechS adds --vad-max-speech-duration-s. Empty leaves the cap off.
// vadSpeechPadMS adds --vad-speech-pad-ms. Empty stays at 30.
// vadSamplesOverlap adds --vad-samples-overlap. Empty stays at 0.1.
// tinyDiarize adds --tinydiarize. A tinydiarize model may end a segment with " [SPEAKER_TURN]".
// TextArgs is a text-only transcription for the live socket.
// Empty language leaves the CLI default. Timestamp tokens are skipped.
// Beam size and best-of are 1. The CLI default beam size is 5, and any
// value above 1 switches the decoder from one greedy pass into beam search.
func TextArgs(modelPath, inPath, outBase, language string) []string {
	return Args(modelPath, inPath, outBase, language, "", "", "", "", "", "", "1", "1", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", false, false, false, false, false, false, false, true, false, false, false, false)
}

func Args(modelPath, inPath, outBase, language, prompt, temperature, temperatureInc, entropyThold, logprobThold, noSpeechThold, beamSize, bestOf, maxLen, wordThold, maxContext, audioCtx, offsetT, durationMS, suppressRegex, grammar, grammarRule, grammarPenalty, dtw, vadModel, vadThreshold, vadMinSpeechMS, vadMinSilenceMS, vadMaxSpeechS, vadSpeechPadMS, vadSamplesOverlap string, jsonOut, words, translate, diarize, suppressNST, noFallback, carryInitial, noTimestamps, printSpecial, detectLanguage, tinyDiarize, splitOnWord bool) []string {
	args := []string{"-m", modelPath, "-f", inPath, "-otxt", "-of", outBase}
	if language != "" {
		args = append(args, "-l", language)
	}
	if prompt = strings.TrimSpace(prompt); prompt != "" {
		args = append(args, "--prompt", prompt)
	}
	if temperature = strings.TrimSpace(temperature); temperature != "" {
		args = append(args, "--temperature", temperature)
	}
	if temperatureInc = strings.TrimSpace(temperatureInc); temperatureInc != "" {
		args = append(args, "--temperature-inc", temperatureInc)
	}
	if entropyThold = strings.TrimSpace(entropyThold); entropyThold != "" {
		args = append(args, "--entropy-thold", entropyThold)
	}
	if logprobThold = strings.TrimSpace(logprobThold); logprobThold != "" {
		args = append(args, "--logprob-thold", logprobThold)
	}
	if noSpeechThold = strings.TrimSpace(noSpeechThold); noSpeechThold != "" {
		args = append(args, "--no-speech-thold", noSpeechThold)
	}
	if beamSize = strings.TrimSpace(beamSize); beamSize != "" {
		args = append(args, "--beam-size", beamSize)
	}
	if bestOf = strings.TrimSpace(bestOf); bestOf != "" {
		args = append(args, "--best-of", bestOf)
	}
	if maxLen = strings.TrimSpace(maxLen); maxLen != "" {
		args = append(args, "--max-len", maxLen)
	}
	if wordThold = strings.TrimSpace(wordThold); wordThold != "" {
		args = append(args, "--word-thold", wordThold)
	}
	if maxContext = strings.TrimSpace(maxContext); maxContext != "" {
		args = append(args, "--max-context", maxContext)
	}
	if audioCtx = strings.TrimSpace(audioCtx); audioCtx != "" {
		args = append(args, "--audio-ctx", audioCtx)
	}
	if offsetT = strings.TrimSpace(offsetT); offsetT != "" {
		args = append(args, "--offset-t", offsetT)
	}
	if durationMS = strings.TrimSpace(durationMS); durationMS != "" {
		args = append(args, "--duration", durationMS)
	}
	if suppressRegex != "" {
		args = append(args, "--suppress-regex", suppressRegex)
	}
	if grammar != "" {
		args = append(args, "--grammar", grammar, "--grammar-rule", grammarRule)
	}
	if grammarPenalty = strings.TrimSpace(grammarPenalty); grammarPenalty != "" {
		args = append(args, "--grammar-penalty", grammarPenalty)
	}
	if translate {
		args = append(args, "--translate")
	}
	if diarize {
		args = append(args, "--diarize")
	}
	if tinyDiarize {
		args = append(args, "--tinydiarize")
	}
	if suppressNST {
		args = append(args, "--suppress-nst")
	}
	if noFallback {
		args = append(args, "--no-fallback")
	}
	if carryInitial {
		args = append(args, "--carry-initial-prompt")
	}
	if noTimestamps {
		args = append(args, "--no-timestamps")
	}
	if printSpecial {
		args = append(args, "--print-special")
	}
	if detectLanguage {
		args = append(args, "--detect-language")
	}
	if dtw != "" {
		args = append(args, "--dtw", dtw, "--no-flash-attn")
	}
	if vadModel != "" {
		args = append(args, "--vad", "--vad-model", vadModel)
	}
	if vadThreshold != "" {
		args = append(args, "--vad-threshold", vadThreshold)
	}
	if vadMinSpeechMS != "" {
		args = append(args, "--vad-min-speech-duration-ms", vadMinSpeechMS)
	}
	if vadMinSilenceMS != "" {
		args = append(args, "--vad-min-silence-duration-ms", vadMinSilenceMS)
	}
	if vadMaxSpeechS != "" {
		args = append(args, "--vad-max-speech-duration-s", vadMaxSpeechS)
	}
	if vadSpeechPadMS != "" {
		args = append(args, "--vad-speech-pad-ms", vadSpeechPadMS)
	}
	if vadSamplesOverlap != "" {
		args = append(args, "--vad-samples-overlap", vadSamplesOverlap)
	}
	if words {
		args = append(args, "--split-on-word", "-ojf")
	} else if jsonOut {
		args = append(args, "-oj")
	}
	if splitOnWord && !words {
		args = append(args, "--split-on-word")
	}
	return args
}

// WordTimestamps reports whether the form values ask for word times.
// Accepted values are segment and word. Empty is segment, the default.
func WordTimestamps(values []string) (bool, error) {
	words := false
	for _, v := range values {
		for _, part := range strings.Split(v, ",") {
			switch strings.ToLower(strings.TrimSpace(part)) {
			case "", "segment":
			case "word":
				words = true
			default:
				return false, fmt.Errorf("timestamp_granularities must be segment or word")
			}
		}
	}
	return words, nil
}

// token is one whisper.cpp JSON token. P is the probability from -ojf.
type token struct {
	Offsets struct {
		From int64 `json:"from"`
		To   int64 `json:"to"`
	} `json:"offsets"`
	Text string   `json:"text"`
	P    float64  `json:"p"`
	ID   *int     `json:"id"`
	DTW  *float64 `json:"t_dtw"`
}

// ParseJSON reads a whisper.cpp -oj file. offsets are milliseconds.
func ParseJSON(raw []byte) (Transcript, error) {
	var doc struct {
		Result struct {
			Language string `json:"language"`
		} `json:"result"`
		Transcription []struct {
			Offsets struct {
				From int64 `json:"from"`
				To   int64 `json:"to"`
			} `json:"offsets"`
			Text            string  `json:"text"`
			SpeakerTurnNext bool    `json:"speaker_turn_next"`
			Tokens          []token `json:"tokens"`
		} `json:"transcription"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return Transcript{}, err
	}
	var b strings.Builder
	out := Transcript{Language: doc.Result.Language}
	for _, item := range doc.Transcription {
		text := item.Text
		if item.SpeakerTurnNext {
			text += " [SPEAKER_TURN]"
		}
		if strings.TrimSpace(text) == "" {
			continue
		}
		seg := Segment{
			Start:           float64(item.Offsets.From) / 1000,
			End:             float64(item.Offsets.To) / 1000,
			Text:            text,
			SpeakerTurnNext: item.SpeakerTurnNext,
		}
		seg.AvgLogprob, seg.NoSpeechProb = tokenScores(item.Tokens)
		seg.Tokens = tokenIDs(item.Tokens)
		seg.Speaker = SpeakerLabel(text)
		out.Segments = append(out.Segments, seg)
		b.WriteString(text)
		out.Words = append(out.Words, wordsFromTokens(item.Tokens)...)
		out.Logprobs = append(out.Logprobs, tokenLogprobs(item.Tokens)...)
	}
	out.Text = strings.TrimSpace(b.String())
	return out, nil
}

func tokenScores(tokens []token) (avg, noSpeech *float64) {
	var sum float64
	var n int
	for _, tok := range tokens {
		trimmed := strings.TrimSpace(tok.Text)
		if isNoSpeech(trimmed) && tok.P > 0 && tok.P <= 1 && noSpeech == nil {
			p := tok.P
			noSpeech = &p
		}
		if trimmed == "" || isWhisperSpecial(trimmed) || tok.P <= 0 || tok.P > 1 {
			continue
		}
		sum += math.Log(tok.P)
		n++
	}
	if n == 0 {
		return nil, noSpeech
	}
	mean := sum / float64(n)
	return &mean, noSpeech
}

// SpeakerLabel reads a whisper.cpp diarization prefix. Any other text returns empty.
func SpeakerLabel(text string) string {
	s := strings.TrimLeft(text, " \t")
	const prefix = "(speaker "
	if !strings.HasPrefix(s, prefix) {
		return ""
	}
	rest := s[len(prefix):]
	end := strings.IndexByte(rest, ')')
	if end <= 0 {
		return ""
	}
	switch rest[:end] {
	case "0", "1", "?":
		return rest[:end]
	default:
		return ""
	}
}

// StripSpeakerPrefix removes a leading "(speaker 0)", "(speaker 1)", or
// "(speaker ?)" mark. Text without that mark is returned unchanged.
func StripSpeakerPrefix(text string) string {
	s := strings.TrimLeft(text, " \t")
	label := SpeakerLabel(s)
	if label == "" {
		return text
	}
	rest := s[len("(speaker ")+len(label):]
	if len(rest) > 0 && rest[0] == ')' {
		rest = rest[1:]
	}
	if strings.HasPrefix(rest, " ") {
		rest = rest[1:]
	}
	return rest
}

// Grammar reads the multipart grammar and grammar_rule fields. Both empty
// leaves sampling unguided. A set grammar is GBNF text, and grammar_rule is
// the rule it starts from. Tokens the grammar rejects lose grammar_penalty logit
// points. An omitted penalty stays at 100. A grammar uses beam search.
// A grammar whisper.cpp cannot parse fails the transcription.
func Grammar(grammar, rule string) (string, string, error) {
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
	if len(rule) > 64 || strings.ContainsAny(rule, " \t\r\n") {
		return "", "", fmt.Errorf("grammar_rule must be a rule name")
	}
	if !grammarDeclares(grammar, rule) {
		return "", "", fmt.Errorf("grammar_rule was not found in the grammar")
	}
	return grammar, rule, nil
}

// GrammarPenalty reads the multipart grammar_penalty field. Empty leaves the
// whisper.cpp default of 100. A set value, including 0, is the logit points
// subtracted from each token the grammar rejects. 0 leaves those logits unchanged.
func GrammarPenalty(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) || f < 0 || f > 1000 {
		return "", fmt.Errorf("grammar_penalty must be between 0 and 1000")
	}
	return strconv.FormatFloat(f, 'f', -1, 64), nil
}

func grammarDeclares(grammar, rule string) bool {
	for _, line := range strings.Split(grammar, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, rule) {
			continue
		}
		rest := strings.TrimSpace(line[len(rule):])
		if strings.HasPrefix(rest, "::=") {
			return true
		}
	}
	return false
}

// VADFlag reads the multipart vad field. Empty is false.
// True keeps speech and drops the rest. The Silero model path comes from the server.
func VADFlag(s string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "false", "0":
		return false, nil
	case "true", "1":
		return true, nil
	default:
		return false, fmt.Errorf("vad must be true or false")
	}
}

// VADThreshold reads the multipart vad_threshold field. Empty stays at 0.5.
// Speech starts when the Silero probability is at least this value.
// It ends when the probability stays 0.15 below it, floored at 0.01, for vad_min_silence_duration_ms.
// 0 is a real setting and keeps a frame at any probability.
func VADThreshold(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) || f < 0 || f > 1 {
		return "", fmt.Errorf("vad_threshold must be between 0 and 1")
	}
	return strconv.FormatFloat(f, 'f', -1, 64), nil
}

// VADMinSpeechMS reads the multipart vad_min_speech_duration_ms field.
// Empty stays at 250. A speech region of this many milliseconds or shorter is left out.
// 0 is a real setting and keeps a region of any length.
func VADMinSpeechMS(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 || n > 86_400_000 {
		return "", fmt.Errorf("vad_min_speech_duration_ms must be between 0 and 86400000")
	}
	return strconv.Itoa(n), nil
}

// VADMinSilenceMS reads the multipart vad_min_silence_duration_ms field.
// Empty stays at 100. A shorter pause stays inside the speech region.
// 0 ends a region on the first frame below the end threshold.
// Whisper.cpp then joins neighboring regions when the gap is under 200 ms.
func VADMinSilenceMS(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 || n > 86_400_000 {
		return "", fmt.Errorf("vad_min_silence_duration_ms must be between 0 and 86400000")
	}
	return strconv.Itoa(n), nil
}

// VADMaxSpeechS reads the multipart vad_max_speech_duration_s field.
// Empty leaves the cap off. A speech region longer than this many seconds is split.
// Whisper.cpp uses the whole seconds, then subtracts one analysis window and twice
// the speech padding, so the cut lands earlier. The split prefers a silence longer
// than 98 ms. When no such silence exists, the region ends at the limit.
// A value under 1 second leaves the cap off, so it is rejected.
func VADMaxSpeechS(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) || f < 1 || f > 100000 {
		return "", fmt.Errorf("vad_max_speech_duration_s must be between 1 and 100000")
	}
	return strconv.FormatFloat(f, 'f', -1, 64), nil
}

// VADSpeechPadMS reads the multipart vad_speech_pad_ms field.
// Empty stays at 30. Each speech region grows by this many milliseconds on each side,
// stopping at the clip edges. When the gap between two regions is shorter than twice
// this value, the regions meet in the middle of the gap. 0 adds no padding.
// The same padding is subtracted twice from vad_max_speech_duration_s.
func VADSpeechPadMS(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 || n > 86_400_000 {
		return "", fmt.Errorf("vad_speech_pad_ms must be between 0 and 86400000")
	}
	return strconv.Itoa(n), nil
}

// VADSamplesOverlap reads the multipart vad_samples_overlap field.
// Empty stays at 0.1. Except for the last region, the decoder hears this many
// extra seconds after the region, stopping at the clip edge. 0 copies the region
// as detected. Whisper.cpp still inserts 100 ms of silence between regions.
func VADSamplesOverlap(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) || f < 0 || f > 3600 {
		return "", fmt.Errorf("vad_samples_overlap must be between 0 and 3600")
	}
	return strconv.FormatFloat(f, 'f', -1, 64), nil
}

// DTWPreset reads the multipart dtw field. Empty leaves flash attention on.
// A set value is a whisper.cpp alignment preset. The preset must match the loaded model.
// DTW timestamps require flash attention off.
func DTWPreset(s string) (string, error) {
	s = strings.ToLower(strings.TrimSpace(s))
	switch s {
	case "":
		return "", nil
	case "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large.v1", "large.v2", "large.v3", "large.v3.turbo":
		return s, nil
	default:
		return "", fmt.Errorf("dtw must be a whisper model preset")
	}
}

// DetectedLanguage reads whisper.cpp's "auto-detected language" log line.
// The probability is the softmax score of that language, from 0 to 1.
// A missing line returns an empty code and a nil probability. A probability of 0 is kept.
func DetectedLanguage(stderr string) (string, *float64) {
	const mark = "auto-detected language:"
	idx := strings.LastIndex(stderr, mark)
	if idx < 0 {
		return "", nil
	}
	rest := strings.TrimSpace(stderr[idx+len(mark):])
	code, rest, ok := strings.Cut(rest, " ")
	if !ok || code == "" {
		return "", nil
	}
	rest = strings.TrimSpace(rest)
	const pmark = "(p = "
	if !strings.HasPrefix(rest, pmark) {
		return code, nil
	}
	num := rest[len(pmark):]
	end := strings.IndexByte(num, ')')
	if end < 0 {
		return code, nil
	}
	f, err := strconv.ParseFloat(strings.TrimSpace(num[:end]), 64)
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) || f < 0 || f > 1 {
		return code, nil
	}
	return code, &f
}

// LanguageCode reads the multipart language field. Empty leaves whisper.cpp at
// English. auto detects the spoken language and still transcribes. A language
// name such as english is returned as its code. An unknown code is rejected.
func LanguageCode(s string) (string, error) {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" || s == "auto" {
		return s, nil
	}
	if code, ok := whisperLanguages[s]; ok {
		return code, nil
	}
	return "", fmt.Errorf("language must be a whisper language code or auto")
}

// whisperLanguages maps whisper.cpp codes and English names to the code passed as -l.
var whisperLanguages = map[string]string{
	"en": "en", "english": "en",
	"zh": "zh", "chinese": "zh",
	"de": "de", "german": "de",
	"es": "es", "spanish": "es",
	"ru": "ru", "russian": "ru",
	"ko": "ko", "korean": "ko",
	"fr": "fr", "french": "fr",
	"ja": "ja", "japanese": "ja",
	"pt": "pt", "portuguese": "pt",
	"tr": "tr", "turkish": "tr",
	"pl": "pl", "polish": "pl",
	"ca": "ca", "catalan": "ca",
	"nl": "nl", "dutch": "nl",
	"ar": "ar", "arabic": "ar",
	"sv": "sv", "swedish": "sv",
	"it": "it", "italian": "it",
	"id": "id", "indonesian": "id",
	"hi": "hi", "hindi": "hi",
	"fi": "fi", "finnish": "fi",
	"vi": "vi", "vietnamese": "vi",
	"he": "he", "hebrew": "he",
	"uk": "uk", "ukrainian": "uk",
	"el": "el", "greek": "el",
	"ms": "ms", "malay": "ms",
	"cs": "cs", "czech": "cs",
	"ro": "ro", "romanian": "ro",
	"da": "da", "danish": "da",
	"hu": "hu", "hungarian": "hu",
	"ta": "ta", "tamil": "ta",
	"no": "no", "norwegian": "no",
	"th": "th", "thai": "th",
	"ur": "ur", "urdu": "ur",
	"hr": "hr", "croatian": "hr",
	"bg": "bg", "bulgarian": "bg",
	"lt": "lt", "lithuanian": "lt",
	"la": "la", "latin": "la",
	"mi": "mi", "maori": "mi",
	"ml": "ml", "malayalam": "ml",
	"cy": "cy", "welsh": "cy",
	"sk": "sk", "slovak": "sk",
	"te": "te", "telugu": "te",
	"fa": "fa", "persian": "fa",
	"lv": "lv", "latvian": "lv",
	"bn": "bn", "bengali": "bn",
	"sr": "sr", "serbian": "sr",
	"az": "az", "azerbaijani": "az",
	"sl": "sl", "slovenian": "sl",
	"kn": "kn", "kannada": "kn",
	"et": "et", "estonian": "et",
	"mk": "mk", "macedonian": "mk",
	"br": "br", "breton": "br",
	"eu": "eu", "basque": "eu",
	"is": "is", "icelandic": "is",
	"hy": "hy", "armenian": "hy",
	"ne": "ne", "nepali": "ne",
	"mn": "mn", "mongolian": "mn",
	"bs": "bs", "bosnian": "bs",
	"kk": "kk", "kazakh": "kk",
	"sq": "sq", "albanian": "sq",
	"sw": "sw", "swahili": "sw",
	"gl": "gl", "galician": "gl",
	"mr": "mr", "marathi": "mr",
	"pa": "pa", "punjabi": "pa",
	"si": "si", "sinhala": "si",
	"km": "km", "khmer": "km",
	"sn": "sn", "shona": "sn",
	"yo": "yo", "yoruba": "yo",
	"so": "so", "somali": "so",
	"af": "af", "afrikaans": "af",
	"oc": "oc", "occitan": "oc",
	"ka": "ka", "georgian": "ka",
	"be": "be", "belarusian": "be",
	"tg": "tg", "tajik": "tg",
	"sd": "sd", "sindhi": "sd",
	"gu": "gu", "gujarati": "gu",
	"am": "am", "amharic": "am",
	"yi": "yi", "yiddish": "yi",
	"lo": "lo", "lao": "lo",
	"uz": "uz", "uzbek": "uz",
	"fo": "fo", "faroese": "fo",
	"ht": "ht", "haitian creole": "ht",
	"ps": "ps", "pashto": "ps",
	"tk": "tk", "turkmen": "tk",
	"nn": "nn", "nynorsk": "nn",
	"mt": "mt", "maltese": "mt",
	"sa": "sa", "sanskrit": "sa",
	"lb": "lb", "luxembourgish": "lb",
	"my": "my", "myanmar": "my",
	"bo": "bo", "tibetan": "bo",
	"tl": "tl", "tagalog": "tl",
	"mg": "mg", "malagasy": "mg",
	"as": "as", "assamese": "as",
	"tt": "tt", "tatar": "tt",
	"haw": "haw", "hawaiian": "haw",
	"ln": "ln", "lingala": "ln",
	"ha": "ha", "hausa": "ha",
	"ba": "ba", "bashkir": "ba",
	"jw": "jw", "javanese": "jw",
	"su": "su", "sundanese": "su",
	"yue": "yue", "cantonese": "yue",
}

// DetectLanguageFlag reads the multipart detect_language field. Empty is false.
// True asks whisper.cpp for the spoken language and skips the transcript.
func DetectLanguageFlag(s string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "false", "0":
		return false, nil
	case "true", "1":
		return true, nil
	default:
		return false, fmt.Errorf("detect_language must be true or false")
	}
}

// PrintSpecialFlag reads the multipart print_special field. Empty is false.
// True keeps special tokens, such as <|nospeech|> and timestamp marks, in the segment text.
func PrintSpecialFlag(s string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "false", "0":
		return false, nil
	case "true", "1":
		return true, nil
	default:
		return false, fmt.Errorf("print_special must be true or false")
	}
}

// NoTimestampsFlag reads the multipart no_timestamps field. Empty is false.
// True starts the decoder with the no-timestamps token and blocks timestamp tokens.
// Each segment then advances 30 seconds. A first-release distilled model forces this on.
func NoTimestampsFlag(s string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "false", "0":
		return false, nil
	case "true", "1":
		return true, nil
	default:
		return false, fmt.Errorf("no_timestamps must be true or false")
	}
}

// OffsetN reads the multipart offset_n field. Empty, or 0, starts SRT cue
// numbers at 1. A set value is added to each cue number. VTT stays unnumbered.
func OffsetN(s string) (int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < -1_000_000 || n > 1_000_000 {
		return 0, fmt.Errorf("offset_n must be between -1000000 and 1000000")
	}
	return n, nil
}

// SuppressRegex reads the multipart suppress_regex field. Empty, or only
// space, leaves every token available. A set value is a C++ regular expression
// matched against each whole token. A match is left out of the transcript.
// An invalid expression fails the transcription.
func SuppressRegex(s string) (string, error) {
	if strings.TrimSpace(s) == "" {
		return "", nil
	}
	if len(s) > 256 || strings.ContainsRune(s, 0) {
		return "", fmt.Errorf("suppress_regex must be at most 256 bytes")
	}
	return s, nil
}

// DurationMS reads the multipart duration_ms field. Empty, or 0, continues
// to the end of the clip. A set value is milliseconds from 1 to 86400000,
// measured from offset_t. A window shorter than 100 ms yields an empty transcript.
func DurationMS(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 || n > 86_400_000 {
		return "", fmt.Errorf("duration_ms must be between 0 and 86400000")
	}
	if n == 0 {
		return "", nil
	}
	return strconv.Itoa(n), nil
}

// OffsetT reads the multipart offset_t field. Empty, or 0, starts at the
// beginning of the clip. A set value is milliseconds from 1 to 86400000.
// A start that leaves less than 100 ms yields an empty transcript.
func OffsetT(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 || n > 86_400_000 {
		return "", fmt.Errorf("offset_t must be between 0 and 86400000")
	}
	if n == 0 {
		return "", nil
	}
	return strconv.Itoa(n), nil
}

// AudioCtx reads the multipart audio_ctx field. Empty, or 0, uses every encoder
// frame. A set value is from 1 to 1500, the standard Whisper encoder length.
// A value above the loaded model's audio context fails the transcription.
func AudioCtx(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 || n > 1500 {
		return "", fmt.Errorf("audio_ctx must be between 0 and 1500")
	}
	if n == 0 {
		return "", nil
	}
	return strconv.Itoa(n), nil
}

// MaxContext reads the multipart max_context field. Empty, or -1, leaves
// whisper.cpp's cap of 16384 tokens, which is also limited to half the model's
// text context. 0 leaves earlier text out of the next window. A positive value
// is that cap.
func MaxContext(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < -1 || n > 16384 {
		return "", fmt.Errorf("max_context must be between -1 and 16384")
	}
	if n < 0 {
		return "", nil
	}
	return strconv.Itoa(n), nil
}

// WordThold reads the multipart word_thold field. Empty leaves the whisper.cpp
// default of 0.01. A set value is between 0 and 1. A token timestamp is kept
// when its probability is above this value.
func WordThold(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || f < 0 || f > 1 {
		return "", fmt.Errorf("word_thold must be between 0 and 1")
	}
	return strconv.FormatFloat(f, 'f', -1, 64), nil
}

// MaxLen reads the multipart max_len field. Empty, or 0, leaves the character
// cap off. A set value is from 1 to 1000 and wraps a longer segment. With word
// timestamps the wrap stays on a word.
func MaxLen(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 || n > 1000 {
		return "", fmt.Errorf("max_len must be between 0 and 1000")
	}
	if n == 0 {
		return "", nil
	}
	return strconv.Itoa(n), nil
}

// BestOf reads the multipart best_of field. Empty leaves the whisper.cpp
// default of 5. A set value is from 1 to 16. That count is the number of
// candidates when the temperature is above 0.
func BestOf(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 1 || n > 16 {
		return "", fmt.Errorf("best_of must be between 1 and 16")
	}
	return strconv.Itoa(n), nil
}

// BeamSize reads the multipart beam_size field. Empty leaves the whisper.cpp
// default of 5. A set value is from 1 to 16. That width is the number of
// beam-search candidates when the temperature is 0. A value of 1 decodes greedily.
func BeamSize(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 1 || n > 16 {
		return "", fmt.Errorf("beam_size must be between 1 and 16")
	}
	return strconv.Itoa(n), nil
}

// NoSpeechThold reads the multipart no_speech_thold field. Empty leaves the
// whisper.cpp default of 0.6. A set value is between 0 and 1. A segment is
// silence when its no-speech probability is above this value and its average
// log probability is below logprob_thold.
func NoSpeechThold(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || f < 0 || f > 1 {
		return "", fmt.Errorf("no_speech_thold must be between 0 and 1")
	}
	return strconv.FormatFloat(f, 'f', -1, 64), nil
}

// LogprobThold reads the multipart logprob_thold field. Empty leaves the
// whisper.cpp default of -1. A set value is between -10 and 0. A segment whose
// average log probability is below this value fails when the clip still looks
// like speech, and whisper.cpp retries.
func LogprobThold(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || f < -10 || f > 0 {
		return "", fmt.Errorf("logprob_thold must be between -10 and 0")
	}
	return strconv.FormatFloat(f, 'f', -1, 64), nil
}

// EntropyThold reads the multipart entropy_thold field. Empty leaves the
// whisper.cpp default of 2.4. A set value is between 0 and 4. A segment whose
// token entropy is below this value fails, and whisper.cpp retries.
func EntropyThold(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || f < 0 || f > 4 {
		return "", fmt.Errorf("entropy_thold must be between 0 and 4")
	}
	return strconv.FormatFloat(f, 'f', -1, 64), nil
}

// TemperatureInc reads the multipart temperature_inc field. Empty leaves the
// whisper.cpp default of 0.2. A set value is between 0 and 1.
func TemperatureInc(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || f < 0 || f > 1 {
		return "", fmt.Errorf("temperature_inc must be between 0 and 1")
	}
	return strconv.FormatFloat(f, 'f', -1, 64), nil
}

// CarryInitialFlag reads the multipart carry_initial_prompt field. Empty is false.
// True keeps the initial prompt in front of every window.
func CarryInitialFlag(s string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "false", "0":
		return false, nil
	case "true", "1":
		return true, nil
	default:
		return false, fmt.Errorf("carry_initial_prompt must be true or false")
	}
}

// NoFallbackFlag reads the multipart no_fallback field. Empty is false.
func NoFallbackFlag(s string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "false", "0":
		return false, nil
	case "true", "1":
		return true, nil
	default:
		return false, fmt.Errorf("no_fallback must be true or false")
	}
}

// SuppressNSTFlag reads the multipart suppress_nst field. Empty is false.
func SuppressNSTFlag(s string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "false", "0":
		return false, nil
	case "true", "1":
		return true, nil
	default:
		return false, fmt.Errorf("suppress_nst must be true or false")
	}
}

// DiarizeFlag reads the multipart diarize field. Empty is false.
// SplitOnWordFlag reads the multipart split_on_word field. Empty is false.
// True keeps a max_len wrap on a token that starts with a space.
func SplitOnWordFlag(s string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "false", "0":
		return false, nil
	case "true", "1":
		return true, nil
	default:
		return false, fmt.Errorf("split_on_word must be true or false")
	}
}

// TinyDiarizeFlag reads the multipart tinydiarize field. Empty is false.
// True lets a tinydiarize model emit a speaker-turn token. The segment text
// then ends with " [SPEAKER_TURN]". It cannot be combined with diarize.
func TinyDiarizeFlag(s string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "false", "0":
		return false, nil
	case "true", "1":
		return true, nil
	default:
		return false, fmt.Errorf("tinydiarize must be true or false")
	}
}

func DiarizeFlag(s string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "false", "0":
		return false, nil
	case "true", "1":
		return true, nil
	default:
		return false, fmt.Errorf("diarize must be true or false")
	}
}

func tokenIDs(tokens []token) []int {
	var out []int
	for _, tok := range tokens {
		if tok.ID == nil {
			continue
		}
		out = append(out, *tok.ID)
	}
	return out
}

func tokenLogprobs(tokens []token) []Logprob {
	var out []Logprob
	for _, tok := range tokens {
		trimmed := strings.TrimSpace(tok.Text)
		if trimmed == "" || isWhisperSpecial(trimmed) || tok.P <= 0 || tok.P > 1 {
			continue
		}
		out = append(out, Logprob{Token: tok.Text, Logprob: math.Log(tok.P)})
	}
	return out
}

// KnownDuration is the clip length to report. A positive measured length,
// such as a WAV byte rate, wins. Otherwise the last cue end is used. Zero
// means neither is known.
func KnownDuration(measured, segmentEnd float64) float64 {
	if measured > 0 {
		return measured
	}
	if segmentEnd > 0 {
		return segmentEnd
	}
	return 0
}

// HeaderAudioSeconds reads an X-Audio-Seconds value. Empty, non-numeric,
// non-positive, and values above one day are ignored.
func HeaderAudioSeconds(v string) (float64, bool) {
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

// WavSeconds is the length of a RIFF WAVE file, from the fmt byte rate and
// the data chunk. A non-wav body, or a fmt chunk with a zero byte rate, returns false.
func WavSeconds(b []byte) (float64, bool) {
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

// AudioSeconds is the length of a speech body. PCM uses rate= on the content
// type when present, otherwise rate, and the sample width (float32 when the
// width is omitted). A WAV uses the fmt byte rate and the data chunk.
func AudioSeconds(contentType string, audio []byte, rate int) (float64, bool) {
	if r, ok := contentTypeRate(contentType); ok {
		rate = r
	}
	lower := strings.ToLower(contentType)
	if strings.Contains(lower, "audio/pcm") {
		return pcmSeconds(lower, len(audio), rate)
	}
	return WavSeconds(audio)
}

// StreamMeter measures a speech body while it is copied. PCM length comes from
// the byte count and the content-type rate. A WAV whose data chunk is the
// streaming sentinel (or longer than the bytes received) uses the fmt byte rate
// and the bytes after the data header.
type StreamMeter struct {
	contentType string
	rate        int
	total       int
	head        []byte
}

func NewStreamMeter(contentType string, rate int) *StreamMeter {
	return &StreamMeter{contentType: contentType, rate: rate}
}

func (m *StreamMeter) Add(p []byte) {
	if m == nil || len(p) == 0 {
		return
	}
	m.total += len(p)
	if len(m.head) < 64 {
		need := 64 - len(m.head)
		if need > len(p) {
			need = len(p)
		}
		m.head = append(m.head, p[:need]...)
	}
}

func (m *StreamMeter) Seconds() (float64, bool) {
	if m == nil || m.total <= 0 {
		return 0, false
	}
	if r, ok := contentTypeRate(m.contentType); ok {
		m.rate = r
	}
	lower := strings.ToLower(m.contentType)
	if strings.Contains(lower, "audio/pcm") {
		return pcmSeconds(lower, m.total, m.rate)
	}
	if strings.Contains(lower, "wav") {
		return wavStreamSeconds(m.head, m.total)
	}
	return 0, false
}

func pcmSeconds(lower string, n, rate int) (float64, bool) {
	if rate <= 0 || n <= 0 {
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
	if n%frame != 0 {
		return 0, false
	}
	return float64(n/frame) / float64(rate), true
}

// wavStreamSeconds reads a RIFF header that may declare an unknown data length.
// total is the number of bytes in the whole response, which can extend past head.
func wavStreamSeconds(head []byte, total int) (float64, bool) {
	if len(head) < 12 || string(head[0:4]) != "RIFF" || string(head[8:12]) != "WAVE" || total < 12 {
		return 0, false
	}
	var byteRate uint32
	var gotFmt bool
	off := 12
	for off+8 <= len(head) {
		id := string(head[off : off+4])
		size := binary.LittleEndian.Uint32(head[off+4 : off+8])
		off += 8
		if id == "data" {
			if !gotFmt || byteRate == 0 || total < off {
				return 0, false
			}
			n := total - off
			if size != 0 && size != ^uint32(0) && int(size) < n {
				n = int(size)
			}
			if n <= 0 {
				return 0, false
			}
			return float64(n) / float64(byteRate), true
		}
		if int(size) < 0 || off+int(size) > len(head) {
			return 0, false
		}
		if id == "fmt " {
			if size < 16 {
				return 0, false
			}
			byteRate = binary.LittleEndian.Uint32(head[off+8 : off+12])
			gotFmt = true
		}
		off += int(size)
		if size%2 == 1 {
			off++
		}
	}
	return 0, false
}

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

// IncludeLogprobs reports whether the form values ask for token logprobs.
// The only accepted value is logprobs. Empty is false.
func IncludeLogprobs(values []string) (bool, error) {
	on := false
	for _, v := range values {
		for _, part := range strings.Split(v, ",") {
			switch strings.ToLower(strings.TrimSpace(part)) {
			case "":
			case "logprobs":
				on = true
			default:
				return false, fmt.Errorf("include must be logprobs")
			}
		}
	}
	return on, nil
}

func isNoSpeech(text string) bool {
	s := strings.ToLower(strings.Trim(text, "[]<>|_ "))
	return s == "nospeech" || s == "no_speech" || s == "no speech"
}

func wordsFromTokens(tokens []token) []Word {
	var out []Word
	var cur Word
	open := false
	flush := func() {
		if !open {
			return
		}
		cur.Word = strings.TrimSpace(cur.Word)
		if cur.Word != "" {
			out = append(out, cur)
		}
		open = false
	}
	for _, tok := range tokens {
		text := tok.Text
		trimmed := strings.TrimSpace(text)
		if trimmed == "" || isWhisperSpecial(trimmed) {
			continue
		}
		start, end := tokenSpan(tok)
		if !open || strings.HasPrefix(text, " ") || strings.HasPrefix(text, "\n") {
			if open && start > cur.Start && tok.DTW != nil && *tok.DTW >= 0 {
				cur.End = start
			}
			flush()
			cur = Word{Word: trimmed, Start: start, End: end}
			open = true
			continue
		}
		cur.Word += text
		if tok.DTW != nil && *tok.DTW >= 0 && start > cur.Start {
			cur.End = start
		} else if tok.Offsets.To >= tok.Offsets.From {
			cur.End = float64(tok.Offsets.To) / 1000
		}
	}
	flush()
	return out
}

func tokenSpan(tok token) (float64, float64) {
	start := float64(tok.Offsets.From) / 1000
	end := float64(tok.Offsets.To) / 1000
	if tok.DTW != nil && *tok.DTW >= 0 {
		start = *tok.DTW / 100
		if end < start {
			end = start
		}
	}
	return start, end
}

func isWhisperSpecial(text string) bool {
	return strings.HasPrefix(text, "<|") || (strings.HasPrefix(text, "[") && strings.HasSuffix(text, "]"))
}

// CueSpan is one transcript cue. seconds is the WAV length. A positive length
// ends the cue there. Zero leaves the end at the start, because the backend
// did not report a time.
func CueSpan(text string, seconds float64) Segment {
	seg := Segment{Text: text}
	if seconds > 0 {
		seg.End = seconds
	}
	return seg
}

// WriteSRT renders SubRip cues. Times are seconds. An empty list is an empty file.
// Cue numbers start at 1.
func WriteSRT(segs []Segment) string {
	return WriteSRTOffset(segs, 0)
}

// WriteSRTOffset renders SubRip cues whose first number is 1 plus offset.
func WriteSRTOffset(segs []Segment, offset int) string {
	return writeCues(segs, "", ",", offset)
}

// WriteVTT renders a WebVTT file. Times are seconds. An empty list is the header only.
func WriteVTT(segs []Segment) string {
	return writeCues(segs, "WEBVTT\n\n", ".", 0)
}

func writeCues(segs []Segment, header, sep string, offset int) string {
	var b strings.Builder
	b.WriteString(header)
	n := 0
	for _, seg := range segs {
		text := strings.TrimSpace(seg.Text)
		if text == "" {
			continue
		}
		start := seg.Start
		end := seg.End
		if start < 0 {
			start = 0
		}
		if end < start {
			end = start
		}
		n++
		if header == "" {
			fmt.Fprintf(&b, "%d\n", n+offset)
		}
		fmt.Fprintf(&b, "%s --> %s\n%s\n\n", stamp(start, sep), stamp(end, sep), text)
	}
	return b.String()
}

func stamp(sec float64, sep string) string {
	ms := int(math.Round(sec * 1000))
	if ms < 0 {
		ms = 0
	}
	h := ms / 3600000
	ms %= 3600000
	m := ms / 60000
	ms %= 60000
	s := ms / 1000
	ms %= 1000
	return fmt.Sprintf("%02d:%02d:%02d%s%03d", h, m, s, sep, ms)
}
