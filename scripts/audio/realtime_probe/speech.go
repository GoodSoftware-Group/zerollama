package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

func parseSpeechLimits(seedFlag, maxTokensFlag string) (*int64, *int, error) {
	var seed *int64
	if strings.TrimSpace(seedFlag) != "" {
		n, err := strconv.ParseInt(strings.TrimSpace(seedFlag), 10, 64)
		if err != nil {
			return nil, nil, fmt.Errorf("seed: %w", err)
		}
		seed = &n
	}
	var maxTokens *int
	if strings.TrimSpace(maxTokensFlag) != "" {
		n, err := strconv.Atoi(strings.TrimSpace(maxTokensFlag))
		if err != nil {
			return nil, nil, fmt.Errorf("max-tokens: %w", err)
		}
		maxTokens = &n
	}
	return seed, maxTokens, nil
}

// speechSample is optional acoustic sampling. Nil pointers are omitted.
type speechSample struct {
	TopK          *int
	TopP          *float64
	RepeatPenalty *float64
}

func parseSpeechSample(topKFlag, topPFlag, repeatFlag string) (speechSample, error) {
	var sample speechSample
	if strings.TrimSpace(topKFlag) != "" {
		n, err := strconv.Atoi(strings.TrimSpace(topKFlag))
		if err != nil {
			return speechSample{}, fmt.Errorf("top-k: %w", err)
		}
		sample.TopK = &n
	}
	if strings.TrimSpace(topPFlag) != "" {
		n, err := strconv.ParseFloat(strings.TrimSpace(topPFlag), 64)
		if err != nil {
			return speechSample{}, fmt.Errorf("top-p: %w", err)
		}
		sample.TopP = &n
	}
	if strings.TrimSpace(repeatFlag) != "" {
		n, err := strconv.ParseFloat(strings.TrimSpace(repeatFlag), 64)
		if err != nil {
			return speechSample{}, fmt.Errorf("repeat-penalty: %w", err)
		}
		sample.RepeatPenalty = &n
	}
	return sample, nil
}

func parseSentenceSilence(s string) (*float64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	n, err := strconv.ParseFloat(s, 64)
	if err != nil || n < 0 || n > 5 {
		return nil, fmt.Errorf("sentence_silence must be between 0 and 5")
	}
	return &n, nil
}

func parseNoiseScale(s string) (*float64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	n, err := strconv.ParseFloat(s, 64)
	if err != nil || n < 0 || n > 2 {
		return nil, fmt.Errorf("noise_scale must be between 0 and 2")
	}
	return &n, nil
}

func parseNoiseW(s string) (*float64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	n, err := strconv.ParseFloat(s, 64)
	if err != nil || n < 0 || n > 2 {
		return nil, fmt.Errorf("noise_w must be between 0 and 2")
	}
	return &n, nil
}

func parseSpeakerID(s string) (*int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 || n > 4095 {
		return nil, fmt.Errorf("speaker_id must be between 0 and 4095")
	}
	if n == 0 {
		return nil, nil
	}
	return &n, nil
}

func parsePhonemeSilence(s string) (map[string]float64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	var m map[string]float64
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		return nil, fmt.Errorf("phoneme_silence must be a JSON object")
	}
	if len(m) > 32 {
		return nil, fmt.Errorf("phoneme_silence accepts at most 32 phonemes")
	}
	for k, v := range m {
		r, size := utf8.DecodeRuneInString(k)
		if r == utf8.RuneError || size != len(k) {
			return nil, fmt.Errorf("phoneme_silence keys must be a single character")
		}
		if v < 0 || v > 5 {
			return nil, fmt.Errorf("phoneme_silence must be between 0 and 5")
		}
	}
	return m, nil
}

func cleanSpeechFormat(s string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "json":
		return "", nil
	case "wav", "pcm", "mp3", "opus", "aac", "flac":
		return strings.ToLower(strings.TrimSpace(s)), nil
	default:
		return "", fmt.Errorf("speech response_format must be wav, pcm, mp3, opus, aac, or flac")
	}
}

func probeSpeech(host, model, text, voice, lang, speakerPath, outPath string, stream bool, format string, seed *int64, maxTokens *int, sample speechSample, silence, noise, width *float64, speaker *int, phonemes map[string]float64, timeout time.Duration) (probeReport, error) {
	body := map[string]any{
		"model": model,
		"input": text,
	}
	if strings.TrimSpace(voice) != "" {
		body["voice"] = strings.TrimSpace(voice)
	}
	if strings.TrimSpace(lang) != "" {
		body["lang"] = strings.TrimSpace(lang)
	}
	if stream {
		body["stream"] = true
	}
	if format != "" {
		body["response_format"] = format
	}
	if seed != nil {
		body["seed"] = *seed
	}
	if maxTokens != nil {
		body["max_new_tokens"] = *maxTokens
	}
	if sample.TopK != nil {
		body["top_k"] = *sample.TopK
	}
	if sample.TopP != nil {
		body["top_p"] = *sample.TopP
	}
	if sample.RepeatPenalty != nil {
		body["repeat_penalty"] = *sample.RepeatPenalty
	}
	if silence != nil {
		body["sentence_silence"] = *silence
	}
	if noise != nil {
		body["noise_scale"] = *noise
	}
	if width != nil {
		body["noise_w"] = *width
	}
	if speaker != nil {
		body["speaker_id"] = *speaker
	}
	if len(phonemes) > 0 {
		body["phoneme_silence"] = phonemes
	}
	if speakerPath != "" {
		raw, err := os.ReadFile(speakerPath)
		if err != nil {
			return probeReport{}, err
		}
		body["speaker_ref"] = base64.StdEncoding.EncodeToString(raw)
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return probeReport{}, err
	}
	req, err := http.NewRequest(http.MethodPost, strings.TrimRight(host, "/")+"/v1/audio/speech", bytes.NewReader(payload))
	if err != nil {
		return probeReport{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: timeout}
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return probeReport{}, err
	}
	defer resp.Body.Close()
	var audio []byte
	var first int64
	buf := make([]byte, 32*1024)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 && first == 0 {
			first = time.Since(start).Milliseconds()
			if first == 0 {
				first = 1
			}
		}
		audio = append(audio, buf[:n]...)
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return probeReport{}, rerr
		}
	}
	elapsed := time.Since(start).Milliseconds()
	report := probeReport{
		Mode:         "speech",
		Model:        model,
		FirstAudioMs: first,
		TotalMs:      elapsed,
		AudioBytes:   len(audio),
		Status:       resp.Header.Get("Content-Type"),
	}
	if rate, ok := contentTypeRate(report.Status); ok {
		report.OutputRate = rate
	} else if rate, ok := wavHeaderRate(audio); ok {
		report.OutputRate = rate
	}
	if resp.StatusCode != http.StatusOK {
		report.Error = strings.TrimSpace(string(audio))
		return report, fmt.Errorf("speech HTTP %d", resp.StatusCode)
	}
	if sec, ok := headerAudioSeconds(resp.Header.Get("X-Audio-Seconds")); ok {
		report.Seconds = sec
	} else if sec, ok := headerAudioSeconds(resp.Trailer.Get("X-Audio-Seconds")); ok {
		report.Seconds = sec
	} else if sec, ok := audioSeconds(report.Status, audio, report.OutputRate); ok {
		report.Seconds = sec
	}
	if outPath != "" && len(audio) > 0 {
		payload := audio
		if report.OutputRate > 0 && len(audio)%4 == 0 && strings.Contains(strings.ToLower(report.Status), "audio/pcm") {
			payload = wavWrapFloat(audio, report.OutputRate)
		}
		if err := os.WriteFile(outPath, payload, 0o644); err != nil {
			return report, err
		}
		report.OutFile = outPath
	}
	return report, nil
}
