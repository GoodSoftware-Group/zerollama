package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/ollama/ollama/server/voicepipe"
)

// followTurn is a later user message and the response.create that answers it.
func followTurn(text string, modalities []string) []map[string]any {
	create := map[string]any{"type": "response.create"}
	if len(modalities) > 0 {
		create["response"] = map[string]any{"output_modalities": modalities}
	}
	return []map[string]any{textItem(text), create}
}

// textItem is a user message on a conversation socket. Transcription sessions
// reject conversation.item.create.
func textItem(text string) map[string]any {
	return map[string]any{
		"type": "conversation.item.create",
		"item": map[string]any{
			"type": "message",
			"role": "user",
			"content": []map[string]string{{
				"type": "input_text",
				"text": text,
			}},
		},
	}
}

// userTurn is what the probe sends after session.created.
// Audio is PCM16 at the session rate. A conversation with no audio is a text
// turn plus response.create. Transcription is audio and a final commit.
func userTurn(text string, samples []int16, rate, chunkMs int, conversation bool, modalities []string, serverVAD, duplex bool) ([]map[string]any, error) {
	text = strings.TrimSpace(text)
	if text != "" && !conversation {
		return nil, fmt.Errorf("a text turn needs a conversation; leave -intent empty")
	}
	if text == "" && len(samples) == 0 {
		return nil, fmt.Errorf("a turn needs -wav or -text")
	}
	var msgs []map[string]any
	if text != "" {
		msgs = append(msgs, textItem(text))
	}
	if len(samples) == 0 {
		create := map[string]any{"type": "response.create"}
		if len(modalities) > 0 {
			create["response"] = map[string]any{"output_modalities": modalities}
		}
		return append(msgs, create), nil
	}
	return append(msgs, turnMessages(samples, rate, chunkMs, conversation, modalities, serverVAD, duplex)...), nil
}

// turnMessages is one audio turn.
// A transcription session ends with input_audio_buffer.commit final=true.
// A conversation does not answer on commit: llama-server waits for response.create.
// modalities is empty for the model's own choice, or text / audio / both.
// serverVAD appends the default quiet stretch as its own chunks and does not
// commit. A quiet stretch mixed into the last speech chunk would still count
// as speech. duplex replies on commit, so the probe does not send response.create.
func turnMessages(samples []int16, rate, chunkMs int, conversation bool, modalities []string, serverVAD, duplex bool) []map[string]any {
	var msgs []map[string]any
	for _, chunk := range chunkSamples(samples, rate, chunkMs) {
		msgs = append(msgs, appendAudio(chunk))
	}
	if serverVAD {
		n := rate * voicepipe.DefaultTurn().SilenceMS / 1000
		if n < 1 {
			n = 1
		}
		silent := make([]int16, n)
		for _, chunk := range chunkSamples(silent, rate, chunkMs) {
			msgs = append(msgs, appendAudio(chunk))
		}
		return msgs
	}
	commit := map[string]any{"type": "input_audio_buffer.commit"}
	if !conversation && !duplex {
		commit["final"] = true
	}
	msgs = append(msgs, commit)
	if conversation && !duplex {
		create := map[string]any{"type": "response.create"}
		if len(modalities) > 0 {
			create["response"] = map[string]any{"output_modalities": modalities}
		}
		msgs = append(msgs, create)
	}
	return msgs
}

func appendAudio(samples []int16) map[string]any {
	return map[string]any{
		"type":  "input_audio_buffer.append",
		"audio": base64.StdEncoding.EncodeToString(pcm16Bytes(samples)),
	}
}

// applyModalities stores output modalities on session.update. A server_vad
// reply is created by the server, which uses the session value.
func applyModalities(update map[string]any, modalities []string) map[string]any {
	if len(modalities) == 0 {
		return update
	}
	if update == nil {
		update = map[string]any{"type": "session.update", "session": map[string]any{}}
	}
	session, _ := update["session"].(map[string]any)
	if session == nil {
		session = map[string]any{}
		update["session"] = session
	}
	session["output_modalities"] = modalities
	return update
}

// realtimeExtra is the duplex query. Voice is the speech model, Speaker is its voice id.
type realtimeExtra struct {
	Pipeline string
	Voice    string
	ASR      string
	Speaker  string
	Rate     int
}

func cleanPipeline(s string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "":
		return "", nil
	case "duplex":
		return "duplex", nil
	default:
		return "", fmt.Errorf("pipeline must be duplex")
	}
}

func parseProbeRate(flag string, wavRate int) (int, error) {
	s := strings.TrimSpace(flag)
	n := wavRate
	if s != "" {
		parsed, err := strconv.Atoi(s)
		if err != nil {
			return 0, fmt.Errorf("rate must be an integer")
		}
		n = parsed
	}
	if n <= 0 || n > 192000 {
		return 0, fmt.Errorf("duplex requires rate for the input pcm16")
	}
	return n, nil
}

func cleanTurn(s string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "none":
		return "", nil
	case "server_vad":
		return "server_vad", nil
	default:
		return "", fmt.Errorf("turn must be server_vad or none")
	}
}

// sessionUpdate is the session.update sent before the first audio or message.
// An empty result means the session keeps llama-server's defaults.
// toolsJSON is a Responses API tool array. maxTokens is a positive integer or "inf".
// language is the live transcription tag (LFM3 renders it into the ASR prompt).
// sessionType is "transcription" or "realtime"; empty leaves the server default.
func sessionUpdate(instructions, toolsJSON, toolChoice, maxTokens, language, sessionType string) (map[string]any, error) {
	session := map[string]any{}
	if s := strings.TrimSpace(instructions); s != "" {
		session["instructions"] = s
	}
	if s := strings.TrimSpace(toolsJSON); s != "" {
		var tools []any
		if err := json.Unmarshal([]byte(s), &tools); err != nil {
			return nil, fmt.Errorf("tools: %w", err)
		}
		session["tools"] = tools
	}
	if s := strings.TrimSpace(toolChoice); s != "" {
		switch s {
		case "auto", "none", "required":
			session["tool_choice"] = s
		default:
			return nil, fmt.Errorf("tool_choice must be auto, none, or required")
		}
	}
	if s := strings.TrimSpace(maxTokens); s != "" {
		if s == "inf" {
			session["max_output_tokens"] = "inf"
		} else {
			n, err := strconv.Atoi(s)
			if err != nil || n < 1 {
				return nil, fmt.Errorf("max_output_tokens must be a positive integer or inf")
			}
			session["max_output_tokens"] = n
		}
	}
	lang, err := cleanLangTag(language)
	if err != nil {
		return nil, err
	}
	if lang != "" {
		session["audio"] = map[string]any{
			"input": map[string]any{
				"transcription": map[string]any{"language": lang},
			},
		}
	}
	switch strings.TrimSpace(sessionType) {
	case "":
	case "transcription", "realtime":
		session["type"] = strings.TrimSpace(sessionType)
	default:
		return nil, fmt.Errorf("session type must be transcription or realtime")
	}
	if len(session) == 0 {
		return nil, nil
	}
	return map[string]any{"type": "session.update", "session": session}, nil
}

func cleanLangTag(s string) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	if len(s) > 32 {
		return "", fmt.Errorf("lang is too long")
	}
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
		default:
			return "", fmt.Errorf("lang must be letters, digits, '-' or '_'")
		}
	}
	return strings.ToLower(s), nil
}

// toolResult is the function_call_output that follows response.done.
// llama-server answers it on the next response.create.
func toolResult(callID, output string) map[string]any {
	return map[string]any{
		"type": "conversation.item.create",
		"item": map[string]any{
			"type":    "function_call_output",
			"call_id": callID,
			"output":  output,
		},
	}
}

func parseModalities(s string) ([]string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	var out []string
	seen := map[string]bool{}
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part != "text" && part != "audio" {
			return nil, errModality(part)
		}
		if seen[part] {
			continue
		}
		seen[part] = true
		out = append(out, part)
	}
	return out, nil
}

type modalityError string

func (e modalityError) Error() string {
	return "output modality must be text or audio, got " + string(e)
}

func errModality(s string) error { return modalityError(s) }

// responseStatus reads response.done's status. "cancelled" is a barge-in.
func responseStatus(ev map[string]any) string {
	resp, _ := ev["response"].(map[string]any)
	status, _ := resp["status"].(string)
	return status
}
