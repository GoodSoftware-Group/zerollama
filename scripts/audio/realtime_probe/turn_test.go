package main

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
)

func TestSessionUpdate(t *testing.T) {
	got, err := sessionUpdate("be brief", `[{"type":"function","name":"lookup","parameters":{"type":"object"}}]`, "required", "16", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if got["type"] != "session.update" {
		t.Fatalf("%v", got["type"])
	}
	session := got["session"].(map[string]any)
	if session["instructions"] != "be brief" || session["tool_choice"] != "required" || session["max_output_tokens"] != 16 {
		t.Fatalf("%v", session)
	}
	if _, err := sessionUpdate("", "", "always", "", "", ""); err == nil {
		t.Fatal("expected tool_choice to fail")
	}
	if _, err := sessionUpdate("", "{", "", "", "", ""); err == nil {
		t.Fatal("expected tools JSON to fail")
	}
	inf, err := sessionUpdate("", "", "", "inf", "", "")
	if err != nil || inf["session"].(map[string]any)["max_output_tokens"] != "inf" {
		t.Fatalf("%v %v", inf, err)
	}
	empty, err := sessionUpdate(" ", "", "", "", "", "")
	if err != nil || empty != nil {
		t.Fatalf("%v %v", empty, err)
	}
	lang, err := sessionUpdate("", "", "", "", "EN", "transcription")
	if err != nil {
		t.Fatal(err)
	}
	audio := lang["session"].(map[string]any)["audio"].(map[string]any)
	in := audio["input"].(map[string]any)
	tr := in["transcription"].(map[string]any)
	if tr["language"] != "en" || lang["session"].(map[string]any)["type"] != "transcription" {
		t.Fatalf("%v", lang["session"])
	}
	if _, err := sessionUpdate("", "", "", "", "en US", ""); err == nil {
		t.Fatal("expected lang to fail")
	}
}

func TestFollowTurn(t *testing.T) {
	got := followTurn("and then", []string{"text"})
	if len(got) != 2 || got[0]["type"] != "conversation.item.create" || got[1]["type"] != "response.create" {
		t.Fatalf("%v", got)
	}
	item := got[0]["item"].(map[string]any)
	if item["role"] != "user" {
		t.Fatalf("%v", item)
	}
	resp := got[1]["response"].(map[string]any)
	mods := resp["output_modalities"].([]string)
	if len(mods) != 1 || mods[0] != "text" {
		t.Fatalf("%v", mods)
	}
}

func TestToolResult(t *testing.T) {
	got := toolResult("call_1", "72F")
	item := got["item"].(map[string]any)
	if got["type"] != "conversation.item.create" || item["type"] != "function_call_output" || item["call_id"] != "call_1" || item["output"] != "72F" {
		t.Fatalf("%v", got)
	}
}

func TestTurnMessages(t *testing.T) {
	samples := []int16{1, 2, 3, 4}
	asr := turnMessages(samples, 16000, 1000, false, nil, false, false)
	if len(asr) != 2 {
		t.Fatalf("transcription events %d", len(asr))
	}
	if asr[1]["type"] != "input_audio_buffer.commit" || asr[1]["final"] != true {
		t.Fatalf("commit %+v", asr[1])
	}
	raw, _ := base64.StdEncoding.DecodeString(asr[0]["audio"].(string))
	if len(raw) != 8 {
		t.Fatalf("pcm bytes %d", len(raw))
	}

	conv := turnMessages(samples, 16000, 1000, true, []string{"audio"}, false, false)
	if len(conv) != 3 || conv[2]["type"] != "response.create" {
		t.Fatalf("conversation events %+v", conv)
	}
	if _, ok := conv[1]["final"]; ok {
		t.Fatal("conversation commit is not a transcription end")
	}
	resp := conv[2]["response"].(map[string]any)
	mods := resp["output_modalities"].([]string)
	if len(mods) != 1 || mods[0] != "audio" {
		t.Fatalf("modalities %+v", mods)
	}
	auto := turnMessages(samples, 16000, 1000, true, nil, false, false)
	if _, ok := auto[2]["response"]; ok {
		t.Fatal("unset modalities should leave the choice to the model")
	}
}

func TestUserTurnText(t *testing.T) {
	got, err := userTurn("hello", nil, 16000, 320, true, []string{"text"}, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0]["type"] != "conversation.item.create" || got[1]["type"] != "response.create" {
		t.Fatalf("%+v", got)
	}
	item := got[0]["item"].(map[string]any)
	content := item["content"].([]map[string]string)
	if content[0]["text"] != "hello" {
		t.Fatalf("%+v", content)
	}
	if _, err := userTurn("hello", []int16{1}, 16000, 320, false, nil, false, false); err == nil {
		t.Fatal("transcription should reject a text turn")
	}
	both, err := userTurn("hello", []int16{1, 2}, 16000, 1000, true, nil, false, false)
	if err != nil || both[0]["type"] != "conversation.item.create" || both[len(both)-1]["type"] != "response.create" {
		t.Fatalf("%v %+v", err, both)
	}
}

func TestServerVADTurn(t *testing.T) {
	got := turnMessages([]int16{1000, -1000}, 16000, 1000, true, []string{"audio"}, true, false)
	if len(got) != 2 || got[0]["type"] != "input_audio_buffer.append" || got[1]["type"] != "input_audio_buffer.append" {
		t.Fatalf("%v", got)
	}
	raw, err := base64.StdEncoding.DecodeString(got[1]["audio"].(string))
	if err != nil || len(raw) != 16000 {
		t.Fatalf("silence %d %v", len(raw), err)
	}
	for _, b := range raw {
		if b != 0 {
			t.Fatal("silence chunk has sound")
		}
	}
	mode, err := cleanTurn("server_vad")
	if err != nil || mode != "server_vad" {
		t.Fatal(err, mode)
	}
	if _, err := cleanTurn("neural"); err == nil {
		t.Fatal("expected turn to fail")
	}
	update := applyModalities(nil, []string{"audio"})
	session := update["session"].(map[string]any)
	mods := session["output_modalities"].([]string)
	if len(mods) != 1 || mods[0] != "audio" {
		t.Fatalf("%v", session)
	}
	u, err := realtimeURL("http://127.0.0.1:11434", "lfm3-audio", "", "", "server_vad", realtimeExtra{})
	if err != nil || !strings.Contains(u, "turn=server_vad") || strings.Contains(u, "intent=") || strings.Contains(u, "language=") {
		t.Fatal(u, err)
	}
	asrURL, err := realtimeURL("http://127.0.0.1:11434", "voxtral-realtime", "transcription", "en", "", realtimeExtra{})
	if err != nil || !strings.Contains(asrURL, "language=en") || !strings.Contains(asrURL, "intent=transcription") {
		t.Fatal(asrURL, err)
	}
	duplexURL, err := realtimeURL("http://127.0.0.1:11434", "llama3.2", "transcription", "en", "server_vad", realtimeExtra{
		Pipeline: "duplex", Voice: "piper", ASR: "whisper", Speaker: "female", Rate: 16000,
	})
	if err != nil || strings.Contains(duplexURL, "intent=") || !strings.Contains(duplexURL, "pipeline=duplex") || !strings.Contains(duplexURL, "voice=piper") || !strings.Contains(duplexURL, "asr=whisper") || !strings.Contains(duplexURL, "speaker=female") || !strings.Contains(duplexURL, "rate=16000") {
		t.Fatal(duplexURL, err)
	}
	clip := turnMessages([]int16{1, 2}, 16000, 1000, true, nil, false, true)
	if len(clip) != 2 || clip[1]["type"] != "input_audio_buffer.commit" {
		t.Fatalf("%v", clip)
	}
	if _, ok := clip[1]["final"]; ok {
		t.Fatal("duplex commit is not a transcription end")
	}
	text, err := userTurn("hello", nil, 16000, 320, true, nil, false, true)
	if err != nil || text[len(text)-1]["type"] != "response.create" {
		t.Fatalf("%v %v", err, text)
	}
	n, err := parseProbeRate("", 8000)
	if err != nil || n != 8000 {
		t.Fatal(err, n)
	}
	if _, err := parseProbeRate("nope", 8000); err == nil {
		t.Fatal("expected rate to fail")
	}
}

func TestParseModalities(t *testing.T) {
	got, err := parseModalities(" text, audio ")
	if err != nil || len(got) != 2 || got[0] != "text" || got[1] != "audio" {
		t.Fatalf("%v %v", got, err)
	}
	if _, err := parseModalities("speech"); err == nil {
		t.Fatal("expected an error")
	}
}

func TestResponseStatus(t *testing.T) {
	var ev map[string]any
	if err := json.Unmarshal([]byte(`{"type":"response.done","response":{"status":"cancelled"}}`), &ev); err != nil {
		t.Fatal(err)
	}
	if responseStatus(ev) != "cancelled" {
		t.Fatal(responseStatus(ev))
	}
}
