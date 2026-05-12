package openai

import (
	"encoding/json"
	"testing"
)

func TestTranscriptionResult(t *testing.T) {
	t.Parallel()
	ct, body, err := TranscriptionResult("hello", "json", "")
	if err != nil || ct != "application/json" {
		t.Fatalf("json: ct=%q err=%v", ct, err)
	}
	var tr TranscriptionResponse
	if err := json.Unmarshal(body, &tr); err != nil || tr.Text != "hello" {
		t.Fatalf("json body: %+v err=%v", tr, err)
	}

	ct, body, err = TranscriptionResult("hi", "verbose_json", "en")
	if err != nil || ct != "application/json" {
		t.Fatalf("verbose: ct=%q err=%v", ct, err)
	}
	var vr TranscriptionVerboseResponse
	if err := json.Unmarshal(body, &vr); err != nil || vr.Text != "hi" || vr.Task != "transcribe" {
		t.Fatalf("verbose body: %+v err=%v", vr, err)
	}

	ct, body, err = TranscriptionResult("x", "text", "")
	if err != nil || ct != "text/plain" || string(body) != "x" {
		t.Fatalf("text: ct=%q body=%q err=%v", ct, body, err)
	}
}
