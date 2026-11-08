package openai

import (
	"encoding/json"
	"testing"
)

func TestFromTranscriptionRequestTemperature(t *testing.T) {
	t.Parallel()
	cap := 32
	req, err := FromTranscriptionRequest(TranscriptionRequest{Model: "voxtral", Temperature: "0.2", MaxTokens: &cap}, false)
	if err != nil {
		t.Fatal(err)
	}
	if req.Options["temperature"] != 0.2 || req.Options["num_predict"] != 32 {
		t.Fatalf("options %v", req.Options)
	}
	req, err = FromTranscriptionRequest(TranscriptionRequest{Model: "voxtral"}, false)
	if err != nil || req.Options["temperature"] != 0.0 {
		t.Fatalf("omitted %+v err=%v", req, err)
	}
	if _, err := FromTranscriptionRequest(TranscriptionRequest{Model: "voxtral", Temperature: "2"}, false); err == nil {
		t.Fatal("expected reject")
	}
}

func TestTranscriptionResult(t *testing.T) {
	t.Parallel()
	ct, body, err := TranscriptionResult("hello", "json", "", "", nil, nil, nil)
	if err != nil || ct != "application/json" {
		t.Fatalf("json: ct=%q err=%v", ct, err)
	}
	var tr TranscriptionResponse
	if err := json.Unmarshal(body, &tr); err != nil || tr.Text != "hello" {
		t.Fatalf("json body: %+v err=%v", tr, err)
	}

	ct, body, err = TranscriptionResult("hi", "verbose_json", "en", "", &TranscriptionUsage{Type: "tokens", InputTokens: 2, OutputTokens: 1, TotalTokens: 3}, nil, nil)
	if err != nil || ct != "application/json" {
		t.Fatalf("verbose: ct=%q err=%v", ct, err)
	}
	var vr TranscriptionVerboseResponse
	if err := json.Unmarshal(body, &vr); err != nil || vr.Text != "hi" || vr.Task != "transcribe" || vr.Usage == nil || vr.Usage.TotalTokens != 3 {
		t.Fatalf("verbose body: %+v err=%v", vr, err)
	}

	ct, body, err = TranscriptionResultDiarized("hello there", "en", "", []TranscriptionDiarizedSegment{{
		ID: "seg_0", Type: "transcript.text.segment", Speaker: "0", Start: 0, End: 1.5, Text: "hello there",
	}}, 1.5, nil, nil)
	if err != nil || ct != "application/json" {
		t.Fatalf("diarized: ct=%q err=%v", ct, err)
	}
	var dr TranscriptionDiarizedResponse
	if err := json.Unmarshal(body, &dr); err != nil || dr.Text != "hello there" || dr.Task != "transcribe" || len(dr.Segments) != 1 || dr.Segments[0].Speaker != "0" || dr.Duration != 1.5 {
		t.Fatalf("diarized body: %+v err=%v", dr, err)
	}

	ct, body, err = TranscriptionResult("x", "text", "", "translate", &TranscriptionUsage{TotalTokens: 9}, nil, nil)
	if err != nil || ct != "text/plain" || string(body) != "x" {
		t.Fatalf("text: ct=%q body=%q err=%v", ct, body, err)
	}
}
