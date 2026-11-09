package server

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ollama/ollama/server/voicepipe"
)

func TestConsumeRuntimeChatNDJSONSpeaksClauses(t *testing.T) {
	body := strings.NewReader(strings.Join([]string{
		`{"message":{"role":"assistant","content":"Hello, "},"done":false}`,
		`{"message":{"role":"assistant","content":"there."},"done":true}`,
		``,
	}, "\n"))
	var got []string
	err := consumeRuntimeChatNDJSON(context.Background(), body, func(sentence string) error {
		got = append(got, sentence)
		return nil
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "|") != "Hello,|there." {
		t.Fatalf("sentences = %#v", got)
	}
}

func TestConsumeRuntimeChatNDJSONErrorChunk(t *testing.T) {
	body := strings.NewReader(`{"error":"model uses zerollama-runtime inference"}` + "\n")
	err := consumeRuntimeChatNDJSON(context.Background(), body, func(string) error { return nil }, nil)
	if err == nil || !strings.Contains(err.Error(), "zerollama-runtime") {
		t.Fatalf("err = %v", err)
	}
}

func TestPostRuntimeChatOmitsThink(t *testing.T) {
	var sawThink bool
	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body = string(raw)
		sawThink = strings.Contains(body, `"think"`)
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = io.WriteString(w, `{"message":{"role":"assistant","content":"Hi."},"done":true}`+"\n")
	}))
	defer srv.Close()

	var got []string
	err := postRuntimeChat(context.Background(), srv.URL, map[string]any{
		"model":    "qwen2.5:7b",
		"messages": []any{map[string]any{"role": "user", "content": "Hey Tony"}},
		"stream":   true,
		"options":  map[string]any{"num_predict": 48},
	}, func(sentence string) error {
		got = append(got, sentence)
		return nil
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if sawThink {
		t.Fatalf("think was sent: %s", body)
	}
	if strings.Join(got, "") != "Hi." {
		t.Fatalf("sentences = %#v", got)
	}
}

func TestRuntimeChatHTTPErrorUsesDetail(t *testing.T) {
	err := runtimeChatHTTPError(http.StatusNotImplemented, []byte(`{"detail":"request requires legacy runner (vision/logprobs/think)"}`))
	if err == nil || !strings.Contains(err.Error(), "legacy runner") {
		t.Fatalf("err = %v", err)
	}
}

func TestRuntimeChatFailureLoadingModel503(t *testing.T) {
	raw := `llama-server /completion HTTP 503: {"error":{"message":"Loading model","type":"unavailable_error","code":503}}`
	err := runtimeChatFailure(raw)
	var coded *voicepipe.Error
	if !errors.As(err, &coded) || coded == nil {
		t.Fatalf("err = %v", err)
	}
	if coded.Code != "model_load_failed" {
		t.Fatalf("code = %q", coded.Code)
	}
	if coded.Message != "the chat model did not finish loading" {
		t.Fatalf("message = %q", coded.Message)
	}
	if strings.Contains(coded.Message, "{") {
		t.Fatalf("message still has JSON: %q", coded.Message)
	}
}

func TestRuntimeChatFailureNotHealthy(t *testing.T) {
	err := runtimeChatFailure("llama-server not healthy at http://127.0.0.1:8082/health within 120s")
	var coded *voicepipe.Error
	if !errors.As(err, &coded) || coded == nil || coded.Code != "model_load_failed" {
		t.Fatalf("err = %v", err)
	}
}

func TestConsumeRuntimeChatNDJSONLoadingErrorIsCoded(t *testing.T) {
	body := strings.NewReader(
		`{"error":"llama-server /completion HTTP 503: {\"error\":{\"message\":\"Loading model\",\"type\":\"unavailable_error\",\"code\":503}}"}` + "\n",
	)
	err := consumeRuntimeChatNDJSON(context.Background(), body, func(string) error { return nil }, nil)
	var coded *voicepipe.Error
	if !errors.As(err, &coded) || coded == nil || coded.Code != "model_load_failed" {
		t.Fatalf("err = %v", err)
	}
	if strings.Contains(coded.Message, "{") {
		t.Fatalf("message still has JSON: %q", coded.Message)
	}
}

func TestConsumeRuntimeChatNDJSONReportsFirstToken(t *testing.T) {
	body := strings.NewReader(`{"message":{"role":"assistant","content":"Hi"},"done":true}` + "\n")
	var stages []string
	rep := &voicepipe.StageReporter{}
	rep.Set(func(status string, _ map[string]any) {
		stages = append(stages, status)
	})
	err := consumeRuntimeChatNDJSON(context.Background(), body, func(string) error { return nil }, rep)
	if err != nil {
		t.Fatal(err)
	}
	if len(stages) != 1 || stages[0] != voicepipe.StageFirstToken {
		t.Fatalf("stages = %#v", stages)
	}
}
