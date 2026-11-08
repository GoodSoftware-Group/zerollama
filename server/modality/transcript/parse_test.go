package transcript

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestParseLlama(t *testing.T) {
	text, usage, err := ParseLlama([]byte(`{"type":"transcript.text.done","text":" hello ","usage":{"type":"tokens","input_tokens":12,"output_tokens":3,"total_tokens":15,"input_tokens_details":{"cached_tokens":1}}}`))
	if err != nil || text != "hello" || usage == nil || usage.Type != "tokens" || usage.InputTokens != 12 || usage.OutputTokens != 3 || usage.TotalTokens != 15 || usage.CachedTokens == nil || *usage.CachedTokens != 1 {
		t.Fatalf("text=%q usage=%+v err=%v", text, usage, err)
	}
	text, usage, err = ParseLlama([]byte(`{"text":"cached","usage":{"type":"tokens","input_tokens":2,"output_tokens":1,"total_tokens":3,"input_tokens_details":{"cached_tokens":0}}}`))
	if err != nil || usage == nil || usage.CachedTokens == nil || *usage.CachedTokens != 0 {
		t.Fatalf("text=%q usage=%+v err=%v", text, usage, err)
	}
	text, usage, err = ParseLlama([]byte(`{"text":"hi"}`))
	if err != nil || text != "hi" || usage != nil {
		t.Fatalf("text=%q usage=%+v err=%v", text, usage, err)
	}
	if _, _, err := ParseLlama([]byte(`{"type":"transcript.text.done"}`)); err != ErrEmpty {
		t.Fatal(err)
	}
	if _, _, err := ParseLlama([]byte(`not-json`)); err == nil {
		t.Fatal("expected json error")
	}
	on, err := StreamFlag(" TRUE ")
	if err != nil || !on {
		t.Fatal(err, on)
	}
	on, err = StreamFlag("")
	if err != nil || on {
		t.Fatal(err, on)
	}
	if _, err := StreamFlag("yes"); err == nil {
		t.Fatal("expected reject")
	}
	n, err := MaxTokens(" 32 ")
	if err != nil || n == nil || *n != 32 {
		t.Fatal(err, n)
	}
	n, err = MaxTokens("")
	if err != nil || n != nil {
		t.Fatal(err, n)
	}
	if _, err := MaxTokens("0"); err == nil {
		t.Fatal("expected reject")
	}
}

func TestNameEvents(t *testing.T) {
	in := "" +
		"data: {\"type\":\"transcript.text.delta\",\"delta\":\"hel\"}\n\n" +
		"event: transcript.text.done\n" +
		"data: {\"type\":\"transcript.text.done\",\"text\":\"hello\"}\n\n" +
		"data: [DONE]\n\n"
	var buf bytes.Buffer
	if err := NameEvents(&buf, strings.NewReader(in), 0); err != nil {
		t.Fatal(err)
	}
	got := buf.String()
	want := "" +
		"event: transcript.text.delta\n" +
		"data: {\"type\":\"transcript.text.delta\",\"delta\":\"hel\"}\n\n" +
		"event: transcript.text.done\n" +
		"data: {\"type\":\"transcript.text.done\",\"text\":\"hello\"}\n\n" +
		"data: [DONE]\n\n"
	if got != want {
		t.Fatalf("got %q", got)
	}
	in = "data: {\"type\":\"transcript.text.done\",\"text\":\"hello\",\"usage\":{\"type\":\"tokens\",\"total_tokens\":4}}\n\n"
	buf.Reset()
	if err := NameEvents(&buf, strings.NewReader(in), 1.5); err != nil {
		t.Fatal(err)
	}
	var ev struct {
		Type  string `json:"type"`
		Usage struct {
			Type    string  `json:"type"`
			Seconds float64 `json:"seconds"`
			Total   int     `json:"total_tokens"`
		} `json:"usage"`
	}
	line := ""
	for _, part := range strings.Split(buf.String(), "\n") {
		if strings.HasPrefix(part, "data: ") {
			line = strings.TrimPrefix(part, "data: ")
		}
	}
	if err := json.Unmarshal([]byte(line), &ev); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "event: transcript.text.done") || ev.Usage.Type != "tokens" || ev.Usage.Seconds != 1.5 || ev.Usage.Total != 4 {
		t.Fatalf("%s", buf.String())
	}
}
