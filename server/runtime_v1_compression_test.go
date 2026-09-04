package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestAttachCompressionMetaToV1JSON(t *testing.T) {
	meta := &api.ChatCompressionMeta{Mode: "placeholder", ElideFrom: 4, PrefixReuseTokens: 12}
	out := attachCompressionMetaToV1JSON([]byte(`{"object":"chat.completion","usage":{"prompt_tokens":3}}`), meta)
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	usage := got["usage"].(map[string]any)
	cm := usage["compression_meta"].(map[string]any)
	if cm["mode"] != "placeholder" {
		t.Fatalf("%v", cm)
	}
	if int(cm["elide_from"].(float64)) != 4 {
		t.Fatalf("elide_from %v", cm["elide_from"])
	}
}

func TestCopyRuntimeV1SSEInjectsUsageChunk(t *testing.T) {
	meta := &api.ChatCompressionMeta{Mode: "placeholder", ElideFrom: 2}
	body := "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\ndata: [DONE]\n\n"
	rec := httptest.NewRecorder()
	resp := &http.Response{
		Header: http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:   io.NopCloser(strings.NewReader(body)),
	}
	if err := copyRuntimeV1ChatBody(rec, resp, meta, true); err != nil {
		t.Fatal(err)
	}
	s := rec.Body.String()
	if !strings.Contains(s, `"compression_meta"`) {
		t.Fatalf("missing compression_meta: %s", s)
	}
	if !strings.Contains(s, "data: [DONE]") {
		t.Fatalf("missing DONE: %s", s)
	}
	if idxMeta, idxDone := strings.Index(s, "compression_meta"), strings.Index(s, "[DONE]"); idxMeta < 0 || idxDone < 0 || idxMeta > idxDone {
		t.Fatalf("usage chunk should precede DONE: %s", s)
	}
}
