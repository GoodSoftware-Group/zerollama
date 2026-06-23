package modality

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestChatRequestHasMultimodalPayload(t *testing.T) {
	t.Parallel()
	if ChatRequestHasMultimodalPayload(nil) {
		t.Fatal("nil request")
	}
	if ChatRequestHasMultimodalPayload(&api.ChatRequest{}) {
		t.Fatal("empty request")
	}
	req := &api.ChatRequest{
		Messages: []api.Message{{AudioClips: []api.AudioData{{1}}}},
	}
	if !ChatRequestHasMultimodalPayload(req) {
		t.Fatal("expected audio")
	}
}

func TestChatRequestHasVideoPayload(t *testing.T) {
	t.Parallel()
	if ChatRequestHasVideoPayload(nil) {
		t.Fatal("nil")
	}
	if ChatRequestHasVideoPayload(&api.ChatRequest{Messages: []api.Message{{Content: "hi"}}}) {
		t.Fatal("text only")
	}
	if !ChatRequestHasVideoPayload(&api.ChatRequest{Messages: []api.Message{{Videos: []api.VideoData{{1}}}}}) {
		t.Fatal("raw video")
	}
	if !ChatRequestHasVideoPayload(&api.ChatRequest{Messages: []api.Message{{
		Images:     []api.ImageData{{1}, {2}},
		VideoSpans: []api.VideoSpan{{FrameCount: 2}},
	}}}) {
		t.Fatal("pre-expanded spans")
	}
}
