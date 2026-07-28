package server

import (
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestSglangShouldProxyChat(t *testing.T) {
	t.Setenv("OLLAMA_SGLANG_URL", "http://127.0.0.1:30000")

	dca := &Model{
		Config: model.ConfigV2{
			ModalityBackends: map[string]string{
				model.ModalityInference: model.BackendSGLang,
			},
		},
	}
	if !sglangShouldProxyChat(dca, false) {
		t.Fatal("inference=sglang should proxy without video")
	}
	if !sglangShouldProxyChat(dca, true) {
		t.Fatal("inference=sglang should proxy with video too")
	}

	videoOnly := &Model{
		Config: model.ConfigV2{
			ModalityBackends: map[string]string{
				model.ModalityVideoUnderstanding: model.BackendSGLang,
			},
		},
	}
	if sglangShouldProxyChat(videoOnly, false) {
		t.Fatal("video_understanding=sglang must not proxy text-only chat")
	}
	if !sglangShouldProxyChat(videoOnly, true) {
		t.Fatal("video_understanding=sglang should proxy when video_url present")
	}

	plain := &Model{Config: model.ConfigV2{}}
	if sglangShouldProxyChat(plain, true) {
		t.Fatal("no sglang backend → no proxy")
	}

	t.Setenv("OLLAMA_SGLANG_URL", "")
	if sglangShouldProxyChat(dca, false) {
		t.Fatal("missing OLLAMA_SGLANG_URL → no proxy")
	}
}
