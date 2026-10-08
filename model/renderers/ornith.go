package renderers

import "github.com/ollama/ollama/types/model"

type OrnithRenderer struct {
	Qwen35Renderer
}

func newOrnithRenderer() Renderer {
	return &OrnithRenderer{
		Qwen35Renderer: Qwen35Renderer{
			isThinking:                      true,
			alwaysRenderAssistantThinkBlock: true,
			emitEmptyThinkOnNoThink:         true,
			useImgTags:                      RenderImgTags,
		},
	}
}

func (r *OrnithRenderer) Thinking() *model.Thinking {
	return r.Qwen35Renderer.Thinking()
}
