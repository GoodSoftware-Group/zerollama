package server

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestModelUsesSystemOneScore(t *testing.T) {
	laya := &Model{Config: model.ConfigV2{Capabilities: []string{"decision"}, ModelFamily: "laya"}}
	require.False(t, modelUsesSystemOneScore(laya))

	layaArch := &Model{Config: model.ConfigV2{Capabilities: []string{"decision"}, ModelFamilies: []string{"LayaForDecision"}}}
	require.False(t, modelUsesSystemOneScore(layaArch))

	clef := &Model{Config: model.ConfigV2{Capabilities: []string{"decision", "vision"}, Renderer: "clef"}}
	require.True(t, modelUsesSystemOneScore(clef))

	nimble := &Model{Config: model.ConfigV2{Capabilities: []string{"completion", "decision"}, Renderer: "qwen3.5"}}
	require.True(t, modelUsesSystemOneScore(nimble))

	tev1 := &Model{Config: model.ConfigV2{Capabilities: []string{"decision"}, Renderer: "tev1"}}
	require.True(t, modelUsesSystemOneScore(tev1))

	strands := &Model{Config: model.ConfigV2{Capabilities: []string{"decision"}, Renderer: "strands"}}
	require.True(t, modelUsesSystemOneScore(strands))

	chatOnly := &Model{Config: model.ConfigV2{Capabilities: []string{"completion"}, Renderer: "qwen3.5"}}
	require.False(t, modelUsesSystemOneScore(chatOnly))

	// Bare CAPABILITY decision without score renderer/family must not steal Decider.
	bare := &Model{Config: model.ConfigV2{Capabilities: []string{"decision"}, Renderer: "llama"}}
	require.False(t, modelUsesSystemOneScore(bare))
}

func TestAPIDecisionsToLLMCopiesImages(t *testing.T) {
	req := api.DecisionsRequest{
		Model:  "clef",
		State:  "screenshot",
		Images: []api.ImageData{[]byte("png")},
		Questions: map[string]api.DecisionQuestion{
			"has_ui": {Type: "noul", Instructions: "Is a UI visible?"},
		},
	}
	out, err := apiDecisionsToLLM(req)
	require.NoError(t, err)
	require.Len(t, out.Images, 1)
	require.Equal(t, []byte("png"), []byte(out.Images[0]))
}
