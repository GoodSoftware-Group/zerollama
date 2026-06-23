package llm

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLlamaServerTokenLogprob(t *testing.T) {
	require.Equal(t, -1.5, llamaServerTokenLogprob(llamaServerTokenProb{Logprob: -1.5}))
	require.InDelta(t, math.Log(0.5), llamaServerTokenLogprob(llamaServerTokenProb{Prob: 0.5}), 1e-9)
	require.Equal(t, 0.0, llamaServerTokenLogprob(llamaServerTokenProb{}))
}

func TestLlamaServerLogprobForToken(t *testing.T) {
	probs := []llamaServerTokenProb{{
		ID:      42,
		Token:   "yes",
		Logprob: -0.1,
		TopLogprobs: []llamaServerTokenProb{
			{ID: 7, Token: "no", Logprob: -2.0},
		},
		TopProbs: []llamaServerTokenProb{
			{ID: 9, Token: "maybe", Prob: 0.25},
		},
	}}

	lp, tok, ok := llamaServerLogprobForToken(probs, 42)
	require.True(t, ok)
	require.Equal(t, "yes", tok)
	require.Equal(t, -0.1, lp)

	lp, tok, ok = llamaServerLogprobForToken(probs, 7)
	require.True(t, ok)
	require.Equal(t, "no", tok)
	require.Equal(t, -2.0, lp)

	lp, tok, ok = llamaServerLogprobForToken(probs, 9)
	require.True(t, ok)
	require.Equal(t, "maybe", tok)
	require.InDelta(t, math.Log(0.25), lp, 1e-9)

	_, _, ok = llamaServerLogprobForToken(probs, 999)
	require.False(t, ok)
}
