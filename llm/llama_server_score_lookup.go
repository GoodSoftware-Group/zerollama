package llm

import "math"

// llamaServerTokenLogprob returns the log probability from a llama-server token prob entry.
func llamaServerTokenLogprob(p llamaServerTokenProb) float64 {
	if p.Logprob != 0 || p.Prob == 0 {
		return p.Logprob
	}
	return math.Log(p.Prob)
}

// llamaServerLogprobForToken finds the logprob of target in completion_probabilities.
func llamaServerLogprobForToken(probs []llamaServerTokenProb, target int) (logprob float64, token string, ok bool) {
	for _, prob := range probs {
		if prob.ID == target {
			return llamaServerTokenLogprob(prob), prob.Token, true
		}
		for _, top := range prob.TopLogprobs {
			if top.ID == target {
				return llamaServerTokenLogprob(top), top.Token, true
			}
		}
		for _, top := range prob.TopProbs {
			if top.ID == target {
				return llamaServerTokenLogprob(top), top.Token, true
			}
		}
	}
	return 0, "", false
}
