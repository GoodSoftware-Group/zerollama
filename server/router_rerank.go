package server

import (
	"context"
	"fmt"
	"strings"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

func routerIsRerank(spec RouterSpec) bool {
	c := strings.ToLower(strings.TrimSpace(spec.Classifier))
	return c == "rerank" || c == "colbert"
}

func routerRerankModel(spec RouterSpec) string {
	if strings.TrimSpace(spec.Reranker) != "" {
		return spec.Reranker
	}
	return strings.TrimSpace(spec.Embedder)
}

func policyDocument(p RouterPolicy) string {
	if strings.TrimSpace(p.Description) != "" {
		return p.Description
	}
	return p.Label
}

func decideRouterRerank(ctx context.Context, name string, spec RouterSpec, user string, rerank routerRerankFn, dec *RouterDecision, fallback func(string) (RouterDecision, error), finish func([]string) (RouterDecision, error)) (RouterDecision, error) {
	model := routerRerankModel(spec)
	if model == "" || rerank == nil {
		return fallback("reranker unavailable")
	}
	docs := make([]string, 0, len(spec.Policies))
	labels := make([]string, 0, len(spec.Policies))
	for _, p := range spec.Policies {
		if p.Label == "" {
			continue
		}
		labels = append(labels, p.Label)
		docs = append(docs, policyDocument(p))
	}
	if len(docs) == 0 {
		return fallback("no policies")
	}

	hits, err := rerank(ctx, model, user, docs)
	if err != nil {
		return fallback("rerank failed: " + err.Error())
	}

	thresh := spec.ActivationThreshold
	if thresh <= 0 {
		thresh = defaultRouterActivation
	}

	scoreByLabel := make(map[string]float64, len(labels))
	for _, h := range hits {
		if h.Index < 0 || h.Index >= len(labels) {
			continue
		}
		scoreByLabel[labels[h.Index]] = h.RelevanceScore
	}

	var active []string
	for _, lab := range labels {
		sc := scoreByLabel[lab]
		on := sc >= thresh
		dec.Scores = append(dec.Scores, RouterLabelScore{
			Label:          lab,
			RelevanceScore: sc,
			Softmax:        sc,
			Active:         on,
		})
		if on {
			active = append(active, lab)
		}
	}
	if len(active) == 0 {
		return fallback("no labels above threshold")
	}
	return finish(active)
}

func (s *Server) rerankRouterDocs(ctx context.Context, modelName, query string, docs []string) ([]llm.RerankHit, error) {
	r, _, _, _, releaseQoS, err := s.scheduleRunner(ctx, modelName, []model.Capability{}, nil, nil, nil, nil, nil)
	if err != nil {
		return nil, err
	}
	defer releaseQoS()
	rr, ok := r.(llm.Reranker)
	if !ok {
		return nil, fmt.Errorf("reranker %q does not support rerank", modelName)
	}
	resp, err := rr.Rerank(ctx, llm.RerankRequest{Query: query, Documents: docs})
	if err != nil {
		return nil, err
	}
	return resp.Results, nil
}
