package server

import (
	"context"
	"fmt"
	"math"
	"os"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
)

const defaultRouterActivation = 0.15

// RouterFile is ~/.ollama/router.yaml (LA11). Missing file means routing is off.
type RouterFile struct {
	Routers map[string]RouterSpec `yaml:"routers" json:"routers"`
}

type RouterSpec struct {
	Classifier          string            `yaml:"classifier" json:"classifier"`
	Embedder            string            `yaml:"embedder" json:"embedder"`
	Reranker            string            `yaml:"reranker" json:"reranker"`
	Fallback            string            `yaml:"fallback" json:"fallback"`
	ActivationThreshold float64           `yaml:"activation_threshold" json:"activation_threshold"`
	LengthNormalize     bool              `yaml:"length_normalize" json:"length_normalize"`
	Policies            []RouterPolicy    `yaml:"policies" json:"policies"`
	Candidates          []RouterCandidate `yaml:"candidates" json:"candidates"`
	KNN                 *RouterKNN        `yaml:"knn" json:"knn"`
}

type RouterPolicy struct {
	Label       string `yaml:"label" json:"label"`
	Description string `yaml:"description" json:"description"`
}

type RouterCandidate struct {
	Model  string   `yaml:"model" json:"model"`
	Labels []string `yaml:"labels" json:"labels"`
}

type RouterDecision struct {
	Router            string             `json:"router"`
	Classifier        string             `json:"classifier"`
	Labels            []string           `json:"labels"`
	Candidate         string             `json:"candidate"`
	Fallback          bool               `json:"fallback"`
	Scores            []RouterLabelScore `json:"scores,omitempty"`
	Neighbors         []RouterNeighbor   `json:"neighbors,omitempty"`
	NearestSimilarity float64            `json:"nearest_similarity,omitempty"`
	LatencyMs         int64              `json:"latency_ms"`
}

type RouterLabelScore struct {
	Label          string  `json:"label"`
	LogProb        float64 `json:"log_prob"`
	RelevanceScore float64 `json:"relevance_score,omitempty"`
	Softmax        float64 `json:"softmax"`
	NumTokens      int     `json:"num_tokens"`
	Active         bool    `json:"active"`
}

type routerScoreFn func(ctx context.Context, classifier, prompt string, candidates []string, lengthNorm bool) ([]llm.CandidateScore, error)

type routerEmbedFn func(ctx context.Context, model, text string) ([]float32, error)

type routerRerankFn func(ctx context.Context, model, query string, docs []string) ([]llm.RerankHit, error)

var (
	routerFileMu    sync.Mutex
	routerFilePath  string
	routerFileMod   time.Time
	routerFileCache *RouterFile
)

func loadRouterFile() (*RouterFile, error) {
	path := envconfig.RouterConfigPath()
	if path == "" {
		return nil, nil
	}
	st, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	routerFileMu.Lock()
	defer routerFileMu.Unlock()
	if routerFileCache != nil && routerFilePath == path && st.ModTime().Equal(routerFileMod) {
		return routerFileCache, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var file RouterFile
	if err := yaml.Unmarshal(raw, &file); err != nil {
		return nil, fmt.Errorf("router config: %w", err)
	}
	if file.Routers == nil {
		file.Routers = map[string]RouterSpec{}
	}
	routerFileCache = &file
	routerFilePath = path
	routerFileMod = st.ModTime()
	return &file, nil
}

func lookupRouter(name string) (RouterSpec, bool) {
	file, err := loadRouterFile()
	if err != nil || file == nil {
		return RouterSpec{}, false
	}
	spec, ok := file.Routers[name]
	if !ok || len(spec.Candidates) == 0 {
		return RouterSpec{}, false
	}
	return spec, true
}

func matchRouterCandidate(candidates []RouterCandidate, active []string) string {
	if len(active) == 0 {
		return ""
	}
	for _, c := range candidates {
		if labelSetCovers(c.Labels, active) {
			return c.Model
		}
	}
	return ""
}

func labelSetCovers(have, needed []string) bool {
	for _, n := range needed {
		if !slices.Contains(have, n) {
			return false
		}
	}
	return true
}

func softmax(logprobs []float64) []float64 {
	out := make([]float64, len(logprobs))
	if len(logprobs) == 0 {
		return out
	}
	maxv := logprobs[0]
	for _, v := range logprobs[1:] {
		if v > maxv {
			maxv = v
		}
	}
	var sum float64
	for i, v := range logprobs {
		out[i] = math.Exp(v - maxv)
		sum += out[i]
	}
	if sum == 0 || math.IsInf(sum, 0) || math.IsNaN(sum) {
		eq := 1 / float64(len(logprobs))
		for i := range out {
			out[i] = eq
		}
		return out
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func buildRouterPrompt(spec RouterSpec, user string) string {
	var b strings.Builder
	b.WriteString("Classify the user request. Pick the label that best matches intent.\n\nLabels:\n")
	for _, p := range spec.Policies {
		b.WriteString("- ")
		b.WriteString(p.Label)
		if p.Description != "" {
			b.WriteString(": ")
			b.WriteString(p.Description)
		}
		b.WriteByte('\n')
	}
	b.WriteString("\nUser request:\n")
	b.WriteString(user)
	b.WriteString("\n\nThe best label is:\n")
	return b.String()
}

func decideRouter(ctx context.Context, name string, spec RouterSpec, user string, score routerScoreFn, embed routerEmbedFn, rerank routerRerankFn) (RouterDecision, error) {
	start := time.Now()
	dec := RouterDecision{
		Router:     name,
		Classifier: spec.Classifier,
	}

	fallback := func(reason string) (RouterDecision, error) {
		if spec.Fallback == "" {
			return dec, fmt.Errorf("router %q: %s and no fallback", name, reason)
		}
		dec.Candidate = spec.Fallback
		dec.Fallback = true
		dec.LatencyMs = time.Since(start).Milliseconds()
		return dec, nil
	}

	finish := func(active []string) (RouterDecision, error) {
		dec.Labels = active
		chosen := matchRouterCandidate(spec.Candidates, active)
		if chosen == "" {
			return fallback("no candidate covers labels")
		}
		if _, isRouter := lookupRouter(chosen); isRouter {
			return dec, fmt.Errorf("router candidate %q is itself a router", chosen)
		}
		dec.Candidate = chosen
		dec.LatencyMs = time.Since(start).Milliseconds()
		return dec, nil
	}

	if strings.TrimSpace(user) == "" {
		return fallback("empty prompt")
	}
	if routerIsKNN(spec) {
		return decideRouterKNN(ctx, name, spec, user, embed, &dec, fallback, finish)
	}
	if routerIsRerank(spec) {
		return decideRouterRerank(ctx, name, spec, user, rerank, &dec, fallback, finish)
	}

	thresh := spec.ActivationThreshold
	if thresh <= 0 {
		thresh = defaultRouterActivation
	}
	if spec.Classifier == "" || score == nil {
		return fallback("classifier unavailable")
	}
	labels := make([]string, 0, len(spec.Policies))
	for _, p := range spec.Policies {
		if p.Label != "" {
			labels = append(labels, p.Label)
		}
	}
	if len(labels) == 0 {
		return fallback("no policies")
	}

	scored, err := score(ctx, spec.Classifier, buildRouterPrompt(spec, user), labels, spec.LengthNormalize)
	if err != nil {
		return fallback("score failed: " + err.Error())
	}

	logps := make([]float64, len(labels))
	toks := make([]int, len(labels))
	for i, lab := range labels {
		logps[i] = math.Inf(-1)
		for _, s := range scored {
			if s.Candidate == lab {
				lp := s.LogProb
				if spec.LengthNormalize && s.LengthNormalizedLogProb != 0 {
					lp = s.LengthNormalizedLogProb
				}
				logps[i] = lp
				toks[i] = s.NumTokens
				break
			}
		}
	}
	sm := softmax(logps)
	var active []string
	dec.Scores = make([]RouterLabelScore, len(labels))
	for i, lab := range labels {
		on := sm[i] >= thresh
		dec.Scores[i] = RouterLabelScore{
			Label:     lab,
			LogProb:   logps[i],
			Softmax:   sm[i],
			NumTokens: toks[i],
			Active:    on,
		}
		if on {
			active = append(active, lab)
		}
	}
	return finish(active)
}

func decideRouterKNN(ctx context.Context, name string, spec RouterSpec, user string, embed routerEmbedFn, dec *RouterDecision, fallback func(string) (RouterDecision, error), finish func([]string) (RouterDecision, error)) (RouterDecision, error) {
	embedder := strings.TrimSpace(spec.Embedder)
	if embedder == "" || embed == nil {
		return fallback("knn embedder unavailable")
	}
	entries := knnCorpus(name, spec)
	if len(entries) == 0 {
		return fallback("knn corpus empty")
	}
	k, sim, vote := knnParams(spec)
	query, err := cachedEmbed(ctx, embed, embedder, user)
	if err != nil {
		return fallback("knn embed failed: " + err.Error())
	}
	indexed := make([]knnIndexed, 0, len(entries))
	for _, e := range entries {
		vec, err := cachedEmbed(ctx, embed, embedder, e.Text)
		if err != nil {
			continue
		}
		id := e.ID
		if id == "" {
			id = knnCacheKey(embedder, e.Text)[:16]
		}
		indexed = append(indexed, knnIndexed{ID: id, Labels: e.Labels, Vec: vec})
	}
	if len(indexed) == 0 {
		return fallback("knn corpus embed failed")
	}
	neighbors := knnSearch(query, indexed, k)
	dec.Neighbors = neighbors
	active, shares, best := knnVote(neighbors, sim, vote)
	dec.NearestSimilarity = best
	labels := make([]string, 0, len(shares))
	for l := range shares {
		labels = append(labels, l)
	}
	sort.Slice(labels, func(i, j int) bool {
		if shares[labels[i]] != shares[labels[j]] {
			return shares[labels[i]] > shares[labels[j]]
		}
		return labels[i] < labels[j]
	})
	activeSet := map[string]struct{}{}
	for _, l := range active {
		activeSet[l] = struct{}{}
	}
	dec.Scores = make([]RouterLabelScore, 0, len(labels))
	for _, l := range labels {
		_, on := activeSet[l]
		dec.Scores = append(dec.Scores, RouterLabelScore{
			Label:   l,
			Softmax: shares[l],
			Active:  on,
		})
	}
	return finish(active)
}

func cachedEmbed(ctx context.Context, embed routerEmbedFn, model, text string) ([]float32, error) {
	key := knnCacheKey(model, text)
	knnVecMu.Lock()
	if v, ok := knnVecCache[key]; ok {
		knnVecMu.Unlock()
		return v, nil
	}
	knnVecMu.Unlock()
	v, err := embed(ctx, model, text)
	if err != nil {
		return nil, err
	}
	knnVecMu.Lock()
	knnVecCache[key] = v
	knnVecMu.Unlock()
	return v, nil
}
