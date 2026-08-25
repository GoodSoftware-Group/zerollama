package server

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"sort"
	"strings"
	"sync"
)

const (
	defaultKNNK          = 3
	defaultKNNSimilarity = 0.80
	defaultKNNVote       = 0.5
	maxKNNK              = 32
)

// RouterKNN is the LA11b labelled-corpus classifier (no score LM).
type RouterKNN struct {
	K                   int                 `yaml:"k" json:"k"`
	SimilarityThreshold float64             `yaml:"similarity_threshold" json:"similarity_threshold"`
	VoteThreshold       float64             `yaml:"vote_threshold" json:"vote_threshold"`
	Corpus              []RouterCorpusEntry `yaml:"corpus" json:"corpus"`
}

type RouterCorpusEntry struct {
	ID     string   `yaml:"id,omitempty" json:"id,omitempty"`
	Text   string   `yaml:"text" json:"text"`
	Labels []string `yaml:"labels" json:"labels"`
}

type RouterNeighbor struct {
	ID         string   `json:"id,omitempty"`
	Similarity float64  `json:"similarity"`
	Labels     []string `json:"labels,omitempty"`
}

type knnIndexed struct {
	ID     string
	Labels []string
	Vec    []float32
}

var (
	knnExtraMu sync.Mutex
	knnExtra   = map[string][]RouterCorpusEntry{}

	knnVecMu    sync.Mutex
	knnVecCache = map[string][]float32{}
)

func routerIsKNN(spec RouterSpec) bool {
	return strings.EqualFold(strings.TrimSpace(spec.Classifier), "knn")
}

func knnParams(spec RouterSpec) (k int, sim, vote float64) {
	k, sim, vote = defaultKNNK, defaultKNNSimilarity, defaultKNNVote
	if spec.KNN == nil {
		return k, sim, vote
	}
	if spec.KNN.K > 0 {
		k = spec.KNN.K
		if k > maxKNNK {
			k = maxKNNK
		}
	}
	if spec.KNN.SimilarityThreshold > 0 {
		sim = spec.KNN.SimilarityThreshold
	}
	if spec.KNN.VoteThreshold > 0 {
		vote = spec.KNN.VoteThreshold
	}
	return k, sim, vote
}

func knnCorpus(name string, spec RouterSpec) []RouterCorpusEntry {
	var out []RouterCorpusEntry
	if spec.KNN != nil {
		out = append(out, spec.KNN.Corpus...)
	}
	knnExtraMu.Lock()
	out = append(out, knnExtra[name]...)
	knnExtraMu.Unlock()
	return out
}

func addKNNCorpus(name string, entries []RouterCorpusEntry) int {
	knnExtraMu.Lock()
	defer knnExtraMu.Unlock()
	n := 0
	for _, e := range entries {
		text := strings.TrimSpace(e.Text)
		if text == "" || len(e.Labels) == 0 {
			continue
		}
		if e.ID == "" {
			sum := sha256.Sum256([]byte(text))
			e.ID = hex.EncodeToString(sum[:8])
		}
		e.Text = text
		knnExtra[name] = append(knnExtra[name], e)
		n++
	}
	return n
}

func knnCorpusStats(name string, spec RouterSpec) (total int, byLabel map[string]int) {
	byLabel = map[string]int{}
	for _, e := range knnCorpus(name, spec) {
		total++
		seen := map[string]struct{}{}
		for _, l := range e.Labels {
			if _, ok := seen[l]; ok {
				continue
			}
			seen[l] = struct{}{}
			byLabel[l]++
		}
	}
	return total, byLabel
}

func cosineSim(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		av, bv := float64(a[i]), float64(b[i])
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func knnSearch(query []float32, indexed []knnIndexed, k int) []RouterNeighbor {
	if k <= 0 || len(indexed) == 0 || len(query) == 0 {
		return nil
	}
	hits := make([]RouterNeighbor, 0, len(indexed))
	for _, e := range indexed {
		hits = append(hits, RouterNeighbor{
			ID:         e.ID,
			Labels:     e.Labels,
			Similarity: cosineSim(query, e.Vec),
		})
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Similarity != hits[j].Similarity {
			return hits[i].Similarity > hits[j].Similarity
		}
		return hits[i].ID < hits[j].ID
	})
	if k > len(hits) {
		k = len(hits)
	}
	return hits[:k]
}

func knnVote(neighbors []RouterNeighbor, simThresh, voteThresh float64) (active []string, shares map[string]float64, best float64) {
	shares = map[string]float64{}
	var total float64
	for _, n := range neighbors {
		if n.Similarity > best {
			best = n.Similarity
		}
		if n.Similarity < simThresh || len(n.Labels) == 0 {
			continue
		}
		total += n.Similarity
		for _, l := range n.Labels {
			shares[l] += n.Similarity
		}
	}
	if total == 0 {
		return nil, shares, best
	}
	labels := make([]string, 0, len(shares))
	for l := range shares {
		shares[l] /= total
		labels = append(labels, l)
	}
	sort.Slice(labels, func(i, j int) bool {
		if shares[labels[i]] != shares[labels[j]] {
			return shares[labels[i]] > shares[labels[j]]
		}
		return labels[i] < labels[j]
	})
	for _, l := range labels {
		if shares[l] >= voteThresh {
			active = append(active, l)
		}
	}
	return active, shares, best
}

func knnCacheKey(embedder, text string) string {
	sum := sha256.Sum256([]byte(embedder + "\n" + text))
	return hex.EncodeToString(sum[:])
}
