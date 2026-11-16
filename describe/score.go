package describe

import "math"

// Cosine returns the cosine similarity of two vectors. Zero when either is empty
// or dimensions differ.
func Cosine(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		fa := float64(a[i])
		fb := float64(b[i])
		dot += fa * fb
		na += fa * fa
		nb += fb * fb
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// KeepTags returns labels whose cosine with the image vector is at least minScore.
// No top-k fill: zero matches yields an empty slice.
func KeepTags(labels []string, labelVecs [][]float32, image []float32, minScore float64) []TagScore {
	if minScore <= 0 {
		minScore = DefaultMinScore
	}
	out := make([]TagScore, 0)
	n := len(labels)
	if len(labelVecs) < n {
		n = len(labelVecs)
	}
	for i := 0; i < n; i++ {
		score := Cosine(image, labelVecs[i])
		if score >= minScore {
			out = append(out, TagScore{Label: labels[i], Score: score})
		}
	}
	return out
}
