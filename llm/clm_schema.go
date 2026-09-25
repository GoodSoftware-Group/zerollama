package llm

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// CLM question packing mirrors Contrastive-LM schema.py (TypeSafe /v1/systemone).

func clmToText(x any, indent int) string {
	if x == nil {
		return ""
	}
	switch v := x.(type) {
	case string:
		return v
	case bool:
		if v {
			return "true"
		}
		return "false"
	case float64:
		return strconv.FormatFloat(v, 'g', -1, 64)
	case float32:
		return strconv.FormatFloat(float64(v), 'g', -1, 32)
	case int:
		return strconv.Itoa(v)
	case int64:
		return strconv.FormatInt(v, 10)
	case json.Number:
		return v.String()
	case map[string]any:
		pad := strings.Repeat(" ", indent)
		parts := make([]string, 0, len(v))
		// Preserve insertion order when possible via sorted keys only as fallback —
		// Go maps are unordered; prefer json.RawMessage path for criteria.
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			val := v[k]
			switch val.(type) {
			case map[string]any, []any:
				if val != nil {
					parts = append(parts, fmt.Sprintf("%s%s:\n%s", pad, k, clmToText(val, indent+2)))
					continue
				}
			}
			parts = append(parts, fmt.Sprintf("%s%s: %s", pad, k, clmToText(val, 0)))
		}
		sep := "\n"
		if indent == 0 {
			sep = "\n\n"
		}
		return strings.Join(parts, sep)
	case []any:
		pad := strings.Repeat(" ", indent)
		parts := make([]string, 0, len(v))
		for _, val := range v {
			switch val.(type) {
			case map[string]any, []any:
				parts = append(parts, fmt.Sprintf("%s-\n%s", pad, clmToText(val, indent+2)))
			default:
				parts = append(parts, fmt.Sprintf("%s- %s", pad, clmToText(val, 0)))
			}
		}
		return strings.Join(parts, "\n")
	default:
		b, err := json.Marshal(x)
		if err != nil {
			return fmt.Sprint(x)
		}
		return string(b)
	}
}

func clmStateText(state any, instructions string) string {
	s := strings.TrimSpace(clmToText(state, 0))
	i := strings.TrimSpace(instructions)
	switch {
	case s != "" && i != "":
		return s + "\n\n" + i
	case s != "":
		return s
	default:
		return i
	}
}

// clmOrderedObjectKeys returns object keys in JSON key order from RawMessage.
func clmOrderedObjectKeys(raw json.RawMessage) ([]string, map[string]json.RawMessage, error) {
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil {
		return nil, nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, nil, fmt.Errorf("expected JSON object")
	}
	keys := []string{}
	vals := map[string]json.RawMessage{}
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil, nil, err
		}
		k, ok := kt.(string)
		if !ok {
			return nil, nil, fmt.Errorf("expected string key")
		}
		var v json.RawMessage
		if err := dec.Decode(&v); err != nil {
			return nil, nil, err
		}
		keys = append(keys, k)
		vals[k] = v
	}
	return keys, vals, nil
}

type clmPair struct {
	StateText string
	Keys      []string
	Cands     []string
}

func clmCandidates(q DecisionQuestion) (keys []string, texts []string, err error) {
	t := strings.ToLower(strings.TrimSpace(q.Type))
	ins := strings.TrimSpace(q.Instructions)
	switch t {
	case "choice":
		if len(q.Criteria) == 0 {
			return nil, nil, fmt.Errorf("choice question needs a non-empty criteria object")
		}
		keys, vals, err := clmOrderedObjectKeys(q.Criteria)
		if err != nil {
			return nil, nil, fmt.Errorf("choice criteria: %w", err)
		}
		if len(keys) == 0 {
			return nil, nil, fmt.Errorf("choice question needs a non-empty criteria object")
		}
		texts = make([]string, len(keys))
		for i, k := range keys {
			var s string
			_ = json.Unmarshal(vals[k], &s)
			if strings.TrimSpace(s) == "" {
				// non-string or empty → render
				var anyv any
				_ = json.Unmarshal(vals[k], &anyv)
				rendered := strings.TrimSpace(clmToText(anyv, 0))
				if rendered == "" || rendered == "null" {
					texts[i] = k
				} else {
					texts[i] = rendered
				}
			} else {
				texts[i] = s
			}
		}
		return keys, texts, nil
	case "score":
		var levels []any
		if err := json.Unmarshal(q.Criteria, &levels); err != nil || len(levels) < 2 {
			return nil, nil, fmt.Errorf("score question needs criteria as an ordered list of >= 2 levels")
		}
		keys = make([]string, len(levels))
		texts = make([]string, len(levels))
		for i, c := range levels {
			keys[i] = strconv.Itoa(i)
			texts[i] = clmToText(c, 0)
		}
		return keys, texts, nil
	case "noul":
		crit := map[string]string{}
		if len(q.Criteria) > 0 {
			_ = json.Unmarshal(q.Criteria, &crit)
		}
		keys = []string{"false", "true"}
		texts = make([]string, 2)
		for i, k := range keys {
			d := strings.TrimSpace(crit[k])
			if d == "" {
				if ins != "" {
					if k == "true" {
						d = "Yes. This is true: " + ins
					} else {
						d = "No. This is false: " + ins
					}
				} else {
					d = k
				}
			}
			texts[i] = k + ": " + d
		}
		return keys, texts, nil
	default:
		return nil, nil, fmt.Errorf("unknown question type %q", q.Type)
	}
}

func clmBuildPairs(state any, questions map[string]DecisionQuestion) (map[string]clmPair, []string, error) {
	ids := make([]string, 0, len(questions))
	for id := range questions {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make(map[string]clmPair, len(questions))
	for _, id := range ids {
		q := questions[id]
		keys, cands, err := clmCandidates(q)
		if err != nil {
			return nil, nil, fmt.Errorf("question %q: %w", id, err)
		}
		out[id] = clmPair{
			StateText: clmStateText(state, q.Instructions),
			Keys:      keys,
			Cands:     cands,
		}
	}
	return out, ids, nil
}

func clmSoftmax(logits []float64) []float64 {
	if len(logits) == 0 {
		return nil
	}
	m := logits[0]
	for _, v := range logits[1:] {
		if v > m {
			m = v
		}
	}
	e := make([]float64, len(logits))
	var z float64
	for i, v := range logits {
		e[i] = math.Exp(v - m)
		z += e[i]
	}
	for i := range e {
		e[i] /= z
	}
	return e
}

func clmConfidence(probs []float64) float64 {
	if len(probs) < 2 {
		return 1
	}
	j := 0
	for i := 1; i < len(probs); i++ {
		if probs[i] > probs[j] {
			j = i
		}
	}
	var rest float64
	for i, p := range probs {
		if i != j {
			rest += p
		}
	}
	c := probs[j] - rest/float64(len(probs)-1)
	if c < 0 {
		return 0
	}
	if c > 1 {
		return 1
	}
	return c
}

func clmAnswerFromLogits(q DecisionQuestion, keys []string, logits []float64) (json.RawMessage, error) {
	probs := clmSoftmax(logits)
	dist := make(map[string]float64, len(keys))
	for i, k := range keys {
		dist[k] = probs[i]
	}
	t := strings.ToLower(strings.TrimSpace(q.Type))
	switch t {
	case "noul":
		return json.Marshal(map[string]any{
			"type": "noul",
			"noul": dist["true"],
		})
	case "choice":
		j := 0
		for i := 1; i < len(probs); i++ {
			if probs[i] > probs[j] {
				j = i
			}
		}
		return json.Marshal(map[string]any{
			"type":          "choice",
			"choice":        keys[j],
			"confidence":    clmConfidence(probs),
			"probabilities": dist,
		})
	case "score":
		var levels []any
		_ = json.Unmarshal(q.Criteria, &levels)
		var score float64
		for i, p := range probs {
			score += float64(i) * p
		}
		legend := map[string]string{}
		for i, c := range levels {
			if s, ok := c.(string); ok {
				legend[strconv.Itoa(i)] = s
			} else {
				legend[strconv.Itoa(i)] = clmToText(c, 0)
			}
		}
		return json.Marshal(map[string]any{
			"type":          "score",
			"score":         score,
			"confidence":    clmConfidence(probs),
			"legend":        legend,
			"probabilities": dist,
		})
	default:
		return nil, fmt.Errorf("unknown type %q", q.Type)
	}
}
