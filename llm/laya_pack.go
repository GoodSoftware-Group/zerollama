package llm

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

// Packing + calibration for Laya System-1 decisions (LAYA2).
//
// WHY separate from chat/rerank: Laya answers typed choice/score/noul in one
// encoder pass with calibrated probs + act head. Go owns the Jev wire and
// sequence layout; llama-server owns encode+CPU head (LA16 split, same idea as
// /v1/rerank). See docs/laya-llama-cpp-findings.md.

// QTYPES maps Laya question type names to type_emb rows.
var QTYPES = map[string]int{
	"choice": 0,
	"score":  1,
	"noul":   2,
}

// QTYPENames is the reverse of QTYPES.
var QTYPENames = map[int]string{
	0: "choice",
	1: "score",
	2: "noul",
}

const (
	layaTempMin = 0.5
	layaTempMax = 5.0

	defaultLayaMaxLen     = 512
	defaultLayaHeadMaxLen = 192
	layaOptionTokCap      = 48
)

var defaultNoulLabels = map[string]string{"false": "false", "true": "true"}

// LayaTokenizer is the subset of a BERT-style tokenizer needed to pack Laya sequences.
type LayaTokenizer interface {
	Encode(text string) ([]int, error)
	MaskToken() string
	MaskTokenID() int
	ClsTokenID() int
	SepTokenID() int
}

// LayaPackedInput is one tokenized question row for llama-server /v1/decisions.
type LayaPackedInput struct {
	Tokens    []int  `json:"tokens"`
	MarkerPos []int  `json:"marker_pos"`
	QType     int    `json:"qtype"`
	Question  string `json:"question_id,omitempty"`
}

// SerializeState renders state text the way Laya common.py does.
func SerializeState(state any) (string, error) {
	switch v := state.(type) {
	case nil:
		return "", nil
	case string:
		return v, nil
	case json.RawMessage:
		if len(v) == 0 {
			return "", nil
		}
		return string(v), nil
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return "", err
		}
		return string(b), nil
	}
}

// RenderCriterion mirrors laya.common.render_criterion.
func RenderCriterion(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	default:
		return renderCompactJSON(v)
	}
}

func renderCompactJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprint(v)
	}
	// json.Marshal already uses "," and ":" without spaces; Python uses ", " / ": ".
	var out strings.Builder
	inString := false
	escape := false
	for i := 0; i < len(b); i++ {
		c := b[i]
		if inString {
			out.WriteByte(c)
			if escape {
				escape = false
				continue
			}
			if c == '\\' {
				escape = true
				continue
			}
			if c == '"' {
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = true
			out.WriteByte(c)
		case ':':
			out.WriteString(": ")
		case ',':
			out.WriteString(", ")
		default:
			out.WriteByte(c)
		}
	}
	return out.String()
}

func resolveNoulLabels(labels map[string]string) (falseLabel, trueLabel string, err error) {
	if labels == nil {
		labels = defaultNoulLabels
	}
	f, fok := labels["false"]
	t, tok := labels["true"]
	if !fok || !tok || len(labels) != 2 {
		return "", "", fmt.Errorf("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
	}
	f, t = strings.TrimSpace(f), strings.TrimSpace(t)
	if f == "" || t == "" || f == t {
		return "", "", fmt.Errorf("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
	}
	return f, t, nil
}

// RenderOptions renders option texts in label-index order (Laya common.py).
// Noul semantic order is always [false, true].
func RenderOptions(q DecisionQuestion) ([]string, error) {
	t := strings.ToLower(strings.TrimSpace(q.Type))
	if t != "noul" && len(q.Labels) > 0 {
		return nil, fmt.Errorf("labels is only supported for noul questions")
	}
	switch t {
	case "choice":
		pairs, err := orderedObjectPairs(q.Criteria)
		if err != nil {
			return nil, err
		}
		out := make([]string, 0, len(pairs))
		for _, p := range pairs {
			if p.Value == nil || p.Value == "" {
				out = append(out, p.Key)
				continue
			}
			out = append(out, p.Key+": "+RenderCriterion(p.Value))
		}
		return out, nil
	case "score":
		levels, err := decodeStringList(q.Criteria)
		if err != nil {
			return nil, err
		}
		out := make([]string, len(levels))
		for i, c := range levels {
			out[i] = fmt.Sprintf("level %d: %s", i, RenderCriterion(c))
		}
		return out, nil
	case "noul":
		crit, err := decodeObjectAny(q.Criteria)
		if err != nil {
			return nil, err
		}
		falseLabel, trueLabel, err := resolveNoulLabels(q.Labels)
		if err != nil {
			return nil, err
		}
		falseCrit, trueCrit := crit["false"], crit["true"]
		falseText := "no, the statement does not hold"
		if falseCrit != nil && falseCrit != "" {
			falseText = RenderCriterion(falseCrit)
		}
		trueText := "yes, the statement holds"
		if trueCrit != nil && trueCrit != "" {
			trueText = RenderCriterion(trueCrit)
		}
		return []string{
			falseLabel + ": " + falseText,
			trueLabel + ": " + trueText,
		}, nil
	default:
		return nil, fmt.Errorf("unknown question type %q", q.Type)
	}
}

type objectPair struct {
	Key   string
	Value any
}

func orderedObjectPairs(raw json.RawMessage) ([]objectPair, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, fmt.Errorf("choice criteria required")
	}
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	delim, ok := tok.(json.Delim)
	if !ok || delim != '{' {
		return nil, fmt.Errorf("choice criteria must be a JSON object")
	}
	var pairs []objectPair
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		key, ok := keyTok.(string)
		if !ok {
			return nil, fmt.Errorf("choice criteria key must be string")
		}
		var val any
		if err := dec.Decode(&val); err != nil {
			return nil, err
		}
		pairs = append(pairs, objectPair{Key: key, Value: val})
	}
	tok, err = dec.Token()
	if err != nil {
		return nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '}' {
		return nil, fmt.Errorf("choice criteria: expected end of object")
	}
	return pairs, nil
}

func decodeObjectAny(raw json.RawMessage) (map[string]any, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return map[string]any{}, nil
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("noul criteria must be a JSON object: %w", err)
	}
	return m, nil
}

func decodeStringList(raw json.RawMessage) ([]any, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, fmt.Errorf("score criteria required")
	}
	var list []any
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, fmt.Errorf("score criteria must be a JSON array: %w", err)
	}
	return list, nil
}

// BuildSequence packs one question: [CLS] instructions [SEP] [MASK] opt… [SEP] state [SEP].
func BuildSequence(tok LayaTokenizer, state any, q DecisionQuestion, maxLen, headMaxLen int, optionOrder []int, truncateLeft bool) (ids []int, markers []int, err error) {
	if maxLen <= 0 {
		maxLen = defaultLayaMaxLen
	}
	if headMaxLen <= 0 {
		headMaxLen = defaultLayaHeadMaxLen
	}
	maskTok := tok.MaskToken()
	opts, err := RenderOptions(q)
	if err != nil {
		return nil, nil, err
	}
	order := optionOrder
	if order == nil {
		order = make([]int, len(opts))
		for i := range opts {
			order[i] = i
		}
	}
	ins := strings.ReplaceAll(q.Instructions, maskTok, " ")
	headIDs, err := tok.Encode(fmt.Sprintf("%s question: %s", strings.ToLower(strings.TrimSpace(q.Type)), ins))
	if err != nil {
		return nil, nil, err
	}
	optIDs := make([][]int, 0, len(order))
	for _, i := range order {
		if i < 0 || i >= len(opts) {
			return nil, nil, fmt.Errorf("option_order index %d out of range", i)
		}
		text := " " + strings.ReplaceAll(opts[i], maskTok, " ")
		oids, err := tok.Encode(text)
		if err != nil {
			return nil, nil, err
		}
		if len(oids) > layaOptionTokCap {
			oids = oids[:layaOptionTokCap]
		}
		optIDs = append(optIDs, append([]int{tok.MaskTokenID()}, oids...))
	}
	optBudget := headMaxLen
	for _, o := range optIDs {
		optBudget -= len(o)
	}
	if optBudget < 16 {
		per := max(4, (headMaxLen-16)/max(1, len(optIDs)))
		for i := range optIDs {
			if len(optIDs[i]) > per {
				optIDs[i] = optIDs[i][:per]
			}
		}
		optBudget = headMaxLen
		for _, o := range optIDs {
			optBudget -= len(o)
		}
	}
	headCap := max(8, optBudget)
	if len(headIDs) > headCap {
		headIDs = headIDs[:headCap]
	}
	ids = []int{tok.ClsTokenID()}
	ids = append(ids, headIDs...)
	ids = append(ids, tok.SepTokenID())
	for _, o := range optIDs {
		markers = append(markers, len(ids))
		ids = append(ids, o...)
	}
	ids = append(ids, tok.SepTokenID())
	room := max(0, maxLen-len(ids)-1)
	stText, err := SerializeState(state)
	if err != nil {
		return nil, nil, err
	}
	st, err := tok.Encode(strings.ReplaceAll(stText, maskTok, " "))
	if err != nil {
		return nil, nil, err
	}
	if truncateLeft {
		if room < len(st) {
			st = st[len(st)-room:]
		}
	} else if len(st) > room {
		st = st[:room]
	}
	ids = append(ids, st...)
	ids = append(ids, tok.SepTokenID())
	if len(ids) > maxLen {
		ids = ids[:maxLen]
	}
	kept := markers[:0]
	for _, m := range markers {
		if m < maxLen {
			kept = append(kept, m)
		}
	}
	return ids, kept, nil
}

// PackQuestions builds tokenized inputs for every question against one state.
// WHY sort by question_id: Go map iteration is random; server results are
// position-aligned (and may echo question_id). Unstable order mislabels logits.
func PackQuestions(tok LayaTokenizer, state any, questions map[string]DecisionQuestion, maxLen, headMaxLen int) ([]LayaPackedInput, error) {
	ids := make([]string, 0, len(questions))
	for qid := range questions {
		ids = append(ids, qid)
	}
	sort.Strings(ids)
	out := make([]LayaPackedInput, 0, len(questions))
	for _, qid := range ids {
		q := questions[qid]
		qt, ok := QTYPES[strings.ToLower(strings.TrimSpace(q.Type))]
		if !ok {
			return nil, fmt.Errorf("question %q: unknown type %q", qid, q.Type)
		}
		tokIDs, markers, err := BuildSequence(tok, state, q, maxLen, headMaxLen, nil, false)
		if err != nil {
			return nil, fmt.Errorf("question %q: %w", qid, err)
		}
		out = append(out, LayaPackedInput{
			Tokens:    tokIDs,
			MarkerPos: markers,
			QType:     qt,
			Question:  qid,
		})
	}
	return out, nil
}

// ConfidenceFromProbs is normalized Shannon entropy confidence: 1 - H(p)/log(k).
func ConfidenceFromProbs(p []float64, k int) float64 {
	if k < 2 {
		return 1.0
	}
	if k > len(p) {
		k = len(p)
	}
	var ent float64
	for i := 0; i < k; i++ {
		pi := p[i]
		if pi < 1e-12 {
			pi = 1e-12
		} else if pi > 1.0 {
			pi = 1.0
		}
		ent -= pi * math.Log(pi)
	}
	c := 1.0 - ent/math.Log(float64(k))
	if c < 0 {
		return 0
	}
	if c > 1 {
		return 1
	}
	return c
}

// TempBucket returns the calibration bucket key for (qtype, option count).
func TempBucket(qtype, k int) string {
	size := "11+"
	switch {
	case k <= 2:
		size = "2"
	case k <= 5:
		size = "3-5"
	case k <= 10:
		size = "6-10"
	}
	name, ok := QTYPENames[qtype]
	if !ok {
		name = fmt.Sprintf("%d", qtype)
	}
	return name + ":" + size
}

// ClampTemperature confines t to [0.5, 5.0], falling back to 1.0 for non-finite values.
func ClampTemperature(t float64) float64 {
	if math.IsNaN(t) || math.IsInf(t, 0) {
		return 1.0
	}
	if t < layaTempMin {
		return layaTempMin
	}
	if t > layaTempMax {
		return layaTempMax
	}
	return t
}

// SoftmaxTemp applies temperature scaling then softmax over the first k logits.
func SoftmaxTemp(logits []float64, k int, temperature float64) []float64 {
	if k <= 0 {
		return nil
	}
	if k > len(logits) {
		k = len(logits)
	}
	t := temperature
	if t <= 0 || math.IsNaN(t) || math.IsInf(t, 0) {
		t = 1.0
	}
	maxZ := logits[0] / t
	for i := 1; i < k; i++ {
		if z := logits[i] / t; z > maxZ {
			maxZ = z
		}
	}
	out := make([]float64, k)
	var sum float64
	for i := 0; i < k; i++ {
		out[i] = math.Exp(logits[i]/t - maxZ)
		sum += out[i]
	}
	if sum == 0 {
		for i := range out {
			out[i] = 1.0 / float64(k)
		}
		return out
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// DecodeAnswer shapes one question's logits + act into a typed answer JSON.
// temperatures[qtype] and temperatureByOptions[TempBucket] default to 1.0 when missing.
func DecodeAnswer(q DecisionQuestion, logits, act []float64, temperatures []float64, temperatureByOptions map[string]float64) (json.RawMessage, error) {
	t := strings.ToLower(strings.TrimSpace(q.Type))
	qt, ok := QTYPES[t]
	if !ok {
		return nil, fmt.Errorf("unknown question type %q", q.Type)
	}
	opts, err := RenderOptions(q)
	if err != nil {
		return nil, err
	}
	k := len(opts)
	if k == 0 {
		return nil, fmt.Errorf("question has no options")
	}
	temp := 1.0
	if qt < len(temperatures) {
		temp = ClampTemperature(temperatures[qt])
	}
	if temperatureByOptions != nil {
		if v, ok := temperatureByOptions[TempBucket(qt, k)]; ok {
			temp = ClampTemperature(v)
		}
	}
	p := SoftmaxTemp(logits, k, temp)
	actP := 0.0
	if len(act) > 0 {
		actP = act[0]
	}
	ext := DecisionAction{ActProbability: round4(actP)}
	conf := round4(ConfidenceFromProbs(p, k))

	switch t {
	case "choice":
		pairs, err := orderedObjectPairs(q.Criteria)
		if err != nil {
			return nil, err
		}
		keys := make([]string, len(pairs))
		probs := make(map[string]float64, len(pairs))
		best := 0
		for i, pair := range pairs {
			keys[i] = pair.Key
			probs[pair.Key] = round4(p[i])
			if p[i] > p[best] {
				best = i
			}
		}
		return json.Marshal(ChoiceAnswer{
			Type:          "choice",
			Choice:        keys[best],
			Probabilities: probs,
			Confidence:    conf,
			Action:        ext,
		})
	case "score":
		levels, err := decodeStringList(q.Criteria)
		if err != nil {
			return nil, err
		}
		var exp float64
		legend := make(map[string]string, k)
		probs := make(map[string]float64, k)
		for i := 0; i < k; i++ {
			exp += float64(i) * p[i]
			legend[fmt.Sprintf("%d", i)] = RenderCriterion(levels[i])
			probs[fmt.Sprintf("%d", i)] = round4(p[i])
		}
		return json.Marshal(ScoreAnswer{
			Type:          "score",
			Score:         round4(exp),
			Legend:        legend,
			Probabilities: probs,
			Confidence:    conf,
			Action:        ext,
		})
	case "noul":
		noul := 0.0
		if len(p) > 1 {
			noul = p[1]
		}
		return json.Marshal(NoulAnswer{
			Type:       "noul",
			Noul:       round4(noul),
			Confidence: round4(math.Max(noul, 1.0-noul)),
			Action:     ext,
		})
	default:
		return nil, fmt.Errorf("unknown question type %q", q.Type)
	}
}

func round4(v float64) float64 {
	return math.Round(v*10000) / 10000
}
