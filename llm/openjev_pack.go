package llm

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/ollama/ollama/api"
)

// OpenJev / DiffusionGemma DG2 packing (LA16).
//
// WHY this file (not C++ packing):
//   - Same control-plane split as Laya PackQuestions and /v1/rerank: Go owns the
//     public wire + prompt/answer schema; the engine only runs the graph.
//   - An early C++ draft packed {state,questions} and returned answers with
//     model-written confidence — that is "JSON in chat", not System-1.
//
// Contract:
//   Go:   PackOpenJevPrompt → POST sibling /v1/systemone {prompt} → ParseOpenJevAnswers
//   C++:  denoise only ({prompt|messages} → {answer}); see llama/patches/diffusion/
//
// Confidence from the model text is NOT calibrated — stripped on parse; every
// answer gets calibrated:false until marker-logit readout lands.
// Docs: docs/diffusion-gemma-readout-design.md · docs/diffusion-gemma-llama-cpp-findings.md

const openJevPromptPreamble = `You are a structured System-1 decision model. Given STATE and QUESTIONS, emit:
1) One MARKER LINE per question (exact spelling; fill the value), then
2) A JSON object mapping each question_id to an answer.

Marker formats:
choice: <<question_id>> <criteria_key>
noul: <<question_id>> 0|1
score: <<question_id>> <level_index>
Example marker: <<dept>> billing

JSON shapes:
For type=choice: {"type":"choice","choice":"<one criteria key>"}
For type=noul: {"type":"noul","noul":0.0-1.0}  (marker 1 = statement holds / immediate action required; 0 otherwise)
For type=score: {"type":"score","score":<number>}
Do not include confidence. No markdown, no preamble.

`

// OpenJevMarkerText returns the DG3a marker string for a question id (Laya MASK analogue).
// WHY <<qid>>: short, unlikely in state text, must tokenize stably for C++ find.
func OpenJevMarkerText(questionID string) string {
	return "<<" + questionID + ">>"
}

// PackOpenJevPrompt builds the user prompt for DiffusionGemma denoise (DG2 v0 + DG3a markers).
//
// WHY sort question ids: Go map iteration is random; unstable order would make
// multi-question prompts non-reproducible across requests (same WHY as Laya).
func PackOpenJevPrompt(state any, questions map[string]api.DecisionQuestion) (string, error) {
	if len(questions) == 0 {
		return "", fmt.Errorf("questions required")
	}
	if state == nil {
		return "", fmt.Errorf("state is required")
	}
	stateJSON, err := json.Marshal(state)
	if err != nil {
		return "", fmt.Errorf("marshal state: %w", err)
	}
	ids := make([]string, 0, len(questions))
	for id := range questions {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	var b strings.Builder
	b.WriteString(openJevPromptPreamble)
	b.WriteString("STATE:\n")
	b.Write(stateJSON)
	b.WriteString("\n\nQUESTIONS:\n")
	for _, id := range ids {
		q := questions[id]
		qj, err := json.Marshal(q)
		if err != nil {
			return "", fmt.Errorf("marshal question %q: %w", id, err)
		}
		b.WriteString("- id=")
		b.WriteString(id)
		b.WriteByte(' ')
		b.Write(qj)
		b.WriteByte('\n')
	}
	var markerLines []string
	for _, id := range ids {
		q := questions[id]
		switch {
		case strings.EqualFold(q.Type, "choice") && len(q.Criteria) > 0:
			markerLines = append(markerLines, OpenJevMarkerText(id)+" <criteria_key>")
		case strings.EqualFold(q.Type, "noul"):
			markerLines = append(markerLines, OpenJevMarkerText(id)+" 0|1")
		case strings.EqualFold(q.Type, "score") && len(q.Criteria) > 0:
			markerLines = append(markerLines, OpenJevMarkerText(id)+" <level_index>")
		}
	}
	if len(markerLines) > 0 {
		b.WriteString("\nMARKER LINES (emit these first, filled):\n")
		for _, line := range markerLines {
			b.WriteString(line)
			b.WriteByte('\n')
		}
	}
	b.WriteString("\nEmit markers first, then JSON answers object:")
	return b.String(), nil
}

// OpenJevGatherSpec is one option's token ids for C++ answer-span gather (DG2c fallback).
type OpenJevGatherSpec struct {
	QuestionID string `json:"question_id"`
	OptionID   string `json:"option_id"`
	Tokens     []int  `json:"tokens"`
}

// OpenJevSlotSpec is one question's marker + options for DG3a post-marker logits.
type OpenJevSlotSpec struct {
	QuestionID   string              `json:"question_id"`
	MarkerTokens []int               `json:"marker_tokens"`
	Options      []OpenJevGatherSpec `json:"options"`
}

// OpenJevGatherHit is one option's logit stats from C++ readout.
type OpenJevGatherHit struct {
	QuestionID string  `json:"question_id"`
	OptionID   string  `json:"option_id"`
	LogitMax   float64 `json:"logit_max"`
	LogitMean  float64 `json:"logit_mean"`
	NTokens    int     `json:"n_tokens"`
}

// openJevOptionIDs returns the tokenized option vocabulary for gather/slots.
// choice → criteria keys; noul → "0"/"1" (DG9: avoid English true prior); score → "0".."n-1".
func openJevOptionIDs(q api.DecisionQuestion) []string {
	switch {
	case strings.EqualFold(q.Type, "choice"):
		return criteriaKeys(q.Criteria)
	case strings.EqualFold(q.Type, "noul"):
		// Laya semantic: noul = P(true). Slot tokens are "0"/"1" (Finding 17 / DG9).
		return []string{"0", "1"}
	case strings.EqualFold(q.Type, "score"):
		return scoreLevelIDs(q.Criteria)
	default:
		return nil
	}
}

// BuildOpenJevGatherSpecs tokenizes choice criteria keys and noul false/true.
func BuildOpenJevGatherSpecs(questions map[string]api.DecisionQuestion, tokenizeFn func(string) ([]int, error)) ([]OpenJevGatherSpec, error) {
	if tokenizeFn == nil {
		return nil, nil
	}
	ids := make([]string, 0, len(questions))
	for id := range questions {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	var out []OpenJevGatherSpec
	for _, qid := range ids {
		for _, opt := range openJevOptionIDs(questions[qid]) {
			toks, err := tokenizeFn(opt)
			if err != nil {
				return nil, fmt.Errorf("tokenize option %q/%q: %w", qid, opt, err)
			}
			if len(toks) == 0 {
				continue
			}
			out = append(out, OpenJevGatherSpec{QuestionID: qid, OptionID: opt, Tokens: toks})
		}
	}
	return out, nil
}

// BuildOpenJevSlotSpecs tokenizes <<qid>> markers + options for choice and noul (DG5a).
func BuildOpenJevSlotSpecs(questions map[string]api.DecisionQuestion, tokenizeFn func(string) ([]int, error)) ([]OpenJevSlotSpec, error) {
	if tokenizeFn == nil {
		return nil, nil
	}
	ids := make([]string, 0, len(questions))
	for id := range questions {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	var out []OpenJevSlotSpec
	for _, qid := range ids {
		optsIDs := openJevOptionIDs(questions[qid])
		if len(optsIDs) == 0 {
			continue
		}
		mtoks, err := tokenizeFn(OpenJevMarkerText(qid))
		if err != nil {
			return nil, fmt.Errorf("tokenize marker %q: %w", qid, err)
		}
		if len(mtoks) == 0 {
			continue
		}
		var opts []OpenJevGatherSpec
		for _, opt := range optsIDs {
			toks, err := tokenizeFn(opt)
			if err != nil {
				return nil, fmt.Errorf("tokenize option %q/%q: %w", qid, opt, err)
			}
			if len(toks) == 0 {
				continue
			}
			opts = append(opts, OpenJevGatherSpec{QuestionID: qid, OptionID: opt, Tokens: toks})
		}
		if len(opts) == 0 {
			continue
		}
		out = append(out, OpenJevSlotSpec{QuestionID: qid, MarkerTokens: mtoks, Options: opts})
	}
	return out, nil
}

// MergeOpenJevGather overlays temperature-scaled softmax over gather logits onto text-parsed answers.
// scoreSource is written into answers (e.g. answer_marker_logit).
// temperature <= 0 is treated as 1.0.
// calibratedChoiceScore (DG8): when true, choice/score answers get calibrated:true.
// noulBias (DG9): subtracted from the "1"/true logit before softmax (Finding 17).
// calibratedNoul (DG9): when true and bias path used, noul may be calibrated:true.
func MergeOpenJevGather(answers map[string]json.RawMessage, questions map[string]api.DecisionQuestion, hits []OpenJevGatherHit, scoreSource string, temperature float64, calibratedChoiceScore bool, noulBias float64, calibratedNoul bool) (map[string]json.RawMessage, error) {
	if len(hits) == 0 {
		return answers, nil
	}
	if scoreSource == "" {
		scoreSource = "final_answer_logit_gather"
	}
	temp := 1.0
	if temperature > 0 {
		temp = ClampTemperature(temperature)
	}
	byQ := map[string][]OpenJevGatherHit{}
	for _, h := range hits {
		byQ[h.QuestionID] = append(byQ[h.QuestionID], h)
	}
	out := make(map[string]json.RawMessage, len(answers))
	for k, v := range answers {
		out[k] = v
	}
	for qid, hs := range byQ {
		q := questions[qid]
		qtype := strings.ToLower(strings.TrimSpace(q.Type))
		if (qtype != "choice" && qtype != "noul" && qtype != "score") || len(hs) == 0 {
			continue
		}
		// Stable option order for SoftmaxTemp alignment.
		sort.Slice(hs, func(i, j int) bool {
			return hs[i].OptionID < hs[j].OptionID
		})
		logits := make([]float64, len(hs))
		ids := make([]string, len(hs))
		for i, h := range hs {
			logits[i] = h.LogitMean
			ids[i] = h.OptionID
			// DG9: subtract bias from positive (true/"1") class before softmax.
			if qtype == "noul" && noulBias != 0 && (h.OptionID == "1" || h.OptionID == "true") {
				logits[i] -= noulBias
			}
		}
		probsArr := SoftmaxTemp(logits, len(logits), temp)
		probs := make(map[string]float64, len(ids))
		bestID := ids[0]
		bestP := -1.0
		for i, id := range ids {
			probs[id] = probsArr[i]
			if probsArr[i] > bestP {
				bestP = probsArr[i]
				bestID = id
			}
		}
		obj := map[string]any{}
		if raw, ok := out[qid]; ok {
			_ = json.Unmarshal(raw, &obj)
		}
		delete(obj, "confidence")
		obj["score_source"] = scoreSource
		obj["temperature"] = temp
		switch qtype {
		case "noul":
			// Laya: noul = P(true). Slot ids are "0"/"1" (DG9) or legacy "false"/"true".
			noul := probs["1"]
			if _, ok := probs["1"]; !ok {
				noul = probs["true"]
			}
			obj["type"] = "noul"
			obj["noul"] = noul
			obj["probabilities"] = probs
			if noulBias != 0 {
				obj["noul_bias"] = noulBias
			}
			obj["calibrated"] = calibratedNoul
			delete(obj, "choice")
			delete(obj, "score")
		case "score":
			// Laya: score = E[level] under softmax; options are "0".."k-1".
			var exp float64
			legend := scoreLegend(q.Criteria)
			for id, p := range probs {
				var lvl int
				if _, err := fmt.Sscanf(id, "%d", &lvl); err == nil {
					exp += float64(lvl) * p
				}
			}
			obj["type"] = "score"
			obj["score"] = exp
			obj["probabilities"] = probs
			obj["calibrated"] = calibratedChoiceScore
			if len(legend) > 0 {
				obj["legend"] = legend
			}
			delete(obj, "choice")
			delete(obj, "noul")
		default:
			obj["type"] = "choice"
			obj["choice"] = bestID
			obj["probabilities"] = probs
			obj["calibrated"] = calibratedChoiceScore
		}
		b, err := json.Marshal(obj)
		if err != nil {
			return nil, err
		}
		out[qid] = b
	}
	return out, nil
}

// OpenJevTempCase is one labeled choice for FitOpenJevTemperature (DG4b).
type OpenJevTempCase struct {
	OptionIDs []string
	Logits    []float64
	Gold      string
}

// DefaultOpenJevTemperature is the DG4b grid-fit on testdata/openjev fixtures (seed=42).
// Override with ZEROLLAMA_OPENJEV_TEMP. Still not calibrated:true.
const DefaultOpenJevTemperature = 0.5

// DefaultOpenJevNoulBias is DG9 grid-fit offset subtracted from the positive-class logit
// before softmax (Finding 17). Override with ZEROLLAMA_OPENJEV_NOUL_BIAS.
// 0 until fit script writes a lab value into docs/testdata.
const DefaultOpenJevNoulBias = 0.0

// OpenJevNoulCase is one labeled noul for FitOpenJevNoulBias.
type OpenJevNoulCase struct {
	LogitFalse float64 // option "0" or "false"
	LogitTrue  float64 // option "1" or "true"
	GoldTrue   bool
}

// FitOpenJevNoulBias picks bias in [0,20] minimizing mean NLL of gold true/false labels
// at temperature temp (use OpenJevTemperature / DefaultOpenJevTemperature).
func FitOpenJevNoulBias(cases []OpenJevNoulCase, temp float64) (bestBias float64, bestNLL float64) {
	bestBias, bestNLL = 0, math.Inf(1)
	if len(cases) == 0 {
		return bestBias, bestNLL
	}
	if temp <= 0 {
		temp = 1
	}
	temp = ClampTemperature(temp)
	for b := 0.0; b <= 20.0+1e-9; b += 0.25 {
		var sum float64
		n := 0
		for _, c := range cases {
			logits := []float64{c.LogitFalse, c.LogitTrue - b}
			p := SoftmaxTemp(logits, 2, temp)
			prob := p[0]
			if c.GoldTrue {
				prob = p[1]
			}
			if prob < 1e-12 {
				prob = 1e-12
			}
			sum += -math.Log(prob)
			n++
		}
		if n == 0 {
			continue
		}
		nll := sum / float64(n)
		if nll < bestNLL {
			bestNLL = nll
			bestBias = b
		}
	}
	return bestBias, bestNLL
}

// FitOpenJevTemperature picks T in [0.5,5] minimizing mean NLL of gold labels.
// WHY grid (not gradient): tiny fixture sets; bounds match Laya ClampTemperature.
func FitOpenJevTemperature(cases []OpenJevTempCase) (bestT float64, bestNLL float64) {
	bestT, bestNLL = 1.0, math.Inf(1)
	if len(cases) == 0 {
		return bestT, bestNLL
	}
	for t := layaTempMin; t <= layaTempMax+1e-9; t += 0.05 {
		t = ClampTemperature(t)
		var sum float64
		n := 0
		for _, c := range cases {
			if len(c.OptionIDs) == 0 || len(c.OptionIDs) != len(c.Logits) {
				continue
			}
			goldIdx := -1
			for i, id := range c.OptionIDs {
				if id == c.Gold {
					goldIdx = i
					break
				}
			}
			if goldIdx < 0 {
				continue
			}
			p := SoftmaxTemp(append([]float64(nil), c.Logits...), len(c.Logits), t)
			prob := p[goldIdx]
			if prob < 1e-12 {
				prob = 1e-12
			}
			sum += -math.Log(prob)
			n++
		}
		if n == 0 {
			continue
		}
		nll := sum / float64(n)
		if nll < bestNLL {
			bestNLL = nll
			bestT = t
		}
	}
	return bestT, bestNLL
}

// ParseOpenJevAnswers extracts per-question answers from denoise text.
//
// WHY strip confidence: a model-authored "confidence":1.0 is not a calibrated
// score; leaving it would teach agents to trust DG2 v0 like Laya/CLM.
// WHY calibrated:false always: honest until marker-logit readout exists.
func ParseOpenJevAnswers(text string, questions map[string]api.DecisionQuestion) (map[string]json.RawMessage, error) {
	ids := make([]string, 0, len(questions))
	for id := range questions {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	out := make(map[string]json.RawMessage, len(questions))
	parsed, ok := extractJSONObject(text)
	if !ok {
		for _, id := range ids {
			raw, _ := json.Marshal(map[string]any{
				"type":       questions[id].Type,
				"raw":        text,
				"note":       "unparsed_v0_readout",
				"calibrated": false,
			})
			out[id] = raw
		}
		return out, nil
	}

	// Prefer nested "answers", else top-level question ids.
	src := parsed
	if nested, ok := parsed["answers"].(map[string]any); ok {
		src = nested
	}

	for _, id := range ids {
		v, ok := src[id]
		if !ok {
			raw, _ := json.Marshal(map[string]any{
				"type":       questions[id].Type,
				"note":       "missing_answer",
				"calibrated": false,
			})
			out[id] = raw
			continue
		}
		obj, _ := v.(map[string]any)
		if obj == nil {
			raw, _ := json.Marshal(map[string]any{
				"type":       questions[id].Type,
				"raw":        v,
				"calibrated": false,
			})
			out[id] = raw
			continue
		}
		delete(obj, "confidence") // never trust model-written confidence
		obj["calibrated"] = false
		if t, _ := obj["type"].(string); t == "" && questions[id].Type != "" {
			obj["type"] = questions[id].Type
		}
		if choice, _ := obj["choice"].(string); choice != "" && len(questions[id].Criteria) > 0 {
			keys := criteriaKeys(questions[id].Criteria)
			if len(keys) > 0 && !containsString(keys, choice) {
				obj["note"] = "choice_not_in_criteria"
			}
		}
		raw, err := json.Marshal(obj)
		if err != nil {
			return nil, err
		}
		out[id] = raw
	}
	return out, nil
}

func extractJSONObject(text string) (map[string]any, bool) {
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start < 0 || end <= start {
		return nil, false
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(text[start:end+1]), &m); err != nil {
		return nil, false
	}
	return m, true
}

func criteriaKeys(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// scoreLevelIDs returns "0".."n-1" for a score criteria JSON array (Laya order).
func scoreLevelIDs(raw json.RawMessage) []string {
	levels := scoreLevels(raw)
	if len(levels) == 0 {
		return nil
	}
	out := make([]string, len(levels))
	for i := range levels {
		out[i] = fmt.Sprintf("%d", i)
	}
	return out
}

func scoreLevels(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}
	var levels []string
	if err := json.Unmarshal(raw, &levels); err != nil {
		return nil
	}
	return levels
}

func scoreLegend(raw json.RawMessage) map[string]string {
	levels := scoreLevels(raw)
	if len(levels) == 0 {
		return nil
	}
	out := make(map[string]string, len(levels))
	for i, s := range levels {
		out[fmt.Sprintf("%d", i)] = s
	}
	return out
}

func containsString(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}
