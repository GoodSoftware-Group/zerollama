package llm

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/require"
)

func TestMergeOpenJevGather(t *testing.T) {
	qs := map[string]api.DecisionQuestion{
		"dept": {Type: "choice", Criteria: json.RawMessage(`{"billing":"x","tech":"y"}`)},
	}
	answers := map[string]json.RawMessage{
		"dept": json.RawMessage(`{"type":"choice","choice":"tech","calibrated":false}`),
	}
	hits := []OpenJevGatherHit{
		{QuestionID: "dept", OptionID: "billing", LogitMean: 2.0},
		{QuestionID: "dept", OptionID: "tech", LogitMean: 0.1},
	}
	out, err := MergeOpenJevGather(answers, qs, hits, "answer_marker_logit", 1.0, false, 0, false)
	require.NoError(t, err)
	var obj map[string]any
	require.NoError(t, json.Unmarshal(out["dept"], &obj))
	require.Equal(t, "billing", obj["choice"])
	require.Equal(t, false, obj["calibrated"])
	require.Equal(t, "answer_marker_logit", obj["score_source"])
	require.Equal(t, 1.0, obj["temperature"])
	probs, _ := obj["probabilities"].(map[string]any)
	require.Greater(t, probs["billing"].(float64), probs["tech"].(float64))
}

func TestMergeOpenJevGatherCalibratedChoiceScore(t *testing.T) {
	qs := map[string]api.DecisionQuestion{
		"dept":   {Type: "choice", Criteria: json.RawMessage(`{"billing":"x","tech":"y"}`)},
		"urgent": {Type: "noul"},
		"sev":    {Type: "score", Criteria: json.RawMessage(`["low","high"]`)},
	}
	answers := map[string]json.RawMessage{
		"dept":   json.RawMessage(`{"type":"choice"}`),
		"urgent": json.RawMessage(`{"type":"noul"}`),
		"sev":    json.RawMessage(`{"type":"score"}`),
	}
	hits := []OpenJevGatherHit{
		{QuestionID: "dept", OptionID: "billing", LogitMean: 2.0},
		{QuestionID: "dept", OptionID: "tech", LogitMean: 0.1},
		{QuestionID: "urgent", OptionID: "false", LogitMean: 0.2},
		{QuestionID: "urgent", OptionID: "true", LogitMean: 2.5},
		{QuestionID: "sev", OptionID: "0", LogitMean: 0.1},
		{QuestionID: "sev", OptionID: "1", LogitMean: 2.0},
	}
	out, err := MergeOpenJevGather(answers, qs, hits, "answer_marker_logit", 1.0, true, 0, false)
	require.NoError(t, err)
	var dept, urgent, sev map[string]any
	require.NoError(t, json.Unmarshal(out["dept"], &dept))
	require.NoError(t, json.Unmarshal(out["urgent"], &urgent))
	require.NoError(t, json.Unmarshal(out["sev"], &sev))
	require.Equal(t, true, dept["calibrated"])
	require.Equal(t, true, sev["calibrated"])
	require.Equal(t, false, urgent["calibrated"])
}

func TestFitOpenJevTemperaturePrefersGold(t *testing.T) {
	cases := []OpenJevTempCase{
		{OptionIDs: []string{"a", "b"}, Logits: []float64{3.0, 1.0}, Gold: "a"},
		{OptionIDs: []string{"a", "b"}, Logits: []float64{0.5, 2.5}, Gold: "b"},
	}
	tFit, nll := FitOpenJevTemperature(cases)
	require.True(t, tFit >= 0.5 && tFit <= 5.0)
	require.False(t, math.IsInf(nll, 0))
	require.Less(t, nll, math.Inf(1))
}

func TestPackOpenJevPromptIncludesMarkers(t *testing.T) {
	qs := map[string]api.DecisionQuestion{
		"dept":   {Type: "choice", Instructions: "Which?", Criteria: json.RawMessage(`{"billing":"x","tech":"y"}`)},
		"urgent": {Type: "noul", Instructions: "Urgent?"},
	}
	p, err := PackOpenJevPrompt(map[string]string{"body": "refund"}, qs)
	require.NoError(t, err)
	require.Contains(t, p, "<<dept>>")
	require.Contains(t, p, "<<urgent>> 0|1")
	require.Contains(t, p, "marker 1 = statement holds")
	require.Contains(t, p, "MARKER LINES")
	require.Contains(t, p, "Emit markers first")
}

func TestMergeOpenJevGatherNoul(t *testing.T) {
	qs := map[string]api.DecisionQuestion{
		"urgent": {Type: "noul", Instructions: "Urgent?"},
	}
	answers := map[string]json.RawMessage{
		"urgent": json.RawMessage(`{"type":"noul","calibrated":false}`),
	}
	hits := []OpenJevGatherHit{
		{QuestionID: "urgent", OptionID: "0", LogitMean: 0.2},
		{QuestionID: "urgent", OptionID: "1", LogitMean: 2.5},
	}
	out, err := MergeOpenJevGather(answers, qs, hits, "answer_marker_logit", 1.0, false, 0, false)
	require.NoError(t, err)
	var obj map[string]any
	require.NoError(t, json.Unmarshal(out["urgent"], &obj))
	require.Equal(t, "noul", obj["type"])
	require.Greater(t, obj["noul"].(float64), 0.5)
	require.Equal(t, false, obj["calibrated"])
	_, hasChoice := obj["choice"]
	require.False(t, hasChoice)
}

func TestBuildOpenJevSlotSpecsNoul(t *testing.T) {
	qs := map[string]api.DecisionQuestion{
		"urgent": {Type: "noul"},
	}
	tok := func(s string) ([]int, error) {
		switch s {
		case "<<urgent>>":
			return []int{1}, nil
		case "0":
			return []int{2}, nil
		case "1":
			return []int{3}, nil
		default:
			return []int{9}, nil
		}
	}
	slots, err := BuildOpenJevSlotSpecs(qs, tok)
	require.NoError(t, err)
	require.Len(t, slots, 1)
	require.Len(t, slots[0].Options, 2)
}

func TestMergeOpenJevGatherScore(t *testing.T) {
	qs := map[string]api.DecisionQuestion{
		"sev": {Type: "score", Criteria: json.RawMessage(`["low","med","high"]`)},
	}
	answers := map[string]json.RawMessage{
		"sev": json.RawMessage(`{"type":"score","calibrated":false}`),
	}
	hits := []OpenJevGatherHit{
		{QuestionID: "sev", OptionID: "0", LogitMean: 0.1},
		{QuestionID: "sev", OptionID: "1", LogitMean: 0.2},
		{QuestionID: "sev", OptionID: "2", LogitMean: 3.0},
	}
	out, err := MergeOpenJevGather(answers, qs, hits, "answer_marker_logit", 1.0, false, 0, false)
	require.NoError(t, err)
	var obj map[string]any
	require.NoError(t, json.Unmarshal(out["sev"], &obj))
	require.Equal(t, "score", obj["type"])
	require.Greater(t, obj["score"].(float64), 1.5)
	require.Equal(t, false, obj["calibrated"])
	legend, _ := obj["legend"].(map[string]any)
	require.Equal(t, "high", legend["2"])
}

func TestPackOpenJevPromptIncludesScoreMarker(t *testing.T) {
	qs := map[string]api.DecisionQuestion{
		"sev": {Type: "score", Instructions: "Severity?", Criteria: json.RawMessage(`["low","high"]`)},
	}
	p, err := PackOpenJevPrompt(map[string]string{"body": "outage"}, qs)
	require.NoError(t, err)
	require.Contains(t, p, "<<sev>> <level_index>")
}

func TestBuildOpenJevSlotSpecs(t *testing.T) {
	qs := map[string]api.DecisionQuestion{
		"dept": {Type: "choice", Criteria: json.RawMessage(`{"billing":"x","tech":"y"}`)},
	}
	tok := func(s string) ([]int, error) {
		if s == "<<dept>>" {
			return []int{9, 9}, nil
		}
		if s == "billing" {
			return []int{1}, nil
		}
		return []int{2}, nil
	}
	slots, err := BuildOpenJevSlotSpecs(qs, tok)
	require.NoError(t, err)
	require.Len(t, slots, 1)
	require.Equal(t, []int{9, 9}, slots[0].MarkerTokens)
	require.Len(t, slots[0].Options, 2)
}

func TestPackOpenJevPromptStableOrder(t *testing.T) {
	qs := map[string]api.DecisionQuestion{
		"z": {Type: "noul", Instructions: "z?"},
		"a": {Type: "choice", Instructions: "a?", Criteria: json.RawMessage(`{"billing":"refunds","tech":"bugs"}`)},
	}
	p, err := PackOpenJevPrompt(map[string]string{"body": "refund"}, qs)
	require.NoError(t, err)
	require.Contains(t, p, "STATE:")
	require.Contains(t, p, `"body":"refund"`)
	ia := strings.Index(p, "id=a")
	iz := strings.Index(p, "id=z")
	require.GreaterOrEqual(t, ia, 0)
	require.Greater(t, iz, ia)
	require.NotContains(t, p, `"confidence"`)
	require.Contains(t, p, "Do not include confidence")
}

func TestParseOpenJevAnswersStripsConfidence(t *testing.T) {
	qs := map[string]api.DecisionQuestion{
		"dept": {Type: "choice", Criteria: json.RawMessage(`{"billing":"x","tech":"y"}`)},
	}
	text := `{"dept":{"type":"choice","choice":"billing","confidence":1.0}}`
	out, err := ParseOpenJevAnswers(text, qs)
	require.NoError(t, err)
	var obj map[string]any
	require.NoError(t, json.Unmarshal(out["dept"], &obj))
	require.Equal(t, "billing", obj["choice"])
	_, hasConf := obj["confidence"]
	require.False(t, hasConf)
	require.Equal(t, false, obj["calibrated"])
}

func TestParseOpenJevAnswersUnparsed(t *testing.T) {
	qs := map[string]api.DecisionQuestion{"q": {Type: "noul"}}
	out, err := ParseOpenJevAnswers("not json at all", qs)
	require.NoError(t, err)
	var obj map[string]any
	require.NoError(t, json.Unmarshal(out["q"], &obj))
	require.Equal(t, "unparsed_v0_readout", obj["note"])
}

func TestMergeOpenJevGatherNoulBias(t *testing.T) {
	qs := map[string]api.DecisionQuestion{"urgent": {Type: "noul"}}
	answers := map[string]json.RawMessage{"urgent": json.RawMessage(`{"type":"noul"}`)}
	hits := []OpenJevGatherHit{
		{QuestionID: "urgent", OptionID: "0", LogitMean: 1.0},
		{QuestionID: "urgent", OptionID: "1", LogitMean: 3.0},
	}
	out, err := MergeOpenJevGather(answers, qs, hits, "answer_marker_logit", 1.0, false, 5.0, true)
	require.NoError(t, err)
	var obj map[string]any
	require.NoError(t, json.Unmarshal(out["urgent"], &obj))
	require.Less(t, obj["noul"].(float64), 0.5)
	require.Equal(t, true, obj["calibrated"])
	require.Equal(t, 5.0, obj["noul_bias"])
}

func TestFitOpenJevNoulBiasPrefersGold(t *testing.T) {
	cases := []OpenJevNoulCase{
		{LogitFalse: 0.5, LogitTrue: 3.0, GoldTrue: true},
		{LogitFalse: 0.5, LogitTrue: 3.0, GoldTrue: false}, // needs large bias
	}
	b, nll := FitOpenJevNoulBias(cases, 1.0)
	require.Greater(t, b, 0.0)
	require.False(t, math.IsInf(nll, 0))
}
