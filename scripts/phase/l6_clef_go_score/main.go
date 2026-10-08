// Lab smoke without CGO: build a Clef-shaped prompt, tokenize by segment,
// POST /embedding with score_fields, decode a noul answer.
//
//	CLEF_LIVE_PORT=18086 go run ./scripts/phase/l6_clef_go_score
//
// Full decision.Compile coverage stays in decision/*_test.go (no live server).
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strconv"
	"time"
)

type scoreField struct {
	Type     int      `json:"type"`
	Question [2]int   `json:"question"`
	Options  [][2]int `json:"options"`
}

func main() {
	portStr := os.Getenv("CLEF_LIVE_PORT")
	if portStr == "" {
		fatalf("set CLEF_LIVE_PORT")
	}
	port, err := strconv.Atoi(portStr)
	if err != nil || port < 1 {
		fatalf("invalid CLEF_LIVE_PORT=%q", portStr)
	}
	base := fmt.Sprintf("http://127.0.0.1:%d", port)
	client := &http.Client{Timeout: 2 * time.Minute}

	// Segment layout mirrors decision/clef.go encodeClef (noul field).
	segments := []string{
		"<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n",
		"Customer charged twice.",
		"\n\nSCHEMA FIELDS:\n",
		"\nFIELD 1\nID: refund\nTYPE: noul\nINSTRUCTION: ",
		"Refund requested?",
		"\nALLOWED OPTIONS:\n",
		"OPTION 1: ",
		`{"description":"The proposition is false or the answer is no.","option_id":"false"}`,
		"\n",
		"OPTION 2: ",
		`{"description":"The proposition is true or the answer is yes.","option_id":"true"}`,
		"\n",
		"END FIELD\n",
		"\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
	}
	// question = segments[4], options = [7] and [10]
	qSeg, optFalse, optTrue := 4, 7, 10

	var tokens []int
	offsets := []int{0}
	cache := map[string][]int{}
	for _, segment := range segments {
		ids, ok := cache[segment]
		if !ok {
			ids, err = tokenize(client, base, segment)
			if err != nil {
				fatalf("tokenize: %v", err)
			}
			cache[segment] = ids
		}
		tokens = append(tokens, ids...)
		offsets = append(offsets, len(tokens))
	}
	if len(tokens) == 0 || len(tokens) > 512 {
		fatalf("token count %d out of range", len(tokens))
	}

	fields := []scoreField{{
		Type:     0,
		Question: [2]int{offsets[qSeg], offsets[qSeg+1]},
		Options: [][2]int{
			{offsets[optFalse], offsets[optFalse+1]},
			{offsets[optTrue], offsets[optTrue+1]},
		},
	}}
	for _, span := range append([][2]int{fields[0].Question}, fields[0].Options...) {
		if span[0] >= span[1] {
			fatalf("empty token span %v (offsets=%v)", span, offsets)
		}
	}

	var result []struct {
		Logits [][]float32 `json:"logits"`
		Tokens int         `json:"tokens_evaluated"`
	}
	if err := postJSON(client, base+"/embedding", map[string]any{
		"input":        tokens,
		"score_fields": fields,
	}, &result); err != nil {
		fatalf("embedding: %v", err)
	}
	if len(result) != 1 || len(result[0].Logits) != 1 || len(result[0].Logits[0]) != 2 {
		fatalf("unexpected embedding response: %+v", result)
	}
	if result[0].Tokens != len(tokens) {
		fatalf("tokens_evaluated=%d want %d", result[0].Tokens, len(tokens))
	}

	logits := result[0].Logits[0]
	noul := softmaxNoul(logits[0], logits[1]) // P(true) with [false, true] order
	fmt.Printf("PASS: l6_clef_go_score tokens=%d logits=%v noul(true)=%.4f\n", result[0].Tokens, logits, noul)
}

func softmaxNoul(falseLogit, trueLogit float32) float64 {
	m := math.Max(float64(falseLogit), float64(trueLogit))
	ef := math.Exp(float64(falseLogit) - m)
	et := math.Exp(float64(trueLogit) - m)
	return et / (ef + et)
}

func tokenize(client *http.Client, base, content string) ([]int, error) {
	raw := json.RawMessage{}
	if err := postJSON(client, base+"/tokenize", map[string]any{
		"content":       content,
		"add_special":   false,
		"parse_special": true,
	}, &raw); err != nil {
		return nil, err
	}
	var wrapped struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.Unmarshal(raw, &wrapped); err == nil && wrapped.Tokens != nil {
		return wrapped.Tokens, nil
	}
	var arr []int
	if err := json.Unmarshal(raw, &arr); err != nil {
		return nil, fmt.Errorf("tokenize decode: %w body=%s", err, string(raw))
	}
	return arr, nil
}

func postJSON(client *http.Client, url string, in, out any) error {
	data, err := json.Marshal(in)
	if err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return err
	}
	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: HTTP %d: %s", url, res.StatusCode, string(body))
	}
	switch o := out.(type) {
	case *json.RawMessage:
		*o = append((*o)[:0], body...)
		return nil
	default:
		return json.Unmarshal(body, out)
	}
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}
