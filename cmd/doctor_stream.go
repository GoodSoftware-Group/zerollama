package cmd

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// doctorCheckStreamContent covers minefield trap 23: with thinking off, streamed
// answer tokens must arrive in content deltas — not only in reasoning/thinking.
func doctorCheckStreamContent(base string, m doctorLoadedModel) doctorCheck {
	const name = "serving trap-23 (stream content deltas)"

	payload := map[string]any{
		"model": m.Name,
		"messages": []map[string]string{
			{"role": "user", "content": "Capital of Norway? One word."},
		},
		"stream":      true,
		"max_tokens":  32,
		"temperature": 0,
	}
	if m.SupportsThinking {
		payload["think"] = false
		payload["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: err.Error()}
	}
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Post(base+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "stream failed: " + err.Error()}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
		return doctorCheck{
			Name:   name,
			Status: "warn",
			Detail: fmt.Sprintf("HTTP %d: %s", resp.StatusCode, truncateDoctorDetail(string(raw), 160)),
		}
	}

	counts, err := doctorCountStreamDeltaKeys(resp.Body)
	if err != nil {
		return doctorCheck{Name: name, Status: "warn", Detail: "parse stream: " + err.Error()}
	}

	switch {
	case counts["content"] > 0:
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("streamed answer in content deltas %v on %s", counts, m.Name),
		}
	case counts["reasoning"] > 0 || counts["reasoning_content"] > 0 || counts["thinking"] > 0:
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("stream deltas only in reasoning/thinking with think off %v (minefield trap 23)", counts),
			FixHint: "clients that concat content see blank replies; answer must stream in content",
		}
	default:
		return doctorCheck{
			Name:   name,
			Status: "warn",
			Detail: fmt.Sprintf("no non-empty stream deltas seen %v on %s", counts, m.Name),
		}
	}
}

// doctorCountStreamDeltaKeys tallies non-empty OpenAI chat SSE delta field names.
func doctorCountStreamDeltaKeys(r io.Reader) (map[string]int, error) {
	counts := map[string]int{}
	sc := bufio.NewScanner(r)
	// SSE lines can be long for tool calls; raise buffer.
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}
		var chunk map[string]any
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		choices, _ := chunk["choices"].([]any)
		if len(choices) == 0 {
			continue
		}
		ch0, _ := choices[0].(map[string]any)
		delta, _ := ch0["delta"].(map[string]any)
		for k, v := range delta {
			switch t := v.(type) {
			case string:
				if t != "" {
					counts[k]++
				}
			case nil:
			default:
				// tool_calls etc.
				counts[k]++
			}
		}
	}
	return counts, sc.Err()
}
