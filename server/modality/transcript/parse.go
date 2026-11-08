// Package transcript reads llama-server's transcript.text.done body.
package transcript

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// ErrEmpty is returned when the body has no transcript text.
var ErrEmpty = errors.New("empty transcript")

// Usage is the token accounting on transcript.text.done.
// CachedTokens is set when input_tokens_details.cached_tokens is present.
type Usage struct {
	Type         string
	InputTokens  int
	OutputTokens int
	TotalTokens  int
	CachedTokens *int
}

// ParseLlama reads text and usage. A missing usage object returns a nil Usage.
func ParseLlama(body []byte) (string, *Usage, error) {
	var probe struct {
		Text  string `json:"text"`
		Usage struct {
			Type               string `json:"type"`
			InputTokens        int    `json:"input_tokens"`
			OutputTokens       int    `json:"output_tokens"`
			TotalTokens        int    `json:"total_tokens"`
			InputTokensDetails struct {
				CachedTokens *int `json:"cached_tokens"`
			} `json:"input_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &probe); err != nil {
		return "", nil, err
	}
	text := strings.TrimSpace(probe.Text)
	if text == "" {
		return "", nil, ErrEmpty
	}
	u := probe.Usage
	if u.Type == "" && u.InputTokens == 0 && u.OutputTokens == 0 && u.TotalTokens == 0 {
		return text, nil, nil
	}
	return text, &Usage{
		Type:         u.Type,
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
		TotalTokens:  u.TotalTokens,
		CachedTokens: u.InputTokensDetails.CachedTokens,
	}, nil
}

// NameEvents copies a llama-server transcription SSE body and writes an
// event line from each JSON object's type. data: [DONE] is copied as-is.
// A data line that already follows an event line is not named again.
// seconds, when positive, is written onto usage in transcript.text.done.
func NameEvents(dst io.Writer, src io.Reader, seconds float64) error {
	sc := bufio.NewScanner(src)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	named := false
	for sc.Scan() {
		line := sc.Text()
		out := line
		if strings.HasPrefix(line, "event:") {
			named = true
		} else if line == "" {
			named = false
		} else if strings.HasPrefix(line, "data:") {
			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if payload != "" && payload != "[DONE]" {
				var ev struct {
					Type string `json:"type"`
				}
				if json.Unmarshal([]byte(payload), &ev) == nil && ev.Type != "" {
					if !named {
						if _, err := fmt.Fprintf(dst, "event: %s\n", ev.Type); err != nil {
							return err
						}
					}
					if seconds > 0 && ev.Type == "transcript.text.done" {
						if patched, err := addUsageSeconds([]byte(payload), seconds); err == nil {
							out = "data: " + string(patched)
						}
					}
				}
			}
		}
		if _, err := fmt.Fprintf(dst, "%s\n", out); err != nil {
			return err
		}
	}
	return sc.Err()
}

func addUsageSeconds(payload []byte, seconds float64) ([]byte, error) {
	var ev map[string]any
	if err := json.Unmarshal(payload, &ev); err != nil {
		return nil, err
	}
	usage, _ := ev["usage"].(map[string]any)
	if usage == nil {
		usage = map[string]any{"type": "duration"}
		ev["usage"] = usage
	}
	usage["seconds"] = seconds
	return json.Marshal(ev)
}

// MaxTokens reads the multipart max_tokens field. Empty means no cap.
// A set value is an integer from 1 through 1000000.
func MaxTokens(s string) (*int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 1 || n > 1_000_000 {
		return nil, fmt.Errorf("max_tokens must be a positive integer")
	}
	return &n, nil
}

// StreamFlag reads the multipart stream field. Empty is false.
func StreamFlag(s string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "false", "0":
		return false, nil
	case "true", "1":
		return true, nil
	default:
		return false, fmt.Errorf("stream must be true or false")
	}
}
