package server

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/ollama/ollama/api"
)

func copyRuntimeV1ChatBody(w http.ResponseWriter, resp *http.Response, meta *api.ChatCompressionMeta, includeUsage bool) error {
	if meta == nil {
		return copyRuntimeResponseBody(w, resp.Body)
	}
	ct := resp.Header.Get("Content-Type")
	if strings.Contains(strings.ToLower(ct), "text/event-stream") {
		return copyRuntimeV1SSEWithCompression(w, resp.Body, meta, includeUsage)
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	_, err = w.Write(attachCompressionMetaToV1JSON(raw, meta))
	return err
}

func attachCompressionMetaToV1JSON(raw []byte, meta *api.ChatCompressionMeta) []byte {
	if meta == nil {
		return raw
	}
	trim := bytes.TrimSpace(raw)
	if len(trim) == 0 || trim[0] != '{' {
		return raw
	}
	var m map[string]any
	if json.Unmarshal(trim, &m) != nil {
		return raw
	}
	usage, _ := m["usage"].(map[string]any)
	if usage == nil {
		usage = map[string]any{}
		m["usage"] = usage
	}
	usage["compression_meta"] = meta
	out, err := json.Marshal(m)
	if err != nil {
		return raw
	}
	return out
}

func copyRuntimeV1SSEWithCompression(w http.ResponseWriter, r io.Reader, meta *api.ChatCompressionMeta, includeUsage bool) error {
	flusher, canFlush := w.(http.Flusher)
	br := bufio.NewReader(r)
	sawUsage := false
	var event bytes.Buffer
	flushEvent := func() error {
		if event.Len() == 0 {
			return nil
		}
		rewritten, hadUsage := rewriteSSEEvent(event.Bytes(), meta)
		if hadUsage {
			sawUsage = true
		}
		if _, err := w.Write(rewritten); err != nil {
			return err
		}
		if canFlush {
			flusher.Flush()
		}
		event.Reset()
		return nil
	}
	for {
		line, err := br.ReadBytes('\n')
		if len(line) > 0 {
			event.Write(line)
			if bytes.HasSuffix(event.Bytes(), []byte("\n\n")) {
				payload := event.Bytes()
				if includeUsage && !sawUsage && isSSEDone(payload) {
					usageChunk := append([]byte("data: "), mustMarshalV1UsageChunk(meta)...)
					usageChunk = append(usageChunk, '\n', '\n')
					if _, ew := w.Write(usageChunk); ew != nil {
						return ew
					}
					if canFlush {
						flusher.Flush()
					}
					sawUsage = true
				}
				if err := flushEvent(); err != nil {
					return err
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				if event.Len() > 0 {
					return flushEvent()
				}
				return nil
			}
			return err
		}
	}
}

func isSSEDone(event []byte) bool {
	for _, line := range bytes.Split(event, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if bytes.Equal(line, []byte("data: [DONE]")) || bytes.Equal(line, []byte("data:[DONE]")) {
			return true
		}
	}
	return false
}

func rewriteSSEEvent(event []byte, meta *api.ChatCompressionMeta) ([]byte, bool) {
	hadUsage := false
	var b strings.Builder
	start := 0
	for start < len(event) {
		nl := bytes.IndexByte(event[start:], '\n')
		var line []byte
		hasNL := false
		if nl < 0 {
			line = event[start:]
			start = len(event)
		} else {
			line = event[start : start+nl]
			hasNL = true
			start += nl + 1
		}
		if bytes.HasPrefix(line, []byte("data:")) {
			payload := bytes.TrimSpace(line[len("data:"):])
			if len(payload) > 0 && payload[0] == '{' {
				rewritten := attachCompressionMetaToV1JSON(payload, meta)
				var probe map[string]any
				if json.Unmarshal(rewritten, &probe) == nil {
					if _, ok := probe["usage"]; ok {
						hadUsage = true
					}
				}
				b.WriteString("data: ")
				b.Write(rewritten)
			} else {
				b.Write(line)
			}
		} else {
			b.Write(line)
		}
		if hasNL {
			b.WriteByte('\n')
		}
	}
	return []byte(b.String()), hadUsage
}

func mustMarshalV1UsageChunk(meta *api.ChatCompressionMeta) []byte {
	chunk := map[string]any{
		"object":  "chat.completion.chunk",
		"choices": []any{},
		"usage": map[string]any{
			"compression_meta": meta,
		},
	}
	b, err := json.Marshal(chunk)
	if err != nil {
		return []byte("{}")
	}
	return b
}
