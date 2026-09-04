package freetokenlab

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// AnemllRecord is one line of anemll-flash-llama.cpp --moe-trace JSONL
// (llama-context.cpp write_trace).
type AnemllRecord struct {
	Seq         uint64 `json:"seq"`
	Layer       int    `json:"layer"`
	NExpertUsed int    `json:"n_expert_used"`
	NTokens     int    `json:"n_tokens"`
	Experts     []int  `json:"experts"`
}

// LoadedTrace is prefill vs decode steps plus inferred expert pool size.
type LoadedTrace struct {
	Prefill  []TraceStep
	Decode   []TraceStep
	NExperts int
	Records  int
}

// LoadAnemllJSONL parses a Flash-MoE oracle/capture file.
// n_tokens > 1 is treated as prefill; n_tokens == 1 as decode.
func LoadAnemllJSONL(r io.Reader) (LoadedTrace, error) {
	var out LoadedTrace
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16<<20)
	lineNo := 0
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		lineNo++
		var rec AnemllRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			return out, fmt.Errorf("line %d: %w", lineNo, err)
		}
		step := TraceStep{Layer: rec.Layer, Experts: rec.Experts}
		if rec.NTokens > 1 {
			out.Prefill = append(out.Prefill, step)
		} else {
			out.Decode = append(out.Decode, step)
		}
		for _, e := range rec.Experts {
			if e+1 > out.NExperts {
				out.NExperts = e + 1
			}
		}
		out.Records++
	}
	if err := sc.Err(); err != nil {
		return out, err
	}
	return out, nil
}

// LoadAnemllFile opens path and calls LoadAnemllJSONL.
func LoadAnemllFile(path string) (LoadedTrace, error) {
	f, err := os.Open(path)
	if err != nil {
		return LoadedTrace{}, err
	}
	defer f.Close()
	return LoadAnemllJSONL(f)
}
