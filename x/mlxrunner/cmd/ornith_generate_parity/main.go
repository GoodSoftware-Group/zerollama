// F0684 — mlxrunner greedy generate rematch vs F0626 tokens_ref + dump prompt ids.
//
//	cd zerollama && go run ./x/mlxrunner/cmd/ornith_generate_parity
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	qwen35 "github.com/ollama/ollama/x/models/qwen3_5"
)

func main() {
	refPath := os.Getenv("ORNITH_OPTIQ_TOKENS_REF")
	if refPath == "" {
		refPath = "/tmp/uma_mlxrunner_optiq_tokens_ref.json"
	}
	outDir := os.Getenv("ORNITH_OPTIQ_GENERATE_DIR")
	if outDir == "" {
		outDir = "/tmp/uma_optiq_generate_dump"
	}
	npred := 4
	if s := os.Getenv("ORNITH_OPTIQ_NPRED"); s != "" {
		fmt.Sscanf(s, "%d", &npred)
	}

	refRaw, err := os.ReadFile(refPath)
	if err != nil {
		fatal(fmt.Errorf("read tokens_ref %s: %w (run m26 freeze first)", refPath, err))
	}
	var ref struct {
		Model            string  `json:"model"`
		Prompt           string  `json:"prompt"`
		Tokens           []int32 `json:"tokens"`
		PromptEvalCount  int     `json:"prompt_eval_count"`
		EvalCount        int     `json:"eval_count"`
	}
	if err := json.Unmarshal(refRaw, &ref); err != nil {
		fatal(err)
	}
	if len(ref.Tokens) < 2 {
		fatal(fmt.Errorf("tokens_ref too short: n=%d", len(ref.Tokens)))
	}
	// Prefer serve prompt_eval_count (F0686). Fallback: legacy fixed npred tail split.
	promptLen := ref.PromptEvalCount
	if promptLen <= 0 || promptLen >= len(ref.Tokens) {
		if len(ref.Tokens) <= npred {
			fatal(fmt.Errorf("tokens_ref too short: n=%d npred=%d", len(ref.Tokens), npred))
		}
		promptLen = len(ref.Tokens) - npred
	}
	promptIDs := append([]int32(nil), ref.Tokens[:promptLen]...)
	wantGen := append([]int32(nil), ref.Tokens[promptLen:]...)
	npred = len(wantGen)
	if npred < 1 {
		fatal(fmt.Errorf("tokens_ref has empty generate suffix (prompt_eval_count=%d n=%d)", promptLen, len(ref.Tokens)))
	}

	_ = os.RemoveAll(outDir)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fatal(err)
	}

	ctx := context.Background()
	th, err := mlxthread.Start("ornith-gen", func() error {
		mlx.EnableCompile()
		return nil
	})
	if err != nil {
		fatal(err)
	}
	defer th.Stop(ctx, nil)

	var r mlxrunner.Runner
	modelName := ref.Model
	if modelName == "" {
		modelName = "ornith-9b-optiq"
	}
	if err := th.Do(ctx, func() error { return r.Load(modelName) }); err != nil {
		fatal(err)
	}

	var got []int32
	err = th.Do(ctx, func() error {
		m := r.Model.(*qwen35.Model)
		caches := m.NewCaches()
		// Prefill
		hidden := m.Forward(&batch.Batch{
			InputIDs:     mlx.FromValues(promptIDs, 1, len(promptIDs)),
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{int32(len(promptIDs))},
		}, caches)
		hf32 := hidden.AsType(mlx.DTypeFloat32)
		logits := m.Unembed(hidden).AsType(mlx.DTypeFloat32)
		mlx.Eval(hf32, logits)
		mlx.Synchronize()
		hf := hf32.Floats()
		// last token hidden if 3D
		if dims := hf32.Dims(); len(dims) == 3 {
			tlen, D := dims[1], dims[2]
			hf = hf[(tlen-1)*D : tlen*D]
		}
		if err := writeF32(filepath.Join(outDir, "prefill_hidden.bin"), hf); err != nil {
			return err
		}
		got = append([]int32(nil), promptIDs...)
		pos := len(promptIDs)
		for step := 0; step < npred; step++ {
			lf := logits.Floats()
			dims := logits.Dims()
			if len(dims) == 3 {
				tlen, V := dims[1], dims[2]
				lf = lf[(tlen-1)*V : tlen*V]
			}
			tok := argmax(lf)
			got = append(got, tok)
			if step+1 == npred {
				break
			}
			hidden = m.Forward(&batch.Batch{
				InputIDs:     mlx.FromValues([]int32{tok}, 1, 1),
				SeqOffsets:   []int32{int32(pos)},
				SeqQueryLens: []int32{1},
			}, caches)
			pos++
			logits = m.Unembed(hidden).AsType(mlx.DTypeFloat32)
			mlx.Eval(logits)
			mlx.Synchronize()
		}
		return nil
	})
	if err != nil {
		fatal(err)
	}

	meta := map[string]any{
		"model":       modelName,
		"prompt":      ref.Prompt,
		"npred":       npred,
		"prompt_ids":  promptIDs,
		"want_tokens": ref.Tokens,
		"got_tokens":  got,
		"want_gen":    wantGen,
		"got_gen":     got[len(promptIDs):],
	}
	b, _ := json.MarshalIndent(meta, "", "  ")
	_ = os.WriteFile(filepath.Join(outDir, "meta.json"), b, 0o644)
	pb, _ := json.MarshalIndent(promptIDs, "", "  ")
	_ = os.WriteFile(filepath.Join(outDir, "prompt_ids.json"), pb, 0o644)

	if len(got) != len(ref.Tokens) {
		fatal(fmt.Errorf("len got=%d want=%d", len(got), len(ref.Tokens)))
	}
	for i := range got {
		if got[i] != ref.Tokens[i] {
			fatal(fmt.Errorf("mismatch @%d got=%d want=%d (dump %s)", i, got[i], ref.Tokens[i], outDir))
		}
	}
	meta["full_tokens_match"] = true
	b, _ = json.MarshalIndent(meta, "", "  ")
	_ = os.WriteFile(filepath.Join(outDir, "meta.json"), b, 0o644)
	fmt.Printf("PASS: generate parity vs tokens_ref n=%d npred=%d → %s\n", len(got), npred, outDir)
}

func argmax(xs []float32) int32 {
	best := 0
	bestv := float32(-math.MaxFloat32)
	for i, v := range xs {
		if v > bestv {
			bestv = v
			best = i
		}
	}
	return int32(best)
}

func writeF32(path string, xs []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return binary.Write(f, binary.LittleEndian, xs)
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "FAIL: %v\n", err)
	os.Exit(1)
}
