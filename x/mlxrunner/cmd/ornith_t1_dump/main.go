// F0683 — ornith-9b-optiq single-token Forward dump (TOK=1234).
//
//	cd zerollama && go run ./x/mlxrunner/cmd/ornith_t1_dump
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
	tok := int32(1234)
	if s := os.Getenv("ORNITH_OPTIQ_T1_TOK"); s != "" {
		var v int
		if _, err := fmt.Sscanf(s, "%d", &v); err == nil {
			tok = int32(v)
		}
	}
	outDir := os.Getenv("ORNITH_OPTIQ_T1_DIR")
	if outDir == "" {
		outDir = "/tmp/uma_optiq_t1_dump"
	}
	_ = os.RemoveAll(outDir)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fatal(err)
	}

	ctx := context.Background()
	th, err := mlxthread.Start("ornith-t1", func() error {
		mlx.EnableCompile()
		return nil
	})
	if err != nil {
		fatal(err)
	}
	defer th.Stop(ctx, nil)

	var r mlxrunner.Runner
	if err := th.Do(ctx, func() error { return r.Load("ornith-9b-optiq") }); err != nil {
		fatal(err)
	}

	var argmax int32
	err = th.Do(ctx, func() error {
		m := r.Model.(*qwen35.Model)
		hidden := m.Forward(&batch.Batch{
			InputIDs:     mlx.FromValues([]int32{tok}, 1, 1),
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{1},
		}, m.NewCaches())
		if hidden == nil || !hidden.Valid() {
			return fmt.Errorf("invalid hidden")
		}
		hf32 := hidden.AsType(mlx.DTypeFloat32)
		logits := m.Unembed(hidden).AsType(mlx.DTypeFloat32)
		mlx.Eval(hf32, logits)
		mlx.Synchronize()
		hf := hf32.Floats()
		lf := logits.Floats()
		dims := logits.Dims()
		if len(dims) == 3 {
			tlen, V := dims[1], dims[2]
			lf = lf[(tlen-1)*V : tlen*V]
		}
		best := 0
		bestv := float32(-math.MaxFloat32)
		for i, v := range lf {
			if v > bestv {
				bestv = v
				best = i
			}
		}
		argmax = int32(best)
		hp := filepath.Join(outDir, "hidden.bin")
		lp := filepath.Join(outDir, "logits.bin")
		if err := writeF32(hp, hf); err != nil {
			return err
		}
		if err := writeF32(lp, lf); err != nil {
			return err
		}
		meta := map[string]any{
			"model":       "ornith-9b-optiq",
			"tok":         tok,
			"argmax":      argmax,
			"D":           len(hf),
			"V":           len(lf),
			"hidden_path": hp,
			"logits_path": lp,
		}
		b, _ := json.MarshalIndent(meta, "", "  ")
		return os.WriteFile(filepath.Join(outDir, "meta.json"), b, 0o644)
	})
	if err != nil {
		fatal(err)
	}
	fmt.Printf("PASS: t1 dump tok=%d argmax=%d → %s\n", tok, argmax, outDir)
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
