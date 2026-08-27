package mlxrunner

import (
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode"

	"github.com/ollama/ollama/envconfig"
)

// mlx-serve round-cost tables: speculation width learned on this Mac, per
// model, survives process restart. No flag — next Load starts tuned.

type roundCostFile struct {
	Scheduled     int                         `json:"scheduled"`
	ProbeInterval int                         `json:"probe_interval"`
	Cost          map[string]float64          `json:"cost"`
	AccRate       []float64                   `json:"acc_rate"`
	AccSeen       []int                       `json:"acc_seen"`
	CtxBucket     int                         `json:"ctx_bucket,omitempty"`
	ByCtx         map[string]roundCostFile    `json:"by_ctx,omitempty"`
}

func roundCostDir() string {
	return filepath.Join(filepath.Dir(envconfig.Models()), "mlx-round-cost")
}

func roundCostPath(modelName string) string {
	name := strings.TrimSpace(modelName)
	if name == "" {
		return ""
	}
	var b strings.Builder
	for _, r := range name {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '.' || r == '-' || r == '_' {
			b.WriteRune(r)
			continue
		}
		b.WriteByte('_')
	}
	if b.Len() == 0 {
		return ""
	}
	return filepath.Join(roundCostDir(), b.String()+".json")
}

// ctxBucket is mlx-serve's per-context-size table key: short chats and
// 32k prompts do not share a cost curve.
func ctxBucket(promptTokens int) int {
	switch {
	case promptTokens <= 2048:
		return 2048
	case promptTokens <= 4096:
		return 4096
	case promptTokens <= 8192:
		return 8192
	case promptTokens <= 16384:
		return 16384
	default:
		return 32768
	}
}

func (c *depthController) snapshot() roundCostFile {
	f := roundCostFile{
		Scheduled:     c.scheduled,
		ProbeInterval: c.probeInterval,
		CtxBucket:     c.ctxBucket,
		Cost:          map[string]float64{},
	}
	if c.cost != nil {
		for n, ms := range c.cost.ewma {
			f.Cost[strconv.Itoa(n)] = ms
		}
	}
	if c.acc != nil {
		f.AccRate = append([]float64(nil), c.acc.rate...)
		f.AccSeen = append([]int(nil), c.acc.seen...)
	}
	return f
}

func (c *depthController) applySnapshot(f roundCostFile) {
	if f.Scheduled > 0 {
		c.scheduled = f.Scheduled
	}
	if f.ProbeInterval > 0 {
		c.probeInterval = f.ProbeInterval
	}
	if len(f.Cost) > 0 {
		c.cost = newCostModel()
		for k, ms := range f.Cost {
			n, err := strconv.Atoi(k)
			if err != nil || n < 0 || ms <= 0 {
				continue
			}
			c.cost.ewma[n] = ms
			c.cost.depths = appendUniqueSorted(c.cost.depths, n)
		}
	}
	if len(f.AccSeen) > 1 && len(f.AccRate) == len(f.AccSeen) {
		c.acc = newAcceptanceModel()
		c.acc.rate = append([]float64(nil), f.AccRate...)
		c.acc.seen = append([]int(nil), f.AccSeen...)
	}
}

func (c *depthController) applyCtxBucket(promptTokens int) {
	if c == nil {
		return
	}
	b := ctxBucket(promptTokens)
	if c.byCtx == nil {
		c.byCtx = map[int]roundCostFile{}
	}
	if c.ctxBucket == b {
		return
	}
	if c.ctxBucket == 0 {
		c.ctxBucket = b
		if c.cost != nil && c.cost.ready() {
			c.byCtx[b] = c.snapshot()
		}
		return
	}
	if c.ctxBucket != 0 && c.cost != nil && c.cost.ready() {
		c.byCtx[c.ctxBucket] = c.snapshot()
	}
	c.ctxBucket = b
	if f, ok := c.byCtx[b]; ok {
		c.applySnapshot(f)
		return
	}
	c.cost = newCostModel()
	c.acc = newAcceptanceModel()
}

func (c *depthController) restoreRoundCost(modelName string) {
	path := roundCostPath(modelName)
	if c == nil || path == "" {
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var f roundCostFile
	if err := json.Unmarshal(data, &f); err != nil {
		slog.Warn("mlx round-cost table unreadable", "path", path, "error", err)
		return
	}
	c.byCtx = map[int]roundCostFile{}
	for k, b := range f.ByCtx {
		n, err := strconv.Atoi(k)
		if err != nil || n <= 0 {
			continue
		}
		b.ByCtx = nil
		c.byCtx[n] = b
	}
	f.ByCtx = nil
	c.applySnapshot(f)
	if f.CtxBucket > 0 {
		c.ctxBucket = f.CtxBucket
		c.byCtx[f.CtxBucket] = f
	}
	slog.Debug("mlx round-cost restored", "model", modelName, "scheduled", c.scheduled, "ctx_bucket", c.ctxBucket, "buckets", len(c.byCtx))
}

func (c *depthController) saveRoundCost(modelName string) {
	path := roundCostPath(modelName)
	if c == nil || path == "" || c.cost == nil || !c.cost.ready() {
		return
	}
	live := c.snapshot()
	if c.ctxBucket > 0 {
		if c.byCtx == nil {
			c.byCtx = map[int]roundCostFile{}
		}
		c.byCtx[c.ctxBucket] = live
	}
	f := live
	if len(c.byCtx) > 0 {
		f.ByCtx = make(map[string]roundCostFile, len(c.byCtx))
		for n, b := range c.byCtx {
			b.ByCtx = nil
			f.ByCtx[strconv.Itoa(n)] = b
		}
	}
	data, err := json.Marshal(f)
	if err != nil {
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		slog.Debug("mlx round-cost mkdir", "error", err)
		return
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
	}
}

func appendUniqueSorted(ds []int, n int) []int {
	for _, v := range ds {
		if v == n {
			return ds
		}
	}
	ds = append(ds, n)
	for i := len(ds) - 1; i > 0 && ds[i] < ds[i-1]; i-- {
		ds[i], ds[i-1] = ds[i-1], ds[i]
	}
	return ds
}
