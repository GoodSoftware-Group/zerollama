package envconfig

import (
	"fmt"
	"log/slog"
	"math"
	"strconv"
	"strings"
)

// VRAMBudget is an optional cap on allocatable VRAM (LocalAI LA19).
// Zero value means no cap. Percentage form uses fraction (0,1]; absolute uses bytes.
type VRAMBudget struct {
	fraction float64
	absolute uint64
}

var sizeSuffixes = []struct {
	suffix string
	mult   uint64
}{
	{"KIB", 1 << 10}, {"MIB", 1 << 20}, {"GIB", 1 << 30}, {"TIB", 1 << 40},
	{"KB", 1000}, {"MB", 1000 * 1000}, {"GB", 1000 * 1000 * 1000}, {"TB", 1000 * 1000 * 1000 * 1000},
	{"B", 1},
}

// ParseVRAMBudget accepts "", "80%", "0.8", "12GB", "12GiB", or raw bytes.
func ParseVRAMBudget(s string) (VRAMBudget, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return VRAMBudget{}, nil
	}
	upper := strings.ToUpper(s)
	if strings.HasSuffix(upper, "%") {
		num := strings.TrimSpace(strings.TrimSuffix(upper, "%"))
		v, err := strconv.ParseFloat(num, 64)
		if err != nil {
			return VRAMBudget{}, fmt.Errorf("invalid vram budget percentage %q: %w", s, err)
		}
		return vramBudgetFromFraction(v/100.0, s)
	}
	for _, sfx := range sizeSuffixes {
		if strings.HasSuffix(upper, sfx.suffix) {
			num := strings.TrimSpace(strings.TrimSuffix(upper, sfx.suffix))
			v, err := strconv.ParseFloat(num, 64)
			if err != nil || v < 0 {
				return VRAMBudget{}, fmt.Errorf("invalid vram budget %q", s)
			}
			return VRAMBudget{absolute: uint64(v * float64(sfx.mult))}, nil
		}
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return VRAMBudget{}, fmt.Errorf("invalid vram budget %q", s)
	}
	if v < 0 {
		return VRAMBudget{}, fmt.Errorf("invalid vram budget %q: negative", s)
	}
	if v > 0 && v <= 1 {
		return vramBudgetFromFraction(v, s)
	}
	if v != math.Trunc(v) {
		return VRAMBudget{}, fmt.Errorf("invalid vram budget %q: out of range", s)
	}
	return VRAMBudget{absolute: uint64(v)}, nil
}

func vramBudgetFromFraction(f float64, orig string) (VRAMBudget, error) {
	if f <= 0 {
		return VRAMBudget{}, nil
	}
	if f > 1 {
		return VRAMBudget{}, fmt.Errorf("vram budget %q exceeds 100%%", orig)
	}
	return VRAMBudget{fraction: f}, nil
}

func (b VRAMBudget) IsSet() bool { return b.fraction > 0 || b.absolute > 0 }

func (b VRAMBudget) String() string {
	switch {
	case b.fraction > 0:
		return strconv.FormatFloat(b.fraction*100, 'f', -1, 64) + "%"
	case b.absolute == 0:
		return ""
	case b.absolute%(1<<30) == 0:
		return strconv.FormatUint(b.absolute>>30, 10) + "GiB"
	case b.absolute%(1000*1000*1000) == 0:
		return strconv.FormatUint(b.absolute/1_000_000_000, 10) + "GB"
	default:
		return strconv.FormatUint(b.absolute, 10) + "B"
	}
}

// Ceiling is the byte cap against detectedTotal, clamped so it cannot invent VRAM.
func (b VRAMBudget) Ceiling(detectedTotal uint64) uint64 {
	if !b.IsSet() || detectedTotal == 0 {
		return 0
	}
	var ceil uint64
	if b.fraction > 0 {
		ceil = uint64(float64(detectedTotal) * b.fraction)
	} else {
		ceil = b.absolute
	}
	if ceil > detectedTotal {
		ceil = detectedTotal
	}
	return ceil
}

// Apply caps detected total/free. Unchanged when unset or total is unknown.
func (b VRAMBudget) Apply(detectedTotal, detectedFree uint64) (total, free uint64) {
	if !b.IsSet() || detectedTotal == 0 {
		return detectedTotal, detectedFree
	}
	ceil := b.Ceiling(detectedTotal)
	total = min(detectedTotal, ceil)
	free = min(detectedFree, ceil)
	return total, free
}

// VRAMBudgetFromEnv reads ZEROLLAMA_VRAM_BUDGET. Invalid values are treated as unset.
func VRAMBudgetFromEnv() VRAMBudget {
	s := Var("ZEROLLAMA_VRAM_BUDGET")
	if s == "" {
		return VRAMBudget{}
	}
	b, err := ParseVRAMBudget(s)
	if err != nil {
		slog.Warn("invalid ZEROLLAMA_VRAM_BUDGET, ignored", "value", s, "error", err)
		return VRAMBudget{}
	}
	return b
}
