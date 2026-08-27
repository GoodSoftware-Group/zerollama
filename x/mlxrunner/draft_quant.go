package mlxrunner

import (
	"os"
	"reflect"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/models/nn"
)

// Draft heads only propose tokens. mlx-serve 4-bit quantizes dense bf16 MTP /
// assistant weights at load (+10% decode on Qwen 3.8 27B at equal accept).
// ZEROLLAMA_MLX_DRAFT_QUANT=off keeps the checkpoint dtype.
func draftQuantEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_MLX_DRAFT_QUANT"))) {
	case "0", "off", "false", "no":
		return false
	default:
		return true
	}
}

func quantizeDraftCompanion(draft base.DraftModel) int {
	if draft == nil || !draftQuantEnabled() {
		return 0
	}
	return quantizeDraftValue(reflect.ValueOf(draft), map[uintptr]bool{})
}

func skipDraftQuantField(name string) bool {
	switch name {
	case "Centroids", "TokenOrdering":
		return true
	default:
		return false
	}
}

func quantizeDraftValue(v reflect.Value, seen map[uintptr]bool) int {
	if !v.IsValid() {
		return 0
	}
	n := 0
	switch v.Kind() {
	case reflect.Interface:
		if v.IsNil() {
			return 0
		}
		if v.CanSet() && v.Type().Implements(reflect.TypeOf((*nn.LinearLayer)(nil)).Elem()) {
			if layer, ok := v.Interface().(nn.LinearLayer); ok {
				q := nn.QuantizeLinearLayer(layer)
				if q != layer {
					v.Set(reflect.ValueOf(q))
					n++
				}
			}
		}
		return n + quantizeDraftValue(v.Elem(), seen)
	case reflect.Pointer:
		if v.IsNil() {
			return 0
		}
		ptr := v.Pointer()
		if seen[ptr] {
			return 0
		}
		seen[ptr] = true
		return quantizeDraftValue(v.Elem(), seen)
	case reflect.Struct:
		t := v.Type()
		for i := range v.NumField() {
			sf := t.Field(i)
			if !sf.IsExported() || skipDraftQuantField(sf.Name) {
				continue
			}
			n += quantizeDraftValue(v.Field(i), seen)
		}
	case reflect.Slice, reflect.Array:
		for i := range v.Len() {
			n += quantizeDraftValue(v.Index(i), seen)
		}
	}
	return n
}
