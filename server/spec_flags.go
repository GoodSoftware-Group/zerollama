package server

import "github.com/ollama/ollama/llm"

func applySpecFlags(dst *llm.CompletionRequest, opts map[string]any, pld, mtp, drafter *bool) {
	if dst == nil {
		return
	}
	if pld != nil {
		dst.EnablePLD = pld
	} else if v, ok := boolFromMap(opts, "enable_pld"); ok {
		dst.EnablePLD = &v
	}
	if mtp != nil {
		dst.EnableMTP = mtp
	} else if drafter != nil {
		dst.EnableMTP = drafter
	} else if v, ok := boolFromMap(opts, "enable_mtp"); ok {
		dst.EnableMTP = &v
	} else if v, ok := boolFromMap(opts, "enable_drafter"); ok {
		dst.EnableMTP = &v
	}
}
