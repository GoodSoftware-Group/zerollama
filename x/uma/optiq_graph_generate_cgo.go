//go:build darwin && uma && umaoptiq

package uma

/*
#cgo CFLAGS: -I${SRCDIR}/../../../bmtl/hardware_lab/lanes/m4/uma_toolkit/src
#cgo LDFLAGS: -L${SRCDIR}/../../../bmtl/hardware_lab/lanes/m4/uma_toolkit/src -luma_optiq_graph_gen -Wl,-rpath,${SRCDIR}/../../../bmtl/hardware_lab/lanes/m4/uma_toolkit/src
#include "uma_optiq_graph_gen.h"
*/
import "C"

import (
	"fmt"
	"log/slog"
)

func runOptiqGraphGenerateInProcess(prompt []int32, nGen int) ([]int32, error) {
	const cap = 16
	buf := make([]C.int32_t, cap)
	n := C.int(cap)
	var rc C.int
	if len(prompt) == 0 {
		rc = C.uma_optiq_graph_generate(&buf[0], &n)
	} else {
		cPrompt := make([]C.int32_t, len(prompt))
		for i, t := range prompt {
			cPrompt[i] = C.int32_t(t)
		}
		rc = C.uma_optiq_graph_generate_prompt(&cPrompt[0], C.int(len(cPrompt)), C.int(nGen), &buf[0], &n)
	}
	if rc == C.UMA_OPTIQ_GRAPH_GEN_SKIP {
		return nil, fmt.Errorf("optiq GRAPH generate soft-skip (missing dump/blobs)")
	}
	if rc != 0 {
		return nil, fmt.Errorf("uma_optiq_graph_generate(_prompt) rc=%d", int(rc))
	}
	if n <= 0 || int(n) > cap {
		return nil, fmt.Errorf("uma_optiq_graph_generate bad n=%d", int(n))
	}
	ids := make([]int32, int(n))
	for i := 0; i < int(n); i++ {
		ids[i] = int32(buf[i])
	}
	OptiqGraphGenerateLastMode = "in-process"
	slog.Info("optiq GRAPH generate in-process", "n", len(ids), "prompt_n", len(prompt), "n_gen", nGen, "dylib", resolveOptiqGraphGenerateDylib())
	return ids, nil
}
