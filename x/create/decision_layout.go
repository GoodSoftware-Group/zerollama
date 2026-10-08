package create

import (
	"os"
	"path/filepath"
)

// hasDecisionHeadLayout reports Nimble/Tev1-style HF trees: Qwen3.5 backbone plus
// a separate decision head config alongside config.json.
func hasDecisionHeadLayout(modelDir string) bool {
	for _, name := range []string{
		"decision_head_config.json",
		"tev1_config.json",
		"nimble_config.json",
		"decision_config.json",
	} {
		if _, err := os.Stat(filepath.Join(modelDir, name)); err == nil {
			return true
		}
	}
	return false
}
