package server

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/internal/modelhealth"
)

// enrichModelLoadError attaches modelhealth fix hints to common load failures.
func enrichModelLoadError(name string, err error) error {
	if err == nil {
		return nil
	}
	msg := strings.ToLower(err.Error())
	if !strings.Contains(msg, "no such file") &&
		!strings.Contains(msg, "nil pointer") &&
		!strings.Contains(msg, "failed to read config.json") {
		return err
	}
	report, rerr := modelhealth.CheckName(name)
	if rerr != nil || report.Status == modelhealth.StatusOK {
		return err
	}
	if report.FixHint != "" {
		return fmt.Errorf("%w — %s", err, report.FixHint)
	}
	return fmt.Errorf("%w — %s", err, report.Detail)
}
