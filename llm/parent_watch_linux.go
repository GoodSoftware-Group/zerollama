package llm

import (
	"os/exec"
	"syscall"

	"github.com/ollama/ollama/envconfig"
)

func applyParentDeath(cmd *exec.Cmd) {
	if !envconfig.BackendParentWatch() {
		return
	}
	base := syscall.SysProcAttr{}
	if cmd.SysProcAttr != nil {
		base = *cmd.SysProcAttr
	}
	base.Pdeathsig = syscall.SIGKILL
	cmd.SysProcAttr = &base
}
