//go:build linux

package llm

import (
	"os/exec"
	"syscall"
	"testing"
)

func TestApplyParentDeathSetsPdeathsig(t *testing.T) {
	t.Setenv("ZEROLLAMA_BACKEND_PARENT_WATCH", "")
	cmd := exec.Command("true")
	cmd.SysProcAttr = LlamaServerSysProcAttr
	ApplyParentDeath(cmd)
	if cmd.SysProcAttr == nil || cmd.SysProcAttr.Pdeathsig != syscall.SIGKILL {
		t.Fatalf("Pdeathsig=%v", cmd.SysProcAttr)
	}
}

func TestApplyParentDeathDisabled(t *testing.T) {
	t.Setenv("ZEROLLAMA_BACKEND_PARENT_WATCH", "0")
	cmd := exec.Command("true")
	ApplyParentDeath(cmd)
	if cmd.SysProcAttr != nil && cmd.SysProcAttr.Pdeathsig != 0 {
		t.Fatalf("disabled watch should not set Pdeathsig, got %#v", cmd.SysProcAttr)
	}
}
