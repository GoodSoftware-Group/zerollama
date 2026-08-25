package llm

import "os/exec"

// ApplyParentDeath requests the kernel kill this subprocess if the zerollama
// parent dies (LA20). Linux uses PR_SET_PDEATHSIG; other OSes are a no-op.
func ApplyParentDeath(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	applyParentDeath(cmd)
}
