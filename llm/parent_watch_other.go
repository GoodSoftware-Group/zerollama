//go:build !linux

package llm

import "os/exec"

// Darwin/Windows have no Pdeathsig. SIGKILL of the parent orphans llama-server.
func applyParentDeath(_ *exec.Cmd) {}
