//go:build !cgo

package pyembed

import "errors"

func EmbedStart(repoRoot, runtimeParent string, port int) error {
	return errors.New("embedded runtime requires CGO and libpython (python3-embed)")
}

func IsStarted() bool { return false }
