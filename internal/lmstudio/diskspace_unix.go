//go:build !windows

// Platform-specific free-space lookup for LM Studio import disk checks.
// Why a separate file: syscall.Statfs is unavailable on Windows; see diskspace_windows.go.
package lmstudio

import (
	"fmt"
	"syscall"
)

func freeBytesOS(path string) (int64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, fmt.Errorf("statfs %s: %w", path, err)
	}
	return int64(st.Bavail) * int64(st.Bsize), nil
}
