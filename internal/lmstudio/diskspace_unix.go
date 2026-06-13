//go:build !windows

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
