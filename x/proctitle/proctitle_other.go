//go:build !darwin && !linux

package proctitle

func setOS(name string) {
	// Windows and other platforms: os.Args[0] only (set by Set).
	_ = name
}
