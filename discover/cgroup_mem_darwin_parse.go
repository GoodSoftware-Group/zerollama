package discover

import (
	"regexp"
	"strconv"
	"strings"
)

var darwinSwapRe = regexp.MustCompile(`(?i)total\s*=\s*([\d.]+)([KMGT]?)i?B?.*?used\s*=\s*([\d.]+)([KMGT]?)i?B?`)

func parseDarwinSwapUsage(text string) (total, used uint64) {
	m := darwinSwapRe.FindStringSubmatch(text)
	if m == nil {
		return 0, 0
	}
	return darwinSize(m[1], m[2]), darwinSize(m[3], m[4])
}

func darwinSize(num, unit string) uint64 {
	f, err := strconv.ParseFloat(num, 64)
	if err != nil {
		return 0
	}
	mult := uint64(1)
	switch strings.ToUpper(unit) {
	case "K":
		mult = 1024
	case "M":
		mult = 1024 * 1024
	case "G":
		mult = 1024 * 1024 * 1024
	case "T":
		mult = 1024 * 1024 * 1024 * 1024
	}
	return uint64(f * float64(mult))
}
