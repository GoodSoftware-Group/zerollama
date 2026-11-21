package weightedit

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"strings"
)

const hashChunk = 16 << 20

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func sha256File(path string) (string, error) {
	return sha256Range(path, 0, -1)
}

// sha256Range hashes [start, end). end < 0 means through EOF.
func sha256Range(path string, start, end int64) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if end < 0 {
		fi, err := f.Stat()
		if err != nil {
			return "", err
		}
		end = fi.Size()
	}
	if _, err := f.Seek(start, io.SeekStart); err != nil {
		return "", err
	}
	h := sha256.New()
	left := end - start
	buf := make([]byte, hashChunk)
	for left > 0 {
		n := hashChunk
		if int64(n) > left {
			n = int(left)
		}
		nr, err := io.ReadFull(f, buf[:n])
		if err != nil && err != io.ErrUnexpectedEOF && err != io.EOF {
			return "", err
		}
		if nr == 0 {
			return "", fmt.Errorf("%s: file shorter than declared range", path)
		}
		h.Write(buf[:nr])
		left -= int64(nr)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func digestLines(lines []string) string {
	return sha256Hex([]byte(strings.Join(lines, "\n")))
}

func shapeStr(shape []int64) string {
	parts := make([]string, len(shape))
	for i, d := range shape {
		parts[i] = fmt.Sprintf("%d", d)
	}
	return strings.Join(parts, ",")
}

func layoutDigestLines(infos []TensorInfo) string {
	lines := make([]string, 0, len(infos))
	for _, info := range infos {
		lines = append(lines, fmt.Sprintf("%s\t%s\t%s", info.Name, info.Dtype, shapeStr(info.Shape)))
	}
	return digestLines(lines)
}

func fingerprintLines(infos []TensorInfo, hashes map[string]string) string {
	lines := make([]string, 0, len(infos))
	for _, info := range infos {
		lines = append(lines, fmt.Sprintf("%s\t%s\t%s\t%s", info.Name, info.Dtype, shapeStr(info.Shape), hashes[info.Name]))
	}
	return digestLines(lines)
}
