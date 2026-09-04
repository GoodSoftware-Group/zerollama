package storaged

import (
	"fmt"
	"strconv"
	"strings"
)

// contentRange is "bytes <start>-<end>/<total>" (RFC 9110 §14.4).
type contentRange struct {
	start int64
	end   int64
	total int64
}

func parseContentRange(h string) (*contentRange, error) {
	h = strings.TrimSpace(h)
	if h == "" {
		return nil, nil
	}
	const prefix = "bytes "
	if !strings.HasPrefix(strings.ToLower(h), prefix) {
		return nil, fmt.Errorf("invalid Content-Range: missing %q prefix", strings.TrimSpace(prefix))
	}
	spec := strings.TrimSpace(h[len(prefix):])
	slash := strings.IndexByte(spec, '/')
	if slash < 0 {
		return nil, fmt.Errorf("invalid Content-Range: missing /total")
	}
	rangePart, totalPart := spec[:slash], spec[slash+1:]
	dash := strings.IndexByte(rangePart, '-')
	if dash < 0 {
		return nil, fmt.Errorf("invalid Content-Range: missing - separator")
	}
	start, err := strconv.ParseInt(strings.TrimSpace(rangePart[:dash]), 10, 64)
	if err != nil {
		return nil, fmt.Errorf("invalid Content-Range start: %w", err)
	}
	endStr := strings.TrimSpace(rangePart[dash+1:])
	end, err := strconv.ParseInt(endStr, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("invalid Content-Range end: %w", err)
	}
	total, err := strconv.ParseInt(strings.TrimSpace(totalPart), 10, 64)
	if err != nil {
		return nil, fmt.Errorf("invalid Content-Range total: %w", err)
	}
	if start < 0 || end < start || total <= 0 || end >= total {
		return nil, fmt.Errorf("invalid Content-Range range: %d-%d/%d", start, end, total)
	}
	return &contentRange{start: start, end: end, total: total}, nil
}
