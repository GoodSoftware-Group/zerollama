// Pirate Face catalog search (GET /s/QUERY): find GGUF torrents without a local daemon.
package server

import (
	"bufio"
	"context"
	"fmt"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
)

const (
	defaultPirateFaceAPI = "https://pirateface.co"
	pirateFaceSearchCap  = 20
)

// PirateFaceSearchRow is one catalog line from GET /s/QUERY.
type PirateFaceSearchRow struct {
	Repo   string
	Size   string
	Seeds  int
	Magnet string
}

var (
	pirateFaceQuantTag   = regexp.MustCompile(`(?i)(?:^|[-_./])(iq[1-4]_[a-z0-9]+|q[2-8](?:_[a-z0-9]+)+|f16|f32)(?:$|[-_./])`)
	pirateFaceSearchLine = regexp.MustCompile(`^(\S+)\s+(\S+(?:\s+\S+)?)\s+(\d+)\s+seeds\s+(magnet:\S+)\s*$`)
)

// PirateFaceAPIBase returns PF_API or the public Pirate Face origin.
func PirateFaceAPIBase() string {
	base := strings.TrimSpace(os.Getenv("PF_API"))
	if base == "" {
		base = defaultPirateFaceAPI
	}
	return strings.TrimRight(base, "/")
}

// LooksLikePirateFaceGGUF reports whether a repo name is likely a GGUF listing.
func LooksLikePirateFaceGGUF(repo string) bool {
	low := strings.ToLower(repo)
	if strings.Contains(low, "gguf") {
		return true
	}
	return pirateFaceQuantTag.MatchString(low)
}

// ParsePirateFaceSearchLine parses one GET /s/ body line.
func ParsePirateFaceSearchLine(line string) (PirateFaceSearchRow, bool) {
	line = strings.TrimSpace(line)
	if line == "" {
		return PirateFaceSearchRow{}, false
	}
	m := pirateFaceSearchLine.FindStringSubmatch(line)
	if m == nil {
		// Fallback: last token magnet, then "N seeds", then size tokens, then repo.
		parts := strings.Fields(line)
		if len(parts) < 5 {
			return PirateFaceSearchRow{}, false
		}
		magnet := parts[len(parts)-1]
		if !strings.HasPrefix(magnet, "magnet:") {
			return PirateFaceSearchRow{}, false
		}
		if parts[len(parts)-2] != "seeds" {
			return PirateFaceSearchRow{}, false
		}
		seeds, err := strconv.Atoi(parts[len(parts)-3])
		if err != nil {
			return PirateFaceSearchRow{}, false
		}
		repo := parts[0]
		size := strings.Join(parts[1:len(parts)-3], " ")
		return PirateFaceSearchRow{Repo: repo, Size: size, Seeds: seeds, Magnet: magnet}, true
	}
	seeds, err := strconv.Atoi(m[3])
	if err != nil {
		return PirateFaceSearchRow{}, false
	}
	return PirateFaceSearchRow{Repo: m[1], Size: m[2], Seeds: seeds, Magnet: m[4]}, true
}

// FilterPirateFaceSearch applies the default GGUF/seed view.
// When all=false, keep GGUF-like names. Prefer rows with seeds>0 unless every
// match is empty. Cap at pirateFaceSearchCap. Returns shown rows, hidden count,
// and whether only zero-seed rows were available.
func FilterPirateFaceSearch(rows []PirateFaceSearchRow, all bool) (shown []PirateFaceSearchRow, hidden int, zeroOnly bool) {
	var gguf, other []PirateFaceSearchRow
	for _, r := range rows {
		if LooksLikePirateFaceGGUF(r.Repo) {
			gguf = append(gguf, r)
		} else {
			other = append(other, r)
		}
	}
	candidates := gguf
	if all {
		candidates = append(append([]PirateFaceSearchRow{}, gguf...), other...)
	} else {
		hidden = len(other)
	}

	var seeded, empty []PirateFaceSearchRow
	for _, r := range candidates {
		if r.Seeds > 0 {
			seeded = append(seeded, r)
		} else {
			empty = append(empty, r)
		}
	}
	pick := seeded
	if len(pick) == 0 {
		pick = empty
		zeroOnly = len(empty) > 0
	}
	if len(pick) > pirateFaceSearchCap {
		hidden += len(pick) - pirateFaceSearchCap
		pick = pick[:pirateFaceSearchCap]
	}
	return pick, hidden, zeroOnly
}

// FetchPirateFaceSearch downloads and parses GET /s/QUERY.
func FetchPirateFaceSearch(ctx context.Context, query string) ([]PirateFaceSearchRow, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, fmt.Errorf("search query is required")
	}
	q := strings.Join(strings.Fields(query), "+")
	// Keep literal '+' between words (Pirate Face search grammar).
	u := PirateFaceAPIBase() + "/s/" + strings.ReplaceAll(url.PathEscape(strings.ReplaceAll(q, "+", " ")), "%20", "+")
	body, err := pirateFaceGET(ctx, u)
	if err != nil {
		return nil, err
	}
	var rows []PirateFaceSearchRow
	sc := bufio.NewScanner(strings.NewReader(string(body)))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		if row, ok := ParsePirateFaceSearchLine(sc.Text()); ok {
			rows = append(rows, row)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

// FormatPirateFaceSearchPull is the next-step lookup printed by search.
func FormatPirateFaceSearchPull(repo string) string {
	return fmt.Sprintf("zerollama pull pirateface://%s", repo)
}
