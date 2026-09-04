package server

import (
	"strings"
	"sync"
	"time"
)

const stickyElideMaxKeys = 256
const stickyElideTTL = 30 * time.Minute

type stickyElideRec struct {
	from int
	n    int
	at   time.Time
}

var stickyElideMu sync.Mutex
var stickyElideByKey = map[string]stickyElideRec{}
var stickyElideOrder []string

func stickyElideMapKey(model, cacheKey string) string {
	cacheKey = strings.TrimSpace(cacheKey)
	if cacheKey == "" {
		return ""
	}
	return strings.TrimSpace(model) + "\x00" + cacheKey
}

func resetStickyElideForTest() {
	stickyElideMu.Lock()
	defer stickyElideMu.Unlock()
	stickyElideByKey = map[string]stickyElideRec{}
	stickyElideOrder = nil
}

func chatOptionsCacheReset(opts map[string]any) bool {
	if opts == nil {
		return false
	}
	if v, ok := boolFromMap(opts, "cache_reset"); ok && v {
		return true
	}
	if z, ok := opts["zerollama"].(map[string]any); ok {
		if v, ok := boolFromMap(z, "cache_reset"); ok && v {
			return true
		}
	}
	return false
}

func forgetStickyElide(key string) {
	key = strings.TrimSpace(key)
	if key == "" {
		return
	}
	stickyElideMu.Lock()
	defer stickyElideMu.Unlock()
	delete(stickyElideByKey, key)
	out := stickyElideOrder[:0]
	for _, k := range stickyElideOrder {
		if k != key {
			out = append(out, k)
		}
	}
	stickyElideOrder = out
}

func lookupStickyElide(key string, nMsgs int) (int, bool) {
	key = strings.TrimSpace(key)
	if key == "" || nMsgs < 1 {
		return 0, false
	}
	stickyElideMu.Lock()
	defer stickyElideMu.Unlock()
	rec, ok := stickyElideByKey[key]
	if !ok {
		return 0, false
	}
	if stickyElideTTL > 0 && time.Since(rec.at) > stickyElideTTL {
		delete(stickyElideByKey, key)
		return 0, false
	}
	if nMsgs < rec.n || rec.from >= nMsgs {
		delete(stickyElideByKey, key)
		return 0, false
	}
	return rec.from, true
}

func rememberStickyElide(key string, from, nMsgs int) {
	key = strings.TrimSpace(key)
	if key == "" || nMsgs < 1 || from < 0 {
		return
	}
	stickyElideMu.Lock()
	defer stickyElideMu.Unlock()
	pruneExpiredStickyElideLocked(time.Now())
	if _, ok := stickyElideByKey[key]; !ok {
		for len(stickyElideOrder) >= stickyElideMaxKeys {
			old := stickyElideOrder[0]
			stickyElideOrder = stickyElideOrder[1:]
			delete(stickyElideByKey, old)
		}
		stickyElideOrder = append(stickyElideOrder, key)
	}
	stickyElideByKey[key] = stickyElideRec{from: from, n: nMsgs, at: time.Now()}
}

func pruneExpiredStickyElideLocked(now time.Time) {
	if stickyElideTTL <= 0 {
		return
	}
	out := stickyElideOrder[:0]
	for _, k := range stickyElideOrder {
		rec, ok := stickyElideByKey[k]
		if !ok || now.Sub(rec.at) > stickyElideTTL {
			delete(stickyElideByKey, k)
			continue
		}
		out = append(out, k)
	}
	stickyElideOrder = out
}
