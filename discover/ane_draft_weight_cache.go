package discover

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// MaterializeANEDraftWeightFile extracts or reuses a cached MIL weight blob for the lab conv proxy.
// Returns cache path, whether the cache was reused, and any error.
func MaterializeANEDraftWeightFile(entry ANEDraftEntry, tensorName string) (path string, cached bool, err error) {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return "", false, fmt.Errorf("draft sidecar GGUF missing for %s", entry.Tag)
	}
	if strings.TrimSpace(tensorName) == "" {
		if t, _, archErr := DefaultProxyConvTensorForSidecar(draftPath); archErr == nil {
			tensorName = t
		} else {
			tensorName = DefaultProxyConvTensor()
		}
	}
	channels := entry.ProxyChannels
	if channels <= 0 {
		channels, _ = DraftANEProxyDims(entry.EmbeddingLength)
	}
	if channels <= 0 {
		return "", false, fmt.Errorf("invalid proxy channels for %s", entry.Tag)
	}

	cachePath := aneDraftWeightCachePath(draftPath, channels, tensorName)
	wantSize := int64(draftMILWeightBlobBytes(channels))

	sidecarStat, err := os.Stat(draftPath)
	if err != nil {
		return "", false, err
	}
	if cacheStat, err := os.Stat(cachePath); err == nil {
		if cacheStat.Size() == wantSize && !cacheStat.ModTime().Before(sidecarStat.ModTime()) {
			return cachePath, true, nil
		}
	}

	blob, _, err := ExtractProxyConvWeightBlob(draftPath, tensorName, channels)
	if err != nil {
		return "", false, err
	}
	if int64(len(blob)) != wantSize {
		return "", false, fmt.Errorf("extract blob size %d != expected %d", len(blob), wantSize)
	}

	dir := filepath.Dir(cachePath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", false, err
	}
	tmp := cachePath + ".tmp"
	if err := os.WriteFile(tmp, blob, 0o644); err != nil {
		return "", false, err
	}
	if err := os.Rename(tmp, cachePath); err != nil {
		_ = os.Remove(tmp)
		return "", false, err
	}
	return cachePath, false, nil
}

// MaterializeANEDraftMatmulWeightFile extracts blk.0 ffn_gate [inCh×outCh] for ANE draft matmul kernel.
func MaterializeANEDraftMatmulWeightFile(entry ANEDraftEntry, tensorName string, inCh, outCh int) (path string, cached bool, err error) {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return "", false, fmt.Errorf("draft sidecar GGUF missing for %s", entry.Tag)
	}
	if strings.TrimSpace(tensorName) == "" {
		tensorName = "blk.0.ffn_gate.weight"
	}
	if inCh <= 0 {
		inCh, _ = DraftANEProxyDims(entry.EmbeddingLength)
	}
	if outCh <= 0 {
		outCh = inCh
	}
	cachePath := aneDraftWeightCachePath(draftPath, inCh, tensorName) + fmt.Sprintf(".mm%dx%d.v2.bin", inCh, outCh)
	wantSize := int64(draftMILMatmulWeightBlobBytes(inCh, outCh))

	sidecarStat, err := os.Stat(draftPath)
	if err != nil {
		return "", false, err
	}
	if cacheStat, err := os.Stat(cachePath); err == nil {
		if cacheStat.Size() == wantSize && !cacheStat.ModTime().Before(sidecarStat.ModTime()) {
			return cachePath, true, nil
		}
	}

	blob, _, err := ExtractProxyMatmulWeightBlob(draftPath, tensorName, inCh, outCh)
	if err != nil {
		return "", false, err
	}
	if int64(len(blob)) != wantSize {
		return "", false, fmt.Errorf("matmul blob size %d != expected %d", len(blob), wantSize)
	}
	dir := filepath.Dir(cachePath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", false, err
	}
	tmp := cachePath + ".tmp"
	if err := os.WriteFile(tmp, blob, 0o644); err != nil {
		return "", false, err
	}
	if err := os.Rename(tmp, cachePath); err != nil {
		_ = os.Remove(tmp)
		return "", false, err
	}
	return cachePath, false, nil
}

// MaterializeANEDraftNormGammaFile extracts RMS gamma for host hidden_norm (P10 chain 11).
func MaterializeANEDraftNormGammaFile(entry ANEDraftEntry, tensorName string, dim int) (path string, cached bool, err error) {
	draftPath, present := resolveDraftGGUFPath(entry)
	if !present || draftPath == "" {
		return "", false, fmt.Errorf("draft sidecar GGUF missing for %s", entry.Tag)
	}
	if strings.TrimSpace(tensorName) == "" || dim <= 0 {
		return "", false, fmt.Errorf("invalid norm gamma request")
	}

	cachePath := aneDraftWeightCachePath(draftPath, dim, tensorName)
	sidecarStat, err := os.Stat(draftPath)
	if err != nil {
		return "", false, err
	}
	if cacheStat, err := os.Stat(cachePath); err == nil {
		if !cacheStat.ModTime().Before(sidecarStat.ModTime()) && cacheStat.Size() > 0 {
			return cachePath, true, nil
		}
	}

	blob, _, err := ExtractNormVectorWeightBlob(draftPath, tensorName, dim)
	if err != nil {
		return "", false, err
	}
	if err := os.MkdirAll(filepath.Dir(cachePath), 0o755); err != nil {
		return "", false, err
	}
	tmp := cachePath + ".tmp"
	if err := os.WriteFile(tmp, blob, 0o644); err != nil {
		return "", false, err
	}
	if err := os.Rename(tmp, cachePath); err != nil {
		_ = os.Remove(tmp)
		return "", false, err
	}
	return cachePath, false, nil
}

func aneDraftWeightCacheDir() string {
	if v := strings.TrimSpace(os.Getenv("ZEROLLAMA_ANE_DRAFT_WEIGHT_CACHE")); v != "" {
		return v
	}
	home := os.Getenv("HOME")
	if home == "" {
		home = "/"
	}
	return filepath.Join(home, ".cache", "zerollama", "ane-draft-weights")
}

func aneDraftWeightCachePath(sidecarPath string, channels int, tensorName string) string {
	base := strings.TrimSuffix(filepath.Base(sidecarPath), filepath.Ext(sidecarPath))
	tensorSlug := strings.NewReplacer(".", "_", "/", "_").Replace(tensorName)
	return filepath.Join(aneDraftWeightCacheDir(), fmt.Sprintf("%s-%d-%s.v3.weight.bin", base, channels, tensorSlug))
}
