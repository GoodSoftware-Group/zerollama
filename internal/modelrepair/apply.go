package modelrepair

import (
	"context"
	"fmt"
)

// Apply recreates the tag FROM itself with the patch overlay.
// Why FROM=name (same tag): preserves weight blobs and only replaces template /
// parser / parameter layers via /api/create — no re-download. Empty template or
// parser on the request would inherit FROM; BuildPatch always sets both for
// patchable recipes so we never rely on that accidental safety.
func Apply(ctx context.Context, api API, name string, patch *Patch) error {
	if patch == nil {
		return fmt.Errorf("nil patch")
	}
	return api.Create(ctx, name, name, patch.Template, patch.Parser, patch.Parameters)
}
