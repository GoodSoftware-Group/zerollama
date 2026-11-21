// Package gpatch decodes DAWE geodesic-patch v1 files into weightedit.Edit.
package gpatch

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"

	"github.com/ollama/ollama/x/weightedit"
)

const (
	formatName = "geodesic-patch"
	formatVer  = 1
	metaKey    = "geodesic_patch"
)

func init() {
	weightedit.RegisterCodec(&Codec{})
}

// Codec implements weightedit.Codec for geodesic-patch/1.
type Codec struct{}

func (c *Codec) Format() string { return formatName }

func (c *Codec) Detect(prefix []byte) bool {
	if len(prefix) < 8 {
		return false
	}
	headerSize := binary.LittleEndian.Uint64(prefix[:8])
	if headerSize == 0 || headerSize > 100<<20 {
		return false
	}
	if int(headerSize)+8 > len(prefix) {
		// Peek at what we have: look for the format string in the partial header.
		return bytes.Contains(prefix[8:], []byte(`"geodesic-patch`)) ||
			bytes.Contains(prefix[8:], []byte(`geodesic_patch`))
	}
	header := prefix[8 : 8+headerSize]
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(header, &raw); err != nil {
		return false
	}
	mraw, ok := raw["__metadata__"]
	if !ok {
		return false
	}
	var meta map[string]string
	if err := json.Unmarshal(mraw, &meta); err != nil {
		return false
	}
	if meta["format"] == formatName+"/"+fmt.Sprint(formatVer) {
		return true
	}
	if meta[metaKey] != "" {
		var man map[string]any
		if json.Unmarshal([]byte(meta[metaKey]), &man) == nil {
			if man["format"] == formatName {
				return true
			}
		}
	}
	return false
}

func (c *Codec) Decode(r io.Reader) (*weightedit.Edit, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	if len(data) < 8 {
		return nil, fmt.Errorf("gpatch: too short")
	}
	headerSize := binary.LittleEndian.Uint64(data[:8])
	if int(headerSize)+8 > len(data) {
		return nil, fmt.Errorf("gpatch: truncated header")
	}
	headerBytes := data[8 : 8+headerSize]
	dataStart := 8 + int(headerSize)

	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("gpatch: parse header: %w", err)
	}
	mraw, ok := header["__metadata__"]
	if !ok {
		return nil, fmt.Errorf("gpatch: not a dawe patch (no manifest)")
	}
	delete(header, "__metadata__")
	var meta map[string]string
	if err := json.Unmarshal(mraw, &meta); err != nil {
		return nil, fmt.Errorf("gpatch: metadata: %w", err)
	}
	rawMan := meta[metaKey]
	if rawMan == "" {
		return nil, fmt.Errorf("gpatch: not a dawe patch (no manifest)")
	}
	var man map[string]any
	dec := json.NewDecoder(bytes.NewReader([]byte(rawMan)))
	dec.UseNumber()
	if err := dec.Decode(&man); err != nil {
		return nil, fmt.Errorf("gpatch: manifest: %w", err)
	}
	if man["format"] != formatName {
		return nil, fmt.Errorf("gpatch: unsupported format %v", man["format"])
	}
	ver, _ := asInt(man["version"])
	if ver != formatVer {
		return nil, fmt.Errorf("gpatch: unsupported version %v", man["version"])
	}

	expectID, _ := man["patch_id"].(string)
	delete(man, "patch_id")
	canonical, err := weightedit.CanonicalPyJSON(man)
	if err != nil {
		return nil, err
	}
	gotID := sha256Hex(canonical)
	if gotID != expectID {
		return nil, fmt.Errorf("gpatch: manifest was modified (patch_id mismatch)")
	}
	man["patch_id"] = expectID

	type tensorMeta struct {
		Name         string  `json:"name"`
		Dtype        string  `json:"dtype"`
		Shape        []int64 `json:"shape"`
		Numel        int64   `json:"numel"`
		Changed      int     `json:"changed"`
		SourceSHA256 string  `json:"source_sha256"`
		ResultSHA256 string  `json:"result_sha256"`
	}
	tensorsRaw, err := json.Marshal(man["tensors"])
	if err != nil {
		return nil, err
	}
	var tmetas []tensorMeta
	if err := json.Unmarshal(tensorsRaw, &tmetas); err != nil {
		return nil, fmt.Errorf("gpatch: tensors: %w", err)
	}

	type stInfo struct {
		Dtype       string   `json:"dtype"`
		Shape       []int64  `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}
	tensors := map[string]stInfo{}
	for name, raw := range header {
		var info stInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, err
		}
		tensors[name] = info
	}
	get := func(key string) ([]byte, error) {
		info, ok := tensors[key]
		if !ok {
			return nil, fmt.Errorf("gpatch: missing tensor %s", key)
		}
		start := dataStart + int(info.DataOffsets[0])
		end := dataStart + int(info.DataOffsets[1])
		if end > len(data) {
			return nil, fmt.Errorf("gpatch: truncated payload for %s", key)
		}
		return data[start:end], nil
	}

	changes := make([]weightedit.Change, 0, len(tmetas))
	for _, t := range tmetas {
		idxRaw, err := get("index/" + t.Name)
		if err != nil {
			return nil, err
		}
		old, err := get("old/" + t.Name)
		if err != nil {
			return nil, err
		}
		neu, err := get("new/" + t.Name)
		if err != nil {
			return nil, err
		}
		idxDtype := "U32"
		if t.Numel > 1<<32 {
			idxDtype = "U64"
		}
		indices, err := unpackIndices(idxRaw, idxDtype)
		if err != nil {
			return nil, fmt.Errorf("gpatch: %s indices: %w", t.Name, err)
		}
		if len(indices) != t.Changed {
			return nil, fmt.Errorf("gpatch: %s has %d indices, manifest says %d", t.Name, len(indices), t.Changed)
		}
		ch := weightedit.Change{
			Name:         t.Name,
			Dtype:        t.Dtype,
			Shape:        t.Shape,
			Indices:      indices,
			Old:          old,
			New:          neu,
			SourceSHA256: t.SourceSHA256,
			ResultSHA256: t.ResultSHA256,
		}
		if err := weightedit.ValidateChange(ch); err != nil {
			return nil, err
		}
		changes = append(changes, ch)
	}

	payload := meta["format"]
	_ = payload
	if expectPayload, _ := man["payload_sha256"].(string); expectPayload != "" {
		if got := payloadDigest(changes); got != expectPayload {
			return nil, fmt.Errorf("gpatch: payload hash mismatch (file is corrupted or was edited)")
		}
	}

	src, err := parseIdentity(man["source"])
	if err != nil {
		return nil, fmt.Errorf("gpatch: source: %w", err)
	}
	res, err := parseIdentity(man["result"])
	if err != nil {
		return nil, fmt.Errorf("gpatch: result: %w", err)
	}
	name, _ := man["name"].(string)

	return &weightedit.Edit{
		Format:  formatName + "/" + fmt.Sprint(formatVer),
		Name:    name,
		ID:      expectID,
		Source:  src,
		Result:  res,
		Changes: changes,
	}, nil
}

func parseIdentity(v any) (weightedit.Identity, error) {
	if v == nil {
		return weightedit.Identity{}, nil
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return weightedit.Identity{}, err
	}
	var id weightedit.Identity
	if err := json.Unmarshal(raw, &id); err != nil {
		return weightedit.Identity{}, err
	}
	return id, nil
}

func unpackIndices(raw []byte, dtype string) ([]uint64, error) {
	switch dtype {
	case "U32":
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("U32 index length %d not divisible by 4", len(raw))
		}
		n := len(raw) / 4
		out := make([]uint64, n)
		for i := 0; i < n; i++ {
			out[i] = uint64(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, nil
	case "U64":
		if len(raw)%8 != 0 {
			return nil, fmt.Errorf("U64 index length %d not divisible by 8", len(raw))
		}
		n := len(raw) / 8
		out := make([]uint64, n)
		for i := 0; i < n; i++ {
			out[i] = binary.LittleEndian.Uint64(raw[i*8:])
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported index dtype %s", dtype)
	}
}

func packIndices(indices []uint64, dtype string) ([]byte, error) {
	switch dtype {
	case "U32":
		out := make([]byte, len(indices)*4)
		for i, v := range indices {
			if v > uint64(^uint32(0)) {
				return nil, fmt.Errorf("index %d exceeds U32", v)
			}
			binary.LittleEndian.PutUint32(out[i*4:], uint32(v))
		}
		return out, nil
	case "U64":
		out := make([]byte, len(indices)*8)
		for i, v := range indices {
			binary.LittleEndian.PutUint64(out[i*8:], v)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported index dtype %s", dtype)
	}
}

func payloadDigest(changes []weightedit.Change) string {
	h := sha256.New()
	for _, c := range changes {
		idxDtype := "U32"
		if c.Numel() > 1<<32 {
			idxDtype = "U64"
		}
		raw, _ := packIndices(c.Indices, idxDtype)
		h.Write(raw)
		h.Write(c.Old)
		h.Write(c.New)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func asInt(v any) (int, bool) {
	switch x := v.(type) {
	case float64:
		return int(x), true
	case json.Number:
		i, err := x.Int64()
		return int(i), err == nil
	case int:
		return x, true
	case int64:
		return int(x), true
	default:
		return 0, false
	}
}

// Write builds a geodesic-patch v1 file from an Edit. Used by tests and the
// fixture generator. The Edit's Source/Result/Changes must already be filled;
// this computes patch_id and payload_sha256.
func Write(w io.Writer, edit *weightedit.Edit) error {
	if edit == nil {
		return fmt.Errorf("nil edit")
	}
	for _, c := range edit.Changes {
		if err := weightedit.ValidateChange(c); err != nil {
			return err
		}
	}
	tensors := make([]map[string]any, 0, len(edit.Changes))
	changed := 0
	for _, c := range edit.Changes {
		n := len(c.Indices)
		if n == 0 {
			n = int(c.Numel())
		}
		changed += n
		tensors = append(tensors, map[string]any{
			"name":          c.Name,
			"dtype":         c.Dtype,
			"shape":         intsToAny(c.Shape),
			"numel":         c.Numel(),
			"changed":       n,
			"source_sha256": c.SourceSHA256,
			"result_sha256": c.ResultSHA256,
		})
	}
	man := map[string]any{
		"format":          formatName,
		"version":         formatVer,
		"name":            edit.Name,
		"description":     "",
		"source":          identityToMap(edit.Source),
		"result":          identityToMap(edit.Result),
		"tensors":         tensorsToAny(tensors),
		"changed_weights": changed,
		"payload_sha256":  payloadDigest(edit.Changes),
	}
	canonical, err := weightedit.CanonicalPyJSON(man)
	if err != nil {
		return err
	}
	patchID := sha256Hex(canonical)
	man["patch_id"] = patchID

	manJSON, err := weightedit.CanonicalPyJSON(man)
	if err != nil {
		return err
	}

	type tens struct {
		Dtype string
		Shape []int64
		Raw   []byte
	}
	payload := map[string]tens{}
	var names []string
	for _, c := range edit.Changes {
		idxDtype := "U32"
		if c.Numel() > 1<<32 {
			idxDtype = "U64"
		}
		idxRaw, err := packIndices(c.Indices, idxDtype)
		if err != nil {
			return err
		}
		nIdx := int64(len(c.Indices))
		ik := "index/" + c.Name
		ok := "old/" + c.Name
		nk := "new/" + c.Name
		payload[ik] = tens{Dtype: idxDtype, Shape: []int64{nIdx}, Raw: idxRaw}
		payload[ok] = tens{Dtype: c.Dtype, Shape: []int64{nIdx}, Raw: c.Old}
		payload[nk] = tens{Dtype: c.Dtype, Shape: []int64{nIdx}, Raw: c.New}
		names = append(names, ik, ok, nk)
	}

	header := map[string]any{
		"__metadata__": map[string]string{
			metaKey:  string(manJSON),
			"format": formatName + "/" + fmt.Sprint(formatVer),
		},
	}
	var offset int64
	for _, name := range names {
		t := payload[name]
		header[name] = map[string]any{
			"dtype":        t.Dtype,
			"shape":        intsToAny(t.Shape),
			"data_offsets": []any{offset, offset + int64(len(t.Raw))},
		}
		offset += int64(len(t.Raw))
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		return err
	}
	pad := (8 - len(headerJSON)%8) % 8
	headerJSON = append(headerJSON, bytes.Repeat([]byte{' '}, pad)...)

	var sizeBuf [8]byte
	binary.LittleEndian.PutUint64(sizeBuf[:], uint64(len(headerJSON)))
	if _, err := w.Write(sizeBuf[:]); err != nil {
		return err
	}
	if _, err := w.Write(headerJSON); err != nil {
		return err
	}
	for _, name := range names {
		if _, err := w.Write(payload[name].Raw); err != nil {
			return err
		}
	}
	edit.ID = patchID
	edit.Format = formatName + "/" + fmt.Sprint(formatVer)
	return nil
}

func identityToMap(id weightedit.Identity) map[string]any {
	m := map[string]any{}
	if id.LayoutSHA256 != "" {
		m["layout_sha256"] = id.LayoutSHA256
	}
	if id.Fingerprint != "" {
		m["fingerprint"] = id.Fingerprint
	}
	if id.NTensors != 0 {
		m["n_tensors"] = id.NTensors
	}
	if len(id.Config) > 0 {
		cfg := map[string]any{}
		for k, v := range id.Config {
			cfg[k] = v
		}
		m["config"] = cfg
	}
	if len(id.Tokenizer) > 0 {
		tok := map[string]any{}
		for k, v := range id.Tokenizer {
			tok[k] = v
		}
		m["tokenizer"] = tok
	}
	return m
}

func intsToAny(xs []int64) []any {
	out := make([]any, len(xs))
	for i, x := range xs {
		out[i] = x
	}
	return out
}

func tensorsToAny(ts []map[string]any) []any {
	out := make([]any, len(ts))
	for i, t := range ts {
		out[i] = t
	}
	return out
}
