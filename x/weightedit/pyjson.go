package weightedit

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
)

// CanonicalPyJSON encodes v the way Python
// json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
// does: sorted object keys, no spaces, no HTML escaping of &, <, >.
func CanonicalPyJSON(v any) ([]byte, error) {
	var buf bytes.Buffer
	if err := writePyJSON(&buf, v); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func writePyJSON(buf *bytes.Buffer, v any) error {
	switch x := v.(type) {
	case nil:
		buf.WriteString("null")
		return nil
	case bool:
		if x {
			buf.WriteString("true")
		} else {
			buf.WriteString("false")
		}
		return nil
	case float64:
		// encoding/json numbers decode as float64.
		buf.WriteString(strconv.FormatFloat(x, 'g', -1, 64))
		return nil
	case json.Number:
		buf.WriteString(x.String())
		return nil
	case int:
		buf.WriteString(strconv.Itoa(x))
		return nil
	case int64:
		buf.WriteString(strconv.FormatInt(x, 10))
		return nil
	case string:
		return writePyString(buf, x)
	case []any:
		buf.WriteByte('[')
		for i, el := range x {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := writePyJSON(buf, el); err != nil {
				return err
			}
		}
		buf.WriteByte(']')
		return nil
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		buf.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := writePyString(buf, k); err != nil {
				return err
			}
			buf.WriteByte(':')
			if err := writePyJSON(buf, x[k]); err != nil {
				return err
			}
		}
		buf.WriteByte('}')
		return nil
	default:
		// Fall back through encoding/json to a generic tree, then re-encode.
		raw, err := json.Marshal(v)
		if err != nil {
			return err
		}
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.UseNumber()
		var generic any
		if err := dec.Decode(&generic); err != nil {
			return err
		}
		return writePyJSON(buf, generic)
	}
}

func writePyString(buf *bytes.Buffer, s string) error {
	// Match Python json.dumps ensure_ascii=False: escape only control chars,
	// quotes, and backslash. Do NOT escape &, <, > (Go's encoding/json does).
	buf.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"', '\\':
			buf.WriteByte('\\')
			buf.WriteRune(r)
		case '\b':
			buf.WriteString(`\b`)
		case '\f':
			buf.WriteString(`\f`)
		case '\n':
			buf.WriteString(`\n`)
		case '\r':
			buf.WriteString(`\r`)
		case '\t':
			buf.WriteString(`\t`)
		default:
			if r < 0x20 {
				fmt.Fprintf(buf, `\u%04x`, r)
			} else {
				buf.WriteRune(r)
			}
		}
	}
	buf.WriteByte('"')
	return nil
}
