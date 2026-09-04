package api

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"
)

// CoerceToolCalls applies mlx-serve tool autocorrect: JSON Schema type
// coercion plus hoisting a required top-level argument buried in a nested
// object. Correct calls are unchanged. Unknown tools are left alone.
func CoerceToolCalls(calls []ToolCall, tools Tools) []ToolCall {
	if len(calls) == 0 || len(tools) == 0 {
		return calls
	}
	for i := range calls {
		params, ok := toolParamsByName(tools, calls[i].Function.Name)
		if !ok {
			continue
		}
		coerceArgs(&calls[i].Function.Arguments, params)
		hoistBuriedRequired(&calls[i].Function.Arguments, params)
	}
	return calls
}

func toolParamsByName(tools Tools, name string) (ToolFunctionParameters, bool) {
	for _, t := range tools {
		if t.Function.Name == name {
			return t.Function.Parameters, true
		}
	}
	return ToolFunctionParameters{}, false
}

func coerceArgs(args *ToolCallFunctionArguments, params ToolFunctionParameters) {
	if args == nil || params.Properties == nil {
		return
	}
	var keys []string
	for k := range args.All() {
		keys = append(keys, k)
	}
	for _, k := range keys {
		v, ok := args.Get(k)
		if !ok {
			continue
		}
		prop, ok := params.Properties.Get(k)
		if !ok {
			continue
		}
		args.Set(k, coerceJSONValue(v, prop))
	}
}

func coerceJSONValue(v any, prop ToolProperty) any {
	if v == nil {
		return nil
	}
	if len(prop.AnyOf) > 0 {
		for _, alt := range prop.AnyOf {
			if coerced, ok := tryCoerce(v, alt); ok {
				return coerced
			}
		}
		return v
	}
	if coerced, ok := tryCoerce(v, prop); ok {
		return coerced
	}
	return v
}

func tryCoerce(v any, prop ToolProperty) (any, bool) {
	if m, ok := v.(map[string]any); ok && prop.Properties != nil && hasJSONType(prop.Type, "object") {
		for k, child := range prop.Properties.All() {
			if cv, ok := m[k]; ok {
				m[k] = coerceJSONValue(cv, child)
			}
		}
		return m, true
	}
	types := prop.Type
	if len(types) == 0 {
		return v, true
	}
	switch x := v.(type) {
	case string:
		return coerceString(x, types), true
	case json.Number:
		return coerceString(string(x), types), true
	case float64:
		if hasJSONType(types, "integer") && x == math.Trunc(x) {
			return int(x), true
		}
		if hasJSONType(types, "number") || hasJSONType(types, "integer") {
			return x, true
		}
		if hasJSONType(types, "string") {
			if x == math.Trunc(x) {
				return strconv.FormatInt(int64(x), 10), true
			}
			return strconv.FormatFloat(x, 'f', -1, 64), true
		}
	case int:
		if hasJSONType(types, "integer") || hasJSONType(types, "number") {
			return x, true
		}
		if hasJSONType(types, "string") {
			return strconv.Itoa(x), true
		}
	case int64:
		if hasJSONType(types, "integer") || hasJSONType(types, "number") {
			return x, true
		}
		if hasJSONType(types, "string") {
			return strconv.FormatInt(x, 10), true
		}
	case bool:
		if hasJSONType(types, "boolean") {
			return x, true
		}
		if hasJSONType(types, "string") {
			return strconv.FormatBool(x), true
		}
	}
	return v, false
}

func hasJSONType(types PropertyType, want string) bool {
	for _, t := range types {
		if t == want {
			return true
		}
	}
	return false
}

func coerceString(raw string, types PropertyType) any {
	if strings.EqualFold(raw, "null") {
		return nil
	}
	typeSet := map[string]bool{}
	for _, t := range types {
		typeSet[t] = true
	}
	if typeSet["boolean"] {
		switch strings.ToLower(raw) {
		case "true":
			return true
		case "false":
			return false
		}
		if len(types) == 1 {
			return false
		}
	}
	if typeSet["integer"] {
		if i, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 64); err == nil {
			if i >= math.MinInt32 && i <= math.MaxInt32 {
				return int(i)
			}
			return i
		}
		if len(types) == 1 {
			return raw
		}
	}
	if typeSet["number"] {
		if f, err := strconv.ParseFloat(strings.TrimSpace(raw), 64); err == nil {
			if f == math.Trunc(f) {
				i := int64(f)
				if i >= math.MinInt32 && i <= math.MaxInt32 {
					return int(i)
				}
				return i
			}
			return f
		}
		if len(types) == 1 {
			return raw
		}
	}
	if typeSet["array"] {
		var arr []any
		if json.Unmarshal([]byte(raw), &arr) == nil {
			return arr
		}
		if len(types) == 1 {
			return raw
		}
	}
	if typeSet["object"] {
		var obj map[string]any
		if json.Unmarshal([]byte(raw), &obj) == nil {
			return obj
		}
		if len(types) == 1 {
			return raw
		}
	}
	return raw
}

// hoistBuriedRequired moves a missing required top-level key out of a nested
// object when that nested schema does not declare the key (edit.path → path).
func hoistBuriedRequired(args *ToolCallFunctionArguments, params ToolFunctionParameters) {
	if args == nil || params.Properties == nil || len(params.Required) == 0 {
		return
	}
	for _, key := range params.Required {
		if _, ok := args.Get(key); ok {
			continue
		}
		prop, ok := params.Properties.Get(key)
		if !ok {
			continue
		}
		parentKey, nestedVal, ok := findUniqueBuried(args, params, key)
		if !ok {
			continue
		}
		args.Set(key, coerceJSONValue(nestedVal, prop))
		parent, _ := args.Get(parentKey)
		pm, ok := parent.(map[string]any)
		if !ok {
			continue
		}
		delete(pm, key)
		args.Set(parentKey, pm)
	}
}

func findUniqueBuried(args *ToolCallFunctionArguments, params ToolFunctionParameters, key string) (parent string, val any, ok bool) {
	n := 0
	for k, v := range args.All() {
		nested, isMap := v.(map[string]any)
		if !isMap {
			continue
		}
		nv, has := nested[key]
		if !has {
			continue
		}
		if parentProp, okp := params.Properties.Get(k); okp && parentProp.Properties != nil {
			if _, declared := parentProp.Properties.Get(key); declared {
				continue
			}
		}
		n++
		parent, val = k, nv
	}
	if n != 1 {
		return "", nil, false
	}
	return parent, val, true
}
