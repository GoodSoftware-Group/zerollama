package api

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"
)

var (
	generateKnownFieldsOnce sync.Once
	generateKnownFields     map[string]struct{}
)

func knownGenerateTopLevelFields() map[string]struct{} {
	generateKnownFieldsOnce.Do(func() {
		known := make(map[string]struct{}, 32)
		t := reflect.TypeOf(GenerateRequest{})
		for i := 0; i < t.NumField(); i++ {
			tag := t.Field(i).Tag.Get("json")
			if tag == "" || tag == "-" {
				continue
			}
			name, _, _ := strings.Cut(tag, ",")
			if name == "" || name == "-" {
				continue
			}
			known[name] = struct{}{}
		}
		generateKnownFields = known
	})
	return generateKnownFields
}

// CheckUnknownGenerateFields rejects invented top-level /api/generate keys (trap 77).
func CheckUnknownGenerateFields(body []byte) error {
	if len(body) == 0 {
		return nil
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil
	}
	known := knownGenerateTopLevelFields()
	var unknown []string
	for k := range raw {
		if _, ok := known[k]; !ok {
			unknown = append(unknown, k)
		}
	}
	if len(unknown) == 0 {
		return nil
	}
	sort.Strings(unknown)
	return fmt.Errorf("unknown field: %s", strings.Join(unknown, ", "))
}
