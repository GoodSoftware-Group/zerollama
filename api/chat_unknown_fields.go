package api

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"
)

// chatPassthroughFields are recognized native /api/chat keys that may not appear
// on ChatRequest but must not 400. Keep empty unless a real SDK field is needed;
// enable_thinking / chat_template_kwargs stay excluded (minefield trap 77).
var chatPassthroughFields = []string{}

var (
	chatKnownFieldsOnce sync.Once
	chatKnownFields     map[string]struct{}
)

func knownChatTopLevelFields() map[string]struct{} {
	chatKnownFieldsOnce.Do(func() {
		known := make(map[string]struct{}, 24)
		t := reflect.TypeOf(ChatRequest{})
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
		for _, name := range chatPassthroughFields {
			known[name] = struct{}{}
		}
		chatKnownFields = known
	})
	return chatKnownFields
}

// CheckUnknownChatFields returns an error when the body contains top-level keys
// outside the known /api/chat surface (minefield trap 77 parity with /v1).
// Non-object bodies are left for normal JSON bind errors.
func CheckUnknownChatFields(body []byte) error {
	if len(body) == 0 {
		return nil
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil
	}
	known := knownChatTopLevelFields()
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
