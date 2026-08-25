package modelrepair

// Stock Qwen3 ChatML template with /think|/no_think injection and a closed empty
// <think> block when thinking is off.
//
// Why PARSER qwen3 (not qwen3-thinking) in the Modelfile patch: under older serve
// binaries, Init(Think=nil) still treats qwen3-thinking as defaultThinking=on and
// parks answers in Thinking. Content-mode parser keeps default /api/generate
// usable until the operator rebuilds/restarts with Think defaulted before Init.
// After that restart, preferring qwen3-thinking again is optional.
const templateQwen3ThinkNoThink = `{{- $lastUserIdx := -1 -}}
{{- range $idx, $msg := .Messages -}}
{{- if eq $msg.Role "user" }}{{ $lastUserIdx = $idx }}{{ end -}}
{{- end }}
{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{ if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end -}}
<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}
{{- if eq $i $lastUserIdx }}
   {{- if and $.IsThinkSet $.Think }} /think{{- else }} /no_think{{- end -}}
{{- end }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if (and $.IsThinkSet $.Think .Thinking) -}}
<think>{{ .Thinking }}</think>
{{ end -}}
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{- if or (not $.IsThinkSet) (not $.Think) }}
<think>

</think>

{{- end }}
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }} /no_think<|im_end|>
{{ end }}<|im_start|>assistant
<think>

</think>

{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}`

// Same as templateQwen3ThinkNoThink but never emits a system role (system messages
// and .System are dropped). Used when slash-collapse is also diagnosed.
// Why drop system: moophlo-class Qwen3-Coder GGUFs collapse into "/" loops on
// ChatML system; folding system into the user turn still triggered the loop.
// Losing system-following is intentional — better a scorable user answer than
// empty eval after stop ///.
const templateQwen3ThinkNoThinkNoSystem = `{{- $lastUserIdx := -1 -}}
{{- range $idx, $msg := .Messages -}}
{{- if eq $msg.Role "user" }}{{ $lastUserIdx = $idx }}{{ end -}}
{{- end }}
{{- if .Messages }}
{{- range $i, $_ := .Messages }}
{{- if ne .Role "system" -}}
{{- $last := false -}}
{{- $hasMore := false -}}
{{- range $j, $n := $.Messages }}{{ if and (gt $j $i) (ne $n.Role "system") }}{{ $hasMore = true }}{{ end }}{{ end -}}
{{- if not $hasMore }}{{ $last = true }}{{ end -}}
{{- if eq .Role "user" }}<|im_start|>user
Reply with useful content only.
{{ .Content | stripRolePrefixes }}
{{- if eq $i $lastUserIdx }}
   {{- if and $.IsThinkSet $.Think }} /think{{- else }} /no_think{{- end -}}
{{- end }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if (and $.IsThinkSet $.Think .Thinking) -}}
<think>{{ .Thinking }}</think>
{{ end -}}
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if $hasMore }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{- if or (not $.IsThinkSet) (not $.Think) }}
<think>

</think>

{{- end }}
{{ end }}
{{- end }}
{{- end }}
{{- else }}
{{- if .Prompt }}<|im_start|>user
Reply with useful content only.
{{ .Prompt | stripRolePrefixes }} /no_think<|im_end|>
{{ end }}<|im_start|>assistant
<think>

</think>

{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}`

// Stock ChatML with system + generate Response placeholder — used when TEMPLATE
// is empty on a non-thinking Qwen3 tag. Keeps system roles (unlike slash recipes).
const templateChatMLStock = `{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end -}}
<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}`

// Appended when a ChatML Go template is missing {{ .Response }} but is otherwise
// kept. Why append not replace: custom Messages layouts may be intentional;
// generate only needs the continuation placeholder at the end.
const templateResponseSuffix = `{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}`

// Plain ChatML that drops system roles — for non-thinking slash-collapse models.
// Why not keep a stock ChatML system block: the collapse is triggered by the
// system role itself on the bad GGUFs; preserving it would leave the recipe
// ineffective. Operators who need system instructions should put them in the
// user message or use a different checkpoint.
//
// Why stripRolePrefixes + "Reply with useful content only.":
// harnesses often paste System:/User:/Assistant: into the user turn; those
// labels (and phrases like "You output XML only") push some Qwen3-Coder GGUFs
// into `///` comment-fill loops that also poison the runner slot until unload.
// Stripping labels + a one-line steer avoids the loop without client changes.
// Requires serve with template func stripRolePrefixes (zerollama template pkg).
const templateChatMLNoSystem = `{{- if .Messages }}
{{- range $i, $m := .Messages }}
{{- if ne $m.Role "system" -}}
{{- $hasMore := false -}}
{{- range $j, $n := $.Messages }}{{ if and (gt $j $i) (ne $n.Role "system") }}{{ $hasMore = true }}{{ end }}{{ end -}}
{{- if eq $m.Role "user" }}<|im_start|>user
Reply with useful content only.
{{ $m.Content | stripRolePrefixes }}<|im_end|>
{{ else if eq $m.Role "assistant" }}<|im_start|>assistant
{{ if $m.Content }}{{ $m.Content }}
{{- else if $m.ToolCalls }}<tool_call>
{{ range $m.ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if $hasMore }}<|im_end|>
{{ end }}
{{- else if eq $m.Role "tool" }}<|im_start|>user
<tool_response>
{{ $m.Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne $m.Role "assistant") (not $hasMore) }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- end }}
{{- else }}
{{- if .Prompt }}<|im_start|>user
Reply with useful content only.
{{ .Prompt | stripRolePrefixes }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}`
