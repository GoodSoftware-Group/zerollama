package renderers

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	harmonyKnowledgeCutoff = "2024-06"
)

// HarmonyRenderer renders chat prompts for GPT-OSS (harmony format).
type HarmonyRenderer struct{}

func (r *HarmonyRenderer) LeadingBOS() string {
	return ""
}

func (r *HarmonyRenderer) Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	var sb strings.Builder

	hasTools := len(tools) > 0
	sb.WriteString(r.renderSystemMessage(hasTools, think))

	var instructions string
	loop := messages
	if len(loop) > 0 && loop[0].Role == "system" {
		instructions = loop[0].Content
		loop = loop[1:]
	}

	if instructions != "" || hasTools {
		sb.WriteString(r.renderDeveloperMessage(instructions, tools))
	}

	for i, msg := range loop {
		switch msg.Role {
		case "user":
			sb.WriteString(r.renderUserMessage(msg.Content))
		case "assistant":
			sb.WriteString(r.renderAssistantHistory(msg))
		case "tool":
			name := strings.TrimSpace(msg.ToolName)
			if name == "" {
				name = "tool"
			}
			sb.WriteString(fmt.Sprintf("<|start|>functions.%s to=assistant<|channel|>commentary<|message|>%s<|end|>",
				name, msg.Content))
		default:
			if msg.Content != "" {
				sb.WriteString(r.renderUserMessage(msg.Content))
			}
		}

		// Open generation turn after the last message unless we're prefilling assistant.
		if i == len(loop)-1 && msg.Role != "assistant" {
			sb.WriteString("<|start|>assistant")
		}
	}

	if len(loop) == 0 {
		sb.WriteString("<|start|>assistant")
	}

	return sb.String(), nil
}

func (r *HarmonyRenderer) renderSystemMessage(hasTools bool, think *api.ThinkValue) string {
	var sb strings.Builder
	sb.WriteString("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n")
	sb.WriteString("Knowledge cutoff: ")
	sb.WriteString(harmonyKnowledgeCutoff)
	sb.WriteString("\nCurrent date: ")
	sb.WriteString(time.Now().Format("2006-01-02"))
	sb.WriteString("\n\nReasoning: ")
	sb.WriteString(harmonyReasoningLevel(think))
	sb.WriteString("\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.")
	if hasTools {
		sb.WriteString("\nCalls to these tools must go to the commentary channel: 'functions'.")
	}
	sb.WriteString("<|end|>")
	return sb.String()
}

func harmonyReasoningLevel(think *api.ThinkValue) string {
	if think == nil {
		return "medium"
	}
	if think.IsString() {
		return think.String()
	}
	if think.Bool() {
		return "medium"
	}
	return "medium"
}

func (r *HarmonyRenderer) renderDeveloperMessage(instructions string, tools []api.Tool) string {
	var sb strings.Builder
	sb.WriteString("<|start|>developer<|message|>")
	if strings.TrimSpace(instructions) != "" {
		sb.WriteString("# Instructions\n\n")
		sb.WriteString(strings.TrimSpace(instructions))
	}
	if len(tools) > 0 {
		if strings.TrimSpace(instructions) != "" {
			sb.WriteString("\n\n")
		}
		sb.WriteString("# Tools\n\n## functions\n\nnamespace functions {\n\n")
		names := make([]string, 0, len(tools))
		for _, tool := range tools {
			if tool.Function.Name != "" {
				names = append(names, tool.Function.Name)
			}
		}
		sort.Strings(names)
		byName := make(map[string]api.Tool, len(tools))
		for _, tool := range tools {
			byName[tool.Function.Name] = tool
		}
		for _, name := range names {
			sb.WriteString(r.renderHarmonyToolType(byName[name]))
			sb.WriteString("\n")
		}
		sb.WriteString("} // namespace functions")
	}
	sb.WriteString("<|end|>")
	return sb.String()
}

func (r *HarmonyRenderer) renderHarmonyToolType(tool api.Tool) string {
	var sb strings.Builder
	if tool.Function.Description != "" {
		sb.WriteString("// ")
		sb.WriteString(tool.Function.Description)
		sb.WriteString("\n")
	}
	props := tool.Function.Parameters.Properties
	if props == nil || props.Len() == 0 {
		sb.WriteString("type ")
		sb.WriteString(tool.Function.Name)
		sb.WriteString(" = () => any;")
		return sb.String()
	}
	sb.WriteString("type ")
	sb.WriteString(tool.Function.Name)
	sb.WriteString(" = (_: {\n")
	for key, prop := range props.All() {
		if prop.Description != "" {
			sb.WriteString("// ")
			sb.WriteString(prop.Description)
			sb.WriteString("\n")
		}
		sb.WriteString(key)
		if !containsString(tool.Function.Parameters.Required, key) {
			sb.WriteString("?")
		}
		sb.WriteString(": ")
		sb.WriteString(prop.ToTypeScriptType())
		sb.WriteString(",\n")
	}
	sb.WriteString("}) => any;")
	return sb.String()
}

func containsString(list []string, target string) bool {
	for _, s := range list {
		if s == target {
			return true
		}
	}
	return false
}

func (r *HarmonyRenderer) renderUserMessage(content string) string {
	return fmt.Sprintf("<|start|>user<|message|>%s<|end|>", content)
}

func (r *HarmonyRenderer) renderAssistantHistory(msg api.Message) string {
	var sb strings.Builder

	if msg.Thinking != "" {
		sb.WriteString(fmt.Sprintf("<|start|>assistant<|channel|>analysis<|message|>%s<|end|>", msg.Thinking))
	}

	for _, tc := range msg.ToolCalls {
		args, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			args = []byte("{}")
		}
		sb.WriteString(fmt.Sprintf("<|start|>assistant<|channel|>commentary to=functions.%s <|constrain|>json<|message|>%s<|end|>",
			tc.Function.Name, string(args)))
	}

	if msg.Content != "" {
		sb.WriteString(fmt.Sprintf("<|start|>assistant<|channel|>final<|message|>%s<|end|>", msg.Content))
	}

	return sb.String()
}
