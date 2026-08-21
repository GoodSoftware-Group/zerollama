package parsers

func (p *CogitoParser) PreservedTokens() []string {
	return []string{
		cogitoThinkingCloseTag,
		cogitoToolCallsBeginTag,
		cogitoToolCallsEndTag,
		cogitoToolCallBeginTag,
		cogitoToolCallEndTag,
		cogitoToolSepTag,
		cogitoToolOutputBeginTag,
		cogitoToolOutputEndTag,
		cogitoToolOutputsBeginTag,
		cogitoToolOutputsEndTag,
	}
}

func (p *DeepSeek3Parser) PreservedTokens() []string {
	return []string{
		deepseekThinkingCloseTag,
		deepseekToolCallsBeginTag,
		deepseekToolCallsEndTag,
		deepseekToolCallBeginTag,
		deepseekToolCallEndTag,
		deepseekToolSepTag,
		deepseekToolOutputBeginTag,
		deepseekToolOutputEndTag,
	}
}

func (p *FunctionGemmaParser) PreservedTokens() []string {
	return []string{
		functionGemmaFunctionCallOpen,
		functionGemmaFunctionCallClose,
	}
}

func (p *Gemma4Parser) PreservedTokens() []string {
	return []string{
		gemma4ThinkingOpenTag,
		gemma4ThinkingCloseTag,
		gemma4ToolCallOpenTag,
		gemma4ToolCallCloseTag,
		gemma4ToolResponseTag,
		gemma4StringDelimiter,
	}
}

func (p *GLM46Parser) PreservedTokens() []string {
	return []string{
		glm46ThinkingOpenTag,
		glm46ThinkingCloseTag,
		glm46ToolOpenTag,
		glm46ToolCloseTag,
	}
}

func (p *MinistralParser) PreservedTokens() []string {
	return []string{
		ministralToolCallsTag,
		ministralThinkTag,
		ministralThinkEndTag,
		ministralArgsTag,
	}
}

func (p *Olmo3Parser) PreservedTokens() []string {
	return []string{
		olmo3FuncCallsOpenTag,
		olmo3FuncCallsCloseTag,
	}
}

func (p *Olmo3ThinkParser) PreservedTokens() []string {
	return []string{
		olmo3ThinkCloseTag,
	}
}

func (p *Qwen3Parser) PreservedTokens() []string {
	return []string{
		qwen3ThinkingOpenTag,
		qwen3ThinkingCloseTag,
		qwen3ToolOpenTag,
		qwen3ToolCloseTag,
	}
}

func (p *Qwen3VLParser) PreservedTokens() []string {
	return []string{
		thinkingCloseTag,
		toolOpenTag,
		toolCloseTag,
	}
}
