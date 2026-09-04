package api

// OutputBudgetTightThreshold is mlx-serve outputBudgetGuidance: inject concise-reply
// text only when the remaining decode budget is under this many tokens.
const OutputBudgetTightThreshold = 12288

// OutputBudgetGuidance is appended to the last user turn when the budget is tight.
const OutputBudgetGuidance = "Keep the reply concise; the remaining output budget is tight."

func OutputBudgetIsTight(numPredict int) bool {
	return numPredict > 0 && numPredict < OutputBudgetTightThreshold
}

func NumPredictFromMap(opts map[string]any) int {
	if opts == nil {
		return 0
	}
	switch v := opts["num_predict"].(type) {
	case int:
		return v
	case int32:
		return int(v)
	case int64:
		return int(v)
	case float32:
		return int(v)
	case float64:
		return int(v)
	default:
		return 0
	}
}

func AppendOutputBudgetGuidance(msgs []Message, numPredict int) []Message {
	if !OutputBudgetIsTight(numPredict) {
		return msgs
	}
	return appendLastUserParagraph(msgs, OutputBudgetGuidance)
}
