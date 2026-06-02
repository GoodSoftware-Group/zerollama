package server

import "strings"

// TrainingPriority controls idle-wait and defer-queue behavior on submit.
type TrainingPriority int

const (
	TrainingPriorityNormal TrainingPriority = iota
	TrainingPriorityLow
	TrainingPriorityHigh
)

func parseTrainingPriority(raw string) TrainingPriority {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "low", "batch", "background":
		return TrainingPriorityLow
	case "high", "interactive", "urgent":
		return TrainingPriorityHigh
	default:
		return TrainingPriorityNormal
	}
}

func (p TrainingPriority) bypassesIdleWait() bool {
	return p == TrainingPriorityHigh
}

func (p TrainingPriority) prefersDeferQueue() bool {
	return p == TrainingPriorityLow
}
