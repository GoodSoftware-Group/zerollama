package tui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/cmd/launch"
)

func launcherTestState() *launch.LauncherState {
	return &launch.LauncherState{
		LastSelection: "run",
		RunModel:      "qwen3:8b",
		Integrations: map[string]launch.LauncherIntegrationState{
			"zoey": {
				Name:        "zoey",
				DisplayName: "Zoey",
				Description: "Privacy-first, local-first AI agent framework (Rust)",
				Selectable:  true,
				Changeable:  false,
				Installed:   true,
			},
			"eliza": {
				Name:        "eliza",
				DisplayName: "elizaOS",
				Description: "Autonomous agents for everyone",
				Selectable:  true,
				Changeable:  false,
				Installed:   true,
			},
			"openclaw": {
				Name:            "openclaw",
				DisplayName:     "OpenClaw",
				Description:     "Personal AI with 100+ skills",
				Selectable:      true,
				Changeable:      true,
				AutoInstallable: true,
			},
			"opencode": {
				Name:         "opencode",
				DisplayName:  "OpenCode",
				Description:  "Anomaly's open-source coding agent",
				Selectable:   true,
				Changeable:   true,
				CurrentModel: "glm-5:cloud",
			},
			"copilot": {
				Name:        "copilot",
				DisplayName: "Copilot CLI",
				Description: "GitHub's AI coding agent for the terminal",
				Selectable:  true,
				Changeable:  true,
			},
			"hermes": {
				Name:        "hermes",
				DisplayName: "Hermes Agent",
				Description: "Self-improving AI agent built by Nous Research",
				Selectable:  true,
				Changeable:  true,
			},
			"droid": {
				Name:        "droid",
				DisplayName: "Droid",
				Description: "Factory's coding agent across terminal and IDEs",
				Selectable:  true,
				Changeable:  true,
			},
			"pi": {
				Name:        "pi",
				DisplayName: "Pi",
				Description: "Minimal AI agent toolkit with plugin support",
				Selectable:  true,
				Changeable:  true,
			},
		},
	}
}

func findMenuCursorByIntegration(items []menuItem, name string) int {
	for i, item := range items {
		if item.integration == name {
			return i
		}
	}
	return -1
}

func integrationSequence(items []menuItem) []string {
	sequence := make([]string, 0, len(items))
	for _, item := range items {
		switch {
		case item.isRunModel:
			sequence = append(sequence, "run")
		case item.isOthers:
			sequence = append(sequence, "more")
		case item.integration != "":
			sequence = append(sequence, item.integration)
		}
	}
	return sequence
}

func compareStrings(got, want []string) string {
	return cmp.Diff(want, got)
}

func TestMenuRendersPinnedItemsAndMore(t *testing.T) {
	menu := newModel(launcherTestState())
	view := menu.View()
	for _, want := range []string{"Chat with a model", "Launch Zoey", "Launch elizaOS", "Launch Hermes Agent", "More..."} {
		if !strings.Contains(view, want) {
			t.Fatalf("expected menu view to contain %q\n%s", want, view)
		}
	}
	if strings.Contains(view, "Launch OpenCode") {
		t.Fatalf("expected OpenCode to be under More, not pinned\n%s", view)
	}
	wantOrder := []string{"run", "zoey", "eliza", "hermes", "more"}
	if diff := compareStrings(integrationSequence(menu.items), wantOrder); diff != "" {
		t.Fatalf("unexpected pinned order: %s", diff)
	}
}

func TestMenuExpandsOthersFromLastSelection(t *testing.T) {
	state := launcherTestState()
	state.LastSelection = "opencode"

	menu := newModel(state)
	if !menu.showOthers {
		t.Fatal("expected others section to expand when last selection is in the overflow list")
	}
	view := menu.View()
	if !strings.Contains(view, "Launch OpenCode") {
		t.Fatalf("expected expanded view to contain overflow integration\n%s", view)
	}
	if strings.Contains(view, "More...") {
		t.Fatalf("expected expanded view to replace More... item\n%s", view)
	}
	wantOrder := []string{"run", "zoey", "eliza", "hermes", "opencode", "copilot", "droid", "pi", "openclaw"}
	if diff := compareStrings(integrationSequence(menu.items), wantOrder); diff != "" {
		t.Fatalf("unexpected expanded order: %s", diff)
	}
}

func TestMenuEnterOnRunSelectsRun(t *testing.T) {
	menu := newModel(launcherTestState())
	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyEnter})
	got := updated.(model)
	want := TUIAction{Kind: TUIActionRunModel}
	if !got.selected || got.action != want {
		t.Fatalf("expected enter on run to select run action, got selected=%v action=%v", got.selected, got.action)
	}
}

func TestMenuRightOnRunSelectsChangeRun(t *testing.T) {
	menu := newModel(launcherTestState())
	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyRight})
	got := updated.(model)
	want := TUIAction{Kind: TUIActionRunModel, ForceConfigure: true}
	if !got.selected || got.action != want {
		t.Fatalf("expected right on run to select change-run action, got selected=%v action=%v", got.selected, got.action)
	}
}

func TestMenuEnterOnIntegrationSelectsLaunch(t *testing.T) {
	state := launcherTestState()
	state.LastSelection = "opencode"
	menu := newModel(state)
	menu.cursor = findMenuCursorByIntegration(menu.items, "opencode")
	if menu.cursor == -1 {
		t.Fatal("expected opencode menu item")
	}
	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyEnter})
	got := updated.(model)
	want := TUIAction{Kind: TUIActionLaunchIntegration, Integration: "opencode"}
	if !got.selected || got.action != want {
		t.Fatalf("expected enter on integration to launch, got selected=%v action=%v", got.selected, got.action)
	}
}

func TestMenuRightOnIntegrationSelectsConfigure(t *testing.T) {
	state := launcherTestState()
	state.LastSelection = "opencode"
	menu := newModel(state)
	menu.cursor = findMenuCursorByIntegration(menu.items, "opencode")
	if menu.cursor == -1 {
		t.Fatal("expected opencode menu item")
	}
	updated, _ := menu.Update(tea.KeyMsg{Type: tea.KeyRight})
	got := updated.(model)
	want := TUIAction{Kind: TUIActionLaunchIntegration, Integration: "opencode", ForceConfigure: true}
	if !got.selected || got.action != want {
		t.Fatalf("expected right on integration to configure, got selected=%v action=%v", got.selected, got.action)
	}
}

func TestMenuIgnoresDisabledActions(t *testing.T) {
	state := launcherTestState()
	state.LastSelection = "opencode"
	opencode := state.Integrations["opencode"]
	opencode.Selectable = false
	opencode.Changeable = false
	state.Integrations["opencode"] = opencode

	menu := newModel(state)
	menu.cursor = findMenuCursorByIntegration(menu.items, "opencode")
	if menu.cursor == -1 {
		t.Fatal("expected opencode menu item")
	}

	updatedEnter, _ := menu.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if updatedEnter.(model).selected {
		t.Fatal("expected non-selectable integration to ignore enter")
	}

	updatedRight, _ := menu.Update(tea.KeyMsg{Type: tea.KeyRight})
	if updatedRight.(model).selected {
		t.Fatal("expected non-changeable integration to ignore right")
	}
}

func TestMenuShowsCurrentModelSuffixes(t *testing.T) {
	state := launcherTestState()
	state.LastSelection = "opencode"
	menu := newModel(state)
	menu.cursor = 0
	runView := menu.View()
	if !strings.Contains(runView, "(qwen3:8b)") {
		t.Fatalf("expected run row to show current model suffix\n%s", runView)
	}

	menu.cursor = findMenuCursorByIntegration(menu.items, "opencode")
	if menu.cursor == -1 {
		t.Fatal("expected opencode menu item")
	}
	integrationView := menu.View()
	if !strings.Contains(integrationView, "(glm-5:cloud)") {
		t.Fatalf("expected integration row to show current model suffix\n%s", integrationView)
	}
}

func TestMenuShowsInstallStatusAndHint(t *testing.T) {
	state := launcherTestState()
	pi := state.Integrations["pi"]
	pi.Installed = false
	pi.Selectable = false
	pi.Changeable = false
	pi.InstallHint = "Install from https://example.com/pi"
	state.Integrations["pi"] = pi

	state.LastSelection = "pi"
	menu := newModel(state)
	menu.cursor = findMenuCursorByIntegration(menu.items, "pi")
	if menu.cursor == -1 {
		t.Fatal("expected pi menu item in overflow section")
	}
	view := menu.View()
	if !strings.Contains(view, "(not installed)") {
		t.Fatalf("expected not-installed marker\n%s", view)
	}
	if !strings.Contains(view, pi.InstallHint) {
		t.Fatalf("expected install hint in description\n%s", view)
	}
}
