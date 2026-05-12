package launch

import (
	"fmt"
	"os"
	"os/exec"
)

// Eliza implements Runner for the elizaOS CLI; see https://docs.elizaos.ai/
type Eliza struct{}

const elizaNpmPackage = "@elizaos/cli"

func (e *Eliza) String() string { return "elizaOS" }

// standaloneLaunch marks elizaOS as launching without Ollama model selection.
func (*Eliza) standaloneLaunch() {}

func findElizaOS() (string, bool) {
	if p, err := exec.LookPath("elizaos"); err == nil {
		return p, true
	}
	return "", false
}

func (e *Eliza) Run(model string, args []string) error {
	_ = model
	bin, ok := findElizaOS()
	if !ok {
		return fmt.Errorf("elizaos is not installed, install from https://docs.elizaos.ai/installation")
	}
	cmd := exec.Command(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func ensureElizaInstalled() error {
	if _, ok := findElizaOS(); ok {
		return nil
	}
	if _, err := exec.LookPath("npm"); err != nil {
		return fmt.Errorf("elizaos is not installed and npm was not found\n\nInstall Node.js: https://nodejs.org/\n\nThen re-run:\n  zerollama launch eliza")
	}
	ok, err := ConfirmPrompt("elizaOS CLI is not installed. Install globally with npm?")
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("eliza installation cancelled")
	}
	fmt.Fprintf(os.Stderr, "\n%sInstalling elizaOS CLI...%s\n", ansiGray, ansiReset)
	cmd := exec.Command("npm", "install", "-g", elizaNpmPackage+"@latest")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to install elizaos: %w", err)
	}
	if _, ok := findElizaOS(); !ok {
		return fmt.Errorf("elizaos was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}
	fmt.Fprintf(os.Stderr, "%selizaOS CLI installed successfully%s\n\n", ansiGreen, ansiReset)
	return nil
}
