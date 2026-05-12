package launch

import (
	"fmt"
	"os"
	"os/exec"
)

// Zoey implements Runner for the Zoey CLI (Rust); see https://github.com/Agent-Zoey/Zoey
type Zoey struct{}

const zoeyGitURL = "https://github.com/Agent-Zoey/Zoey"

func (z *Zoey) String() string { return "Zoey" }

// standaloneLaunch marks Zoey as launching without Ollama model selection.
func (*Zoey) standaloneLaunch() {}

func findZoey() (string, bool) {
	if p, err := exec.LookPath("zoey"); err == nil {
		return p, true
	}
	return "", false
}

func (z *Zoey) Run(model string, args []string) error {
	_ = model
	bin, ok := findZoey()
	if !ok {
		return fmt.Errorf("zoey is not installed, install from %s", zoeyGitURL)
	}
	cmd := exec.Command(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func ensureZoeyInstalled() error {
	if _, ok := findZoey(); ok {
		return nil
	}
	if _, err := exec.LookPath("cargo"); err != nil {
		return fmt.Errorf(
			"zoey is not installed and Cargo was not found\n\nInstall Rust: https://rustup.rs/\n\nThen install Zoey:\n  cargo install --locked --git %s zoey-cli\n\nThen re-run:\n  zerollama launch zoey",
			zoeyGitURL,
		)
	}
	ok, err := ConfirmPrompt("Zoey is not installed. Install with cargo from source? (This compiles Zoey and may take several minutes.)")
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("zoey installation cancelled")
	}
	fmt.Fprintf(os.Stderr, "\n%sInstalling Zoey CLI (cargo install)...%s\n", ansiGray, ansiReset)
	cmd := exec.Command("cargo", "install", "--locked", "--git", zoeyGitURL, "zoey-cli")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to install zoey: %w", err)
	}
	if _, ok := findZoey(); !ok {
		return fmt.Errorf("zoey was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}
	fmt.Fprintf(os.Stderr, "%sZoey installed successfully%s\n\n", ansiGreen, ansiReset)
	return nil
}
