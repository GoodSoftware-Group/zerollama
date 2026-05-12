package launch

import (
	"github.com/pkg/browser"
)

// ProjectLink opens a project URL in the default browser (no model configuration).
type ProjectLink struct {
	displayName string
	url         string
}

// NewProjectLink returns a runner that opens url when launched from the menu or CLI.
func NewProjectLink(displayName, url string) *ProjectLink {
	return &ProjectLink{displayName: displayName, url: url}
}

func (p *ProjectLink) String() string { return p.displayName }

func (p *ProjectLink) Run(model string, args []string) error {
	_ = model
	_ = args
	return browser.OpenURL(p.url)
}
