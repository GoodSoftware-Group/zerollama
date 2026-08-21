package server

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/envconfig"
)

type aliasFile struct {
	Aliases map[string]string `yaml:"aliases" json:"aliases"`
}

type AliasInfo struct {
	Name   string `json:"name"`
	Target string `json:"target"`
}

var (
	aliasFileMu    sync.Mutex
	aliasFilePath  string
	aliasFileMod   time.Time
	aliasFileCache map[string]string

	aliasExtraMu sync.Mutex
	aliasExtra   = map[string]string{}
)

func loadAliasMap() (map[string]string, error) {
	path := envconfig.AliasesConfigPath()
	out := map[string]string{}
	if path != "" {
		st, err := os.Stat(path)
		if err != nil {
			if !os.IsNotExist(err) {
				return nil, err
			}
		} else {
			aliasFileMu.Lock()
			if aliasFileCache != nil && aliasFilePath == path && st.ModTime().Equal(aliasFileMod) {
				for k, v := range aliasFileCache {
					out[k] = v
				}
				aliasFileMu.Unlock()
			} else {
				raw, err := os.ReadFile(path)
				if err != nil {
					aliasFileMu.Unlock()
					return nil, err
				}
				var file aliasFile
				if err := yaml.Unmarshal(raw, &file); err != nil {
					aliasFileMu.Unlock()
					return nil, fmt.Errorf("aliases config: %w", err)
				}
				if file.Aliases == nil {
					file.Aliases = map[string]string{}
				}
				aliasFileCache = file.Aliases
				aliasFilePath = path
				aliasFileMod = st.ModTime()
				for k, v := range file.Aliases {
					out[k] = v
				}
				aliasFileMu.Unlock()
			}
		}
	}
	aliasExtraMu.Lock()
	for k, v := range aliasExtra {
		out[k] = v
	}
	aliasExtraMu.Unlock()
	return out, nil
}

func listAliases() ([]AliasInfo, error) {
	m, err := loadAliasMap()
	if err != nil {
		return nil, err
	}
	out := make([]AliasInfo, 0, len(m))
	names := make([]string, 0, len(m))
	for name := range m {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		out = append(out, AliasInfo{Name: name, Target: m[name]})
	}
	return out, nil
}

func addAliasOverlay(name, target string) {
	aliasExtraMu.Lock()
	aliasExtra[strings.TrimSpace(name)] = strings.TrimSpace(target)
	aliasExtraMu.Unlock()
}

// resolveAlias follows a one-hop name → target map. Non-aliases return
// (name, false, nil). Chains and missing targets are errors.
func resolveAlias(name string) (served string, aliased bool, err error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return name, false, nil
	}
	m, err := loadAliasMap()
	if err != nil {
		return "", false, err
	}
	target, ok := m[name]
	if !ok {
		return name, false, nil
	}
	target = strings.TrimSpace(target)
	if target == "" {
		return "", true, fmt.Errorf("alias %q has empty target", name)
	}
	if target == name {
		return "", true, fmt.Errorf("alias %q points to itself", name)
	}
	if _, chained := m[target]; chained {
		return "", true, fmt.Errorf("alias %q points to another alias %q (chains are not allowed)", name, target)
	}
	return target, true, nil
}

func applyModelAlias(c *gin.Context, name string) (string, error) {
	served, aliased, err := resolveAlias(name)
	if err != nil {
		return "", err
	}
	if aliased {
		c.Header("X-Zerollama-Alias", name)
		c.Header("X-Zerollama-Alias-Target", served)
	}
	return served, nil
}
