package flows

import (
	"fmt"
	"regexp"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

var placeholder = regexp.MustCompile(`(\\?)\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\}\}`)

func (s *State) render(node, text string) (string, error) {
	var missing error
	out := placeholder.ReplaceAllStringFunc(text, func(match string) string {
		groups := placeholder.FindStringSubmatch(match)
		if groups[1] != "" {
			return match[1:]
		}
		v, ok := s.lookupPath(groups[2])
		if !ok {
			if missing == nil {
				missing = fmt.Errorf("flows: node %q uses {{ %s }}, which is not in state", node, groups[2])
			}
			return match
		}
		return fmt.Sprint(v)
	})
	return out, missing
}

func (s *State) renderMessages(node string, in []llm.Message) ([]llm.Message, error) {
	if len(in) == 0 {
		return in, nil
	}
	out := make([]llm.Message, len(in))
	for i, m := range in {
		out[i] = m
		out[i].Parts = make([]llm.Part, len(m.Parts))
		for j, p := range m.Parts {
			if t, ok := p.(llm.TextPart); ok {
				text, err := s.render(node, t.Text)
				if err != nil {
					return nil, err
				}
				p = llm.TextPart{Text: text}
			}
			out[i].Parts[j] = p
		}
	}
	return out, nil
}

func (s *State) renderNode(name string, n *Node) (*Node, error) {
	out := *n
	out.Name = name
	var err error
	if out.RoleMessage, err = s.render(name, n.RoleMessage); err != nil {
		return nil, err
	}
	if out.RoleMessages, err = s.renderMessages(name, n.RoleMessages); err != nil {
		return nil, err
	}
	if out.TaskMessages, err = s.renderMessages(name, n.TaskMessages); err != nil {
		return nil, err
	}
	if out.PreActions, err = s.renderActions(name, n.PreActions); err != nil {
		return nil, err
	}
	if out.PostActions, err = s.renderActions(name, n.PostActions); err != nil {
		return nil, err
	}
	return &out, nil
}

func (s *State) renderActions(node string, in []Action) ([]Action, error) {
	if len(in) == 0 {
		return in, nil
	}
	out := make([]Action, len(in))
	for i, a := range in {
		if say, ok := a.(TTSSay); ok {
			text, err := s.render(node, say.Text)
			if err != nil {
				return nil, err
			}
			say.Text = text
			a = say
		}
		out[i] = a
	}
	return out, nil
}
