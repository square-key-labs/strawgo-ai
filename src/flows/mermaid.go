package flows

import (
	"fmt"
	"strings"
)

func Mermaid(start *Node, globals ...Function) string {
	var b strings.Builder
	b.WriteString("flowchart TD\n")
	if start == nil {
		return b.String()
	}
	ids := map[*Node]string{}
	var order []*Node
	visit := func(n *Node) string {
		if id, ok := ids[n]; ok {
			return id
		}
		id := fmt.Sprintf("n%d", len(ids))
		ids[n] = id
		order = append(order, n)
		return id
	}
	edges := func(f Function) []*Node {
		out := append([]*Node(nil), f.edges...)
		if f.Next != nil && f.Next != NoResponse {
			out = append(out, f.Next)
		}
		return out
	}

	visit(start)
	var lines []string
	for i := 0; i < len(order); i++ {
		n := order[i]
		for _, f := range n.Functions {
			for _, target := range edges(f) {
				lines = append(lines, fmt.Sprintf("  %s -->|%s| %s", ids[n], mermaidText(f.Name), visit(target)))
			}
		}
	}
	for _, f := range globals {
		for _, target := range edges(f) {
			lines = append(lines, fmt.Sprintf("  global -.->|%s| %s", mermaidText(f.Name), visit(target)))
		}
	}

	for _, n := range order {
		label := n.Name
		if label == "" {
			label = ids[n]
		}
		shape := "[\"%s\"]"
		if n == start {
			shape = "([\"%s\"])"
		}
		fmt.Fprintf(&b, "  %s"+shape+"\n", ids[n], mermaidText(label))
	}
	if len(globals) > 0 {
		names := make([]string, len(globals))
		for i, f := range globals {
			names[i] = f.Name
		}
		fmt.Fprintf(&b, "  global{{\"global: %s\"}}\n", mermaidText(strings.Join(names, ", ")))
	}
	for _, line := range lines {
		b.WriteString(line)
		b.WriteByte('\n')
	}
	return b.String()
}

func mermaidText(s string) string {
	r := strings.NewReplacer(`"`, "#quot;", "|", "#124;", "\n", " ")
	return r.Replace(s)
}
