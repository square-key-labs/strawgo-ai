package llm

import "sync"

type Snapshot struct {
	Messages   []Message
	Tools      Tools
	ToolChoice ToolChoice
	Version    uint64
}

type Context struct {
	mu         sync.RWMutex
	messages   []Message
	tools      Tools
	toolChoice ToolChoice
	version    uint64
}

func NewContext(messages ...Message) *Context {
	c := &Context{}
	c.messages = append(c.messages, messages...)
	return c
}

func (c *Context) Snapshot() Snapshot {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return Snapshot{
		Messages:   c.messages[:len(c.messages):len(c.messages)],
		Tools:      c.tools,
		ToolChoice: c.toolChoice,
		Version:    c.version,
	}
}

func (c *Context) Messages() []Message {
	return c.Snapshot().Messages
}

func (c *Context) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.messages)
}

func (c *Context) Add(messages ...Message) {
	if len(messages) == 0 {
		return
	}
	c.mu.Lock()
	next := make([]Message, len(c.messages), len(c.messages)+len(messages))
	copy(next, c.messages)
	c.messages = append(next, messages...)
	c.version++
	c.mu.Unlock()
}

func (c *Context) Set(messages []Message) {
	c.mu.Lock()
	c.messages = append([]Message(nil), messages...)
	c.version++
	c.mu.Unlock()
}

func (c *Context) Transform(fn func([]Message) []Message) {
	c.mu.Lock()
	in := make([]Message, len(c.messages))
	for i, m := range c.messages {
		in[i] = m.clone()
	}
	c.messages = fn(in)
	c.version++
	c.mu.Unlock()
}

func (c *Context) SetTools(tools Tools) {
	c.mu.Lock()
	c.tools = tools
	c.version++
	c.mu.Unlock()
}

func (c *Context) Tools() Tools {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.tools
}

func (c *Context) SetToolChoice(choice ToolChoice) {
	c.mu.Lock()
	c.toolChoice = choice
	c.version++
	c.mu.Unlock()
}

func (c *Context) ReplaceToolResult(toolCallID, content string, isError bool) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i := len(c.messages) - 1; i >= 0; i-- {
		m := c.messages[i]
		for j, p := range m.Parts {
			r, ok := p.(ToolResultPart)
			if !ok || r.ToolCallID != toolCallID {
				continue
			}
			next := make([]Message, len(c.messages))
			copy(next, c.messages)
			updated := m.clone()
			r.Content = content
			r.IsError = isError
			updated.Parts[j] = r
			next[i] = updated
			c.messages = next
			c.version++
			return true
		}
	}
	return false
}
