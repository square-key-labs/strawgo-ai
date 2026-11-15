package rtvi

import "github.com/square-key-labs/strawgo-ai/src/frames"

func systemFrame(name string) *frames.SystemFrame {
	return &frames.SystemFrame{BaseFrame: frames.NewBaseFrame(name)}
}

type ClientMessageFrame struct {
	*frames.SystemFrame
	MsgID string
	Type  string
	Data  any
}

func NewClientMessageFrame(msgID, typ string, data any) *ClientMessageFrame {
	return &ClientMessageFrame{SystemFrame: systemFrame("RTVIClientMessageFrame"), MsgID: msgID, Type: typ, Data: data}
}

type ServerMessageFrame struct {
	*frames.SystemFrame
	Data any
}

func NewServerMessageFrame(data any) *ServerMessageFrame {
	return &ServerMessageFrame{SystemFrame: systemFrame("RTVIServerMessageFrame"), Data: data}
}

type ServerResponseFrame struct {
	*frames.SystemFrame
	ClientMsg ClientMessage
	Data      any
	Error     string
}

func NewServerResponseFrame(clientMsg ClientMessage, data any) *ServerResponseFrame {
	return &ServerResponseFrame{SystemFrame: systemFrame("RTVIServerResponseFrame"), ClientMsg: clientMsg, Data: data}
}

func NewServerErrorResponseFrame(clientMsg ClientMessage, err string) *ServerResponseFrame {
	return &ServerResponseFrame{SystemFrame: systemFrame("RTVIServerResponseFrame"), ClientMsg: clientMsg, Error: err}
}

type SendTextFrame struct {
	*frames.SystemFrame
	MsgID  string
	Text   string
	UserID string
}

func NewSendTextFrame(msgID, text, userID string) *SendTextFrame {
	return &SendTextFrame{SystemFrame: systemFrame("RTVISendTextFrame"), MsgID: msgID, Text: text, UserID: userID}
}

type ConfigureObserverFrame struct {
	*frames.SystemFrame
	FunctionCallReportLevel map[string]FunctionCallReportLevel
	VADUserSpeakingEnabled  *bool
	BotLLMMarkerEnabled     *bool
}

func NewConfigureObserverFrame() *ConfigureObserverFrame {
	return &ConfigureObserverFrame{SystemFrame: systemFrame("RTVIConfigureObserverFrame")}
}
