package rtvi

import "time"

const (
	ProtocolVersion      = "2.3.0"
	LegacySupportedMajor = 1
	MessageLabel         = "rtvi-ai"
)

const (
	TypeBotReady                  = "bot-ready"
	TypeError                     = "error"
	TypeErrorResponse             = "error-response"
	TypeServerMessage             = "server-message"
	TypeServerResponse            = "server-response"
	TypeBotStartedSpeaking        = "bot-started-speaking"
	TypeBotStoppedSpeaking        = "bot-stopped-speaking"
	TypeBotInterrupted            = "bot-interrupted"
	TypeUserStartedSpeaking       = "user-started-speaking"
	TypeUserStoppedSpeaking       = "user-stopped-speaking"
	TypeVADUserStartedSpeaking    = "vad-user-started-speaking"
	TypeVADUserStoppedSpeaking    = "vad-user-stopped-speaking"
	TypeUserMuteStarted           = "user-mute-started"
	TypeUserMuteStopped           = "user-mute-stopped"
	TypeUserTranscription         = "user-transcription"
	TypeUserInput                 = "user-input"
	TypeUserLLMText               = "user-llm-text"
	TypeBotLLMStarted             = "bot-llm-started"
	TypeBotLLMStopped             = "bot-llm-stopped"
	TypeBotLLMText                = "bot-llm-text"
	TypeBotLLMMarker              = "bot-llm-marker"
	TypeBotTranscription          = "bot-transcription"
	TypeBotTTSStarted             = "bot-tts-started"
	TypeBotTTSStopped             = "bot-tts-stopped"
	TypeBotTTSText                = "bot-tts-text"
	TypeBotOutput                 = "bot-output"
	TypeSTTRawText                = "stt-raw-text"
	TypeLLMRawText                = "llm-raw-text"
	TypeTTSRawText                = "tts-raw-text"
	TypeMetrics                   = "metrics"
	TypeLLMFunctionCallStarted    = "llm-function-call-started"
	TypeLLMFunctionCallInProgress = "llm-function-call-in-progress"
	TypeLLMFunctionCallStopped    = "llm-function-call-stopped"
	TypeUserAudioLevel            = "user-audio-level"
	TypeBotAudioLevel             = "bot-audio-level"
)

type Message struct {
	Label string  `json:"label"`
	Type  string  `json:"type"`
	ID    *string `json:"id,omitempty"`
	Data  any     `json:"data,omitempty"`
}

func (Message) MessageLabel() string { return MessageLabel }

func NewMessage(typ string, data any) Message {
	return Message{Label: MessageLabel, Type: typ, Data: data}
}

func NewResponse(typ, id string, data any) Message {
	return Message{Label: MessageLabel, Type: typ, ID: &id, Data: data}
}

type ClientMessage struct {
	MsgID string
	Type  string
	Data  any
}

type AboutClientData struct {
	Library         string `json:"library"`
	LibraryVersion  string `json:"library_version,omitempty"`
	Platform        string `json:"platform,omitempty"`
	PlatformVersion string `json:"platform_version,omitempty"`
	PlatformDetails any    `json:"platform_details,omitempty"`
}

type Capabilities struct {
	AudioIn  *bool `json:"audio_in,omitempty"`
	AudioOut *bool `json:"audio_out,omitempty"`
	VideoIn  *bool `json:"video_in,omitempty"`
	ScreenIn *bool `json:"screen_in,omitempty"`
	VideoOut *bool `json:"video_out,omitempty"`
	Metrics  *bool `json:"metrics,omitempty"`
}

type BotReadyData struct {
	Version      string        `json:"version"`
	About        any           `json:"about,omitempty"`
	Capabilities *Capabilities `json:"capabilities,omitempty"`
}

type ErrorData struct {
	Error string `json:"error"`
	Fatal bool   `json:"fatal"`
}

type ErrorResponseData struct {
	Error string `json:"error"`
}

type RawServerResponseData struct {
	T string `json:"t"`
	D any    `json:"d,omitempty"`
}

type TextData struct {
	Text string `json:"text"`
}

type UserTranscriptionData struct {
	Text      string `json:"text"`
	UserID    string `json:"user_id"`
	Timestamp string `json:"timestamp"`
	Final     bool   `json:"final"`
}

type Speaker struct {
	Label  string `json:"label"`
	Number int    `json:"number"`
}

type PerceptionLabel struct {
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
}

type PerceptionSignal struct {
	Name   string            `json:"name"`
	Labels []PerceptionLabel `json:"labels"`
	Value  *float64          `json:"value,omitempty"`
}

type UserInputData struct {
	Text       string             `json:"text"`
	InputType  string             `json:"input_type"`
	Timestamp  string             `json:"timestamp"`
	Final      bool               `json:"final"`
	UserID     *string            `json:"user_id,omitempty"`
	MsgID      *string            `json:"msg_id,omitempty"`
	Perception []PerceptionSignal `json:"perception,omitempty"`
	Speaker    *Speaker           `json:"speaker,omitempty"`
}

type STTRawTextData struct {
	Text      string   `json:"text"`
	UserID    string   `json:"user_id"`
	Timestamp string   `json:"timestamp"`
	Final     bool     `json:"final"`
	Speaker   *Speaker `json:"speaker,omitempty"`
}

type BotLLMMarkerData struct {
	Text    string   `json:"text"`
	Kind    *string  `json:"kind,omitempty"`
	Raw     string   `json:"raw"`
	Markers []string `json:"markers"`
}

type SpokenProgress struct {
	AccumulatedText string `json:"accumulated_text"`
	RemainingText   string `json:"remaining_text"`
}

type BotOutputData struct {
	Text           string          `json:"text"`
	OutputType     string          `json:"output_type"`
	TextType       string          `json:"text_type"`
	AggregatedBy   string          `json:"aggregated_by"`
	SegmentID      *int            `json:"segment_id,omitempty"`
	Spoken         *bool           `json:"spoken,omitempty"`
	WillBeSpoken   *bool           `json:"will_be_spoken,omitempty"`
	SpokenStatus   *string         `json:"spoken_status,omitempty"`
	SpokenProgress *SpokenProgress `json:"spoken_progress,omitempty"`
	Delivery       []string        `json:"delivery,omitempty"`
}

type FunctionCallStartedData struct {
	FunctionName *string `json:"function_name,omitempty"`
}

type FunctionCallInProgressData struct {
	ToolCallID   string  `json:"tool_call_id"`
	FunctionName *string `json:"function_name,omitempty"`
	Arguments    any     `json:"arguments,omitempty"`
}

type FunctionCallStoppedData struct {
	ToolCallID   string  `json:"tool_call_id"`
	Cancelled    bool    `json:"cancelled"`
	FunctionName *string `json:"function_name,omitempty"`
	Result       any     `json:"result,omitempty"`
}

type AudioLevelData struct {
	Value float64 `json:"value"`
}

type TimingMetric struct {
	Processor string  `json:"processor"`
	Model     string  `json:"model,omitempty"`
	Value     float64 `json:"value"`
}

type TTFAMetric struct {
	Processor      string  `json:"processor"`
	Model          string  `json:"model,omitempty"`
	TTFA           float64 `json:"ttfa"`
	TTFB           float64 `json:"ttfb"`
	LeadingSilence float64 `json:"leading_silence"`
}

type TTFATMetric struct {
	Processor    string  `json:"processor"`
	Model        string  `json:"model,omitempty"`
	TTFAT        float64 `json:"ttfat"`
	TTFB         float64 `json:"ttfb"`
	ThinkingTime float64 `json:"thinking_time"`
}

type TokenUsage struct {
	PromptTokens             int  `json:"prompt_tokens"`
	CompletionTokens         int  `json:"completion_tokens"`
	TotalTokens              int  `json:"total_tokens"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens,omitempty"`
	ReasoningTokens          *int `json:"reasoning_tokens,omitempty"`
}

type STTUsageValue struct {
	AudioSeconds float64 `json:"audio_seconds"`
}

type STTUsageMetric struct {
	Processor string        `json:"processor"`
	Model     string        `json:"model,omitempty"`
	Value     STTUsageValue `json:"value"`
}

type CharactersMetric struct {
	Processor string `json:"processor"`
	Model     string `json:"model,omitempty"`
	Value     int    `json:"value"`
}

type clientEnvelope struct {
	Label string         `json:"label"`
	Type  string         `json:"type"`
	ID    string         `json:"id"`
	Data  map[string]any `json:"data"`
}

func isoTimestamp(t time.Time) string {
	if t.IsZero() {
		t = time.Now()
	}
	return t.UTC().Format("2006-01-02T15:04:05.000-07:00")
}

func strPtr(s string) *string { return &s }

func boolPtr(b bool) *bool { return &b }
