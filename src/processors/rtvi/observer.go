package rtvi

import (
	"strings"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/pcm"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

type FunctionCallReportLevel string

const (
	FunctionCallReportDisabled  FunctionCallReportLevel = "disabled"
	FunctionCallReportNone      FunctionCallReportLevel = "none"
	FunctionCallReportName      FunctionCallReportLevel = "name"
	FunctionCallReportArguments FunctionCallReportLevel = "arguments"
	FunctionCallReportFull      FunctionCallReportLevel = "full"
)

const (
	DefaultOutputSource = "WebSocketOutput"
	WebRTCOutputSource  = "WebRTCOutput"
)

type ObserverParams struct {
	BotOutputEnabled         bool
	BotLLMEnabled            bool
	BotLLMMarkerEnabled      bool
	BotTTSEnabled            bool
	BotSpeakingEnabled       bool
	BotAudioLevelEnabled     bool
	UserLLMEnabled           bool
	UserSpeakingEnabled      bool
	VADUserSpeakingEnabled   bool
	UserMuteEnabled          bool
	UserTranscriptionEnabled bool
	UserInputEnabled         bool
	UserAudioLevelEnabled    bool
	MetricsEnabled           bool
	STTRawTextEnabled        bool
	LLMRawTextEnabled        bool
	TTSRawTextEnabled        bool
	IgnoredSources           []string
	SkipTextTypes            []frames.TextType
	OutputSources            []string
	AudioLevelPeriod         time.Duration
	FunctionCallReportLevel  map[string]FunctionCallReportLevel
}

func DefaultObserverParams() ObserverParams {
	return ObserverParams{
		BotOutputEnabled:         true,
		BotLLMEnabled:            true,
		BotTTSEnabled:            true,
		BotSpeakingEnabled:       true,
		UserLLMEnabled:           true,
		UserSpeakingEnabled:      true,
		UserMuteEnabled:          true,
		UserTranscriptionEnabled: true,
		UserInputEnabled:         true,
		MetricsEnabled:           true,
		OutputSources:            []string{DefaultOutputSource, WebRTCOutputSource},
		AudioLevelPeriod:         150 * time.Millisecond,
		FunctionCallReportLevel:  map[string]FunctionCallReportLevel{"*": FunctionCallReportNone},
	}
}

type Observer struct {
	rtvi   *Processor
	params ObserverParams

	mu                sync.Mutex
	seen              [512]uint64
	seenNext          int
	botTranscription  strings.Builder
	botSpeaking       bool
	queued            []frames.Frame
	lastUserLevel     time.Time
	lastBotLevel      time.Time
	userLevelSmoothed float64
	botLevelSmoothed  float64
}

var _ pipeline.Observer = (*Observer)(nil)

func NewObserver(rtvi *Processor, params *ObserverParams) *Observer {
	p := DefaultObserverParams()
	if params != nil {
		p = *params
	}
	if p.AudioLevelPeriod <= 0 {
		p.AudioLevelPeriod = 150 * time.Millisecond
	}
	if p.FunctionCallReportLevel == nil {
		p.FunctionCallReportLevel = map[string]FunctionCallReportLevel{"*": FunctionCallReportNone}
	}
	if rtvi != nil {
		rtvi.observed.Store(true)
	}
	return &Observer{rtvi: rtvi, params: p}
}

func (o *Observer) OnProcessFrame(pipeline.ProcessFrameEvent) {}

func (o *Observer) Interested(frame frames.Frame, direction frames.FrameDirection, kind pipeline.EventKind) bool {
	if kind != pipeline.PushEvent {
		return false
	}
	switch frame.(type) {
	case *frames.AudioFrame:
		return o.params.UserAudioLevelEnabled
	case *frames.TTSAudioFrame:
		return o.params.BotAudioLevelEnabled
	}
	if frame.GetBroadcastSiblingID() != "" && direction != frames.Downstream {
		return false
	}
	switch frame.(type) {
	case *frames.TTSTextFrame, *frames.AggregatedTextFrame, *frames.AggregatedTextProgressFrame, *frames.InterruptionFrame,
		*frames.UserStartedSpeakingFrame, *frames.UserStoppedSpeakingFrame, *frames.VADUserStartedSpeakingFrame,
		*frames.VADUserStoppedSpeakingFrame, *frames.UserMuteStartedFrame, *frames.UserMuteStoppedFrame,
		*frames.BotStartedSpeakingFrame, *frames.BotStoppedSpeakingFrame, *frames.UserBackchannelFrame,
		*SendTextFrame, *frames.TranscriptionFrame, *frames.InterimTranscriptionFrame, *frames.LLMContextFrame,
		*frames.LLMFullResponseStartFrame, *frames.LLMFullResponseEndFrame, *frames.LLMTextFrame,
		*frames.LLMMarkerResponseFrame, *frames.TTSStartedFrame, *frames.TTSStoppedFrame, *frames.MetricsFrame,
		*frames.FunctionCallsStartedFrame, *frames.FunctionCallInProgressFrame, *frames.FunctionCallCancelFrame,
		*frames.FunctionCallResultFrame, *frames.ExternalFunctionCallsStartedFrame,
		*frames.ExternalFunctionCallInProgressFrame, *frames.ExternalFunctionCallResultFrame,
		*frames.ExternalFunctionCallCancelFrame, *ServerMessageFrame, *ServerResponseFrame, *ConfigureObserverFrame:
		return true
	}
	return false
}

func (o *Observer) OnPipelineStarted() {}

func (o *Observer) OnPipelineStopped() {}

func (o *Observer) send(msg Message) {
	if o.rtvi != nil {
		o.rtvi.Send(msg)
	}
}

func (o *Observer) firstPush(id uint64) bool {
	for _, v := range o.seen {
		if v == id {
			return false
		}
	}
	o.seen[o.seenNext] = id
	o.seenNext = (o.seenNext + 1) % len(o.seen)
	return true
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func (o *Observer) OnPushFrame(ev pipeline.PushFrameEvent) {
	frame := ev.Frame
	switch frame.(type) {
	case *frames.AudioFrame:
		if !o.params.UserAudioLevelEnabled {
			return
		}
	case *frames.TTSAudioFrame:
		if !o.params.BotAudioLevelEnabled {
			return
		}
	}
	if frame.GetBroadcastSiblingID() != "" && ev.Direction != frames.Downstream {
		return
	}
	if len(o.params.IgnoredSources) > 0 && contains(o.params.IgnoredSources, ev.ProcessorName) {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	switch f := frame.(type) {
	case *frames.TTSTextFrame, *frames.AggregatedTextFrame:
		if contains(o.params.OutputSources, ev.ProcessorName) && (o.params.BotOutputEnabled || o.params.BotTTSEnabled || o.params.TTSRawTextEnabled) && o.firstPush(frame.ID()) {
			o.handleAggregated(frame)
		}
		return
	case *frames.AggregatedTextProgressFrame:
		if contains(o.params.OutputSources, ev.ProcessorName) && o.firstPush(frame.ID()) {
			o.handleProgress(f)
		}
		return
	case *frames.InterruptionFrame:
		if o.rtvi != nil && contains(o.params.OutputSources, ev.ProcessorName) {
			o.rtvi.interruptionReachedOutput(f.ID())
		}
	case *frames.AudioFrame:
		o.audioLevel(f.Data, &o.lastUserLevel, &o.userLevelSmoothed, TypeUserAudioLevel)
		return
	case *frames.TTSAudioFrame:
		o.audioLevel(f.Data, &o.lastBotLevel, &o.botLevelSmoothed, TypeBotAudioLevel)
		return
	}
	if !o.firstPush(frame.ID()) {
		return
	}
	o.handle(frame)
}

func (o *Observer) handle(frame frames.Frame) {
	p := &o.params
	switch f := frame.(type) {
	case *frames.UserStartedSpeakingFrame:
		if p.UserSpeakingEnabled {
			o.send(NewMessage(TypeUserStartedSpeaking, nil))
		}
	case *frames.UserStoppedSpeakingFrame:
		if p.UserSpeakingEnabled {
			o.send(NewMessage(TypeUserStoppedSpeaking, nil))
		}
	case *frames.VADUserStartedSpeakingFrame:
		if p.VADUserSpeakingEnabled {
			o.send(NewMessage(TypeVADUserStartedSpeaking, nil))
		}
	case *frames.VADUserStoppedSpeakingFrame:
		if p.VADUserSpeakingEnabled {
			o.send(NewMessage(TypeVADUserStoppedSpeaking, nil))
		}
	case *frames.UserMuteStartedFrame:
		if p.UserMuteEnabled {
			o.send(NewMessage(TypeUserMuteStarted, nil))
		}
	case *frames.UserMuteStoppedFrame:
		if p.UserMuteEnabled {
			o.send(NewMessage(TypeUserMuteStopped, nil))
		}
	case *frames.BotStartedSpeakingFrame:
		if p.BotSpeakingEnabled {
			o.send(NewMessage(TypeBotStartedSpeaking, nil))
		}
		for _, q := range o.queued {
			o.sendAggregated(q)
		}
		clear(o.queued)
		o.queued = o.queued[:0]
		o.botSpeaking = true
	case *frames.BotStoppedSpeakingFrame:
		if p.BotSpeakingEnabled {
			o.send(NewMessage(TypeBotStoppedSpeaking, nil))
		}
		o.botSpeaking = false
	case *frames.InterruptionFrame:
		if p.BotSpeakingEnabled {
			o.send(NewMessage(TypeBotInterrupted, nil))
		}
		kept := o.queued[:0]
		for _, q := range o.queued {
			if q.SurvivesInterruption() {
				kept = append(kept, q)
			}
		}
		clear(o.queued[len(kept):])
		o.queued = kept
	case *frames.UserBackchannelFrame:
		if p.UserInputEnabled {
			o.sendUserInput(f.Text, "backchannel", f.Timestamp, true, strPtr(f.UserID), nil, nil, nil)
		}
	case *SendTextFrame:
		if p.UserInputEnabled {
			var uid *string
			if f.UserID != "" {
				uid = strPtr(f.UserID)
			}
			o.sendUserInput(f.Text, "chat", time.Now(), true, uid, strPtr(f.MsgID), nil, nil)
		}
	case *frames.TranscriptionFrame:
		uid, _ := f.Metadata()["user_id"].(string)
		o.handleTranscription(f.Text, uid, f.Timestamp, f.IsFinal, nil, nil)
	case *frames.InterimTranscriptionFrame:
		o.handleTranscription(f.Text, f.UserID, f.Timestamp, false, f.Speaker, f.Perception)
	case *frames.LLMContextFrame:
		if p.UserLLMEnabled && f.Context != nil {
			o.handleContext(f.Context)
		}
	case *frames.LLMFullResponseStartFrame:
		if p.BotLLMEnabled {
			o.send(NewMessage(TypeBotLLMStarted, nil))
		}
	case *frames.LLMFullResponseEndFrame:
		if p.BotLLMEnabled {
			o.send(NewMessage(TypeBotLLMStopped, nil))
		}
	case *frames.LLMTextFrame:
		o.handleLLMText(f.Text)
	case *frames.LLMMarkerResponseFrame:
		if p.BotLLMMarkerEnabled {
			markers := f.Markers
			if markers == nil {
				markers = []string{}
			}
			var kind *string
			if f.Kind != "" {
				kind = strPtr(f.Kind)
			}
			o.send(NewMessage(TypeBotLLMMarker, BotLLMMarkerData{Text: f.Marker, Kind: kind, Raw: f.Raw, Markers: markers}))
		}
	case *frames.TTSStartedFrame:
		if p.BotTTSEnabled {
			o.send(NewMessage(TypeBotTTSStarted, nil))
		}
	case *frames.TTSStoppedFrame:
		if p.BotTTSEnabled {
			o.send(NewMessage(TypeBotTTSStopped, nil))
		}
	case *frames.MetricsFrame:
		if p.MetricsEnabled {
			o.handleMetrics(f)
		}
	case *frames.FunctionCallsStartedFrame:
		for _, c := range f.FunctionCalls {
			o.reportFunctionCall(phaseStarted, c.FunctionName, c.ToolCallID, nil, nil, false)
		}
	case *frames.FunctionCallInProgressFrame:
		o.reportFunctionCall(phaseInProgress, f.FunctionName, f.ToolCallID, f.Arguments, nil, false)
	case *frames.FunctionCallCancelFrame:
		o.reportFunctionCall(phaseStopped, f.FunctionName, f.ToolCallID, nil, nil, true)
	case *frames.FunctionCallResultFrame:
		if f.Properties == nil || f.Properties.IsFinal {
			o.reportFunctionCall(phaseStopped, f.FunctionName, f.ToolCallID, nil, f.Result, false)
		}
	case *frames.ExternalFunctionCallsStartedFrame:
		for _, c := range f.FunctionCalls {
			o.reportFunctionCall(phaseStarted, c.FunctionName, c.ToolCallID, nil, nil, false)
		}
	case *frames.ExternalFunctionCallInProgressFrame:
		o.reportFunctionCall(phaseInProgress, f.FunctionName, f.ToolCallID, f.Arguments, nil, false)
	case *frames.ExternalFunctionCallResultFrame:
		if f.IsFinal {
			o.reportFunctionCall(phaseStopped, f.FunctionName, f.ToolCallID, nil, f.Result, false)
		}
	case *frames.ExternalFunctionCallCancelFrame:
		o.reportFunctionCall(phaseStopped, f.FunctionName, f.ToolCallID, nil, nil, true)
	case *ServerMessageFrame:
		o.send(NewMessage(TypeServerMessage, f.Data))
	case *ServerResponseFrame:
		if f.Error != "" {
			o.send(NewResponse(TypeErrorResponse, f.ClientMsg.MsgID, ErrorResponseData{Error: f.Error}))
		} else {
			o.send(NewResponse(TypeServerResponse, f.ClientMsg.MsgID, RawServerResponseData{T: f.ClientMsg.Type, D: f.Data}))
		}
	case *ConfigureObserverFrame:
		if f.FunctionCallReportLevel != nil {
			o.params.FunctionCallReportLevel = f.FunctionCallReportLevel
		}
		if f.VADUserSpeakingEnabled != nil {
			o.params.VADUserSpeakingEnabled = *f.VADUserSpeakingEnabled
		}
		if f.BotLLMMarkerEnabled != nil {
			o.params.BotLLMMarkerEnabled = *f.BotLLMMarkerEnabled
		}
	}
}

func (o *Observer) clientAtLeast(major, minor, patch int) bool {
	return o.rtvi == nil || o.rtvi.clientAtLeast(major, minor, patch)
}

func (o *Observer) legacyClient() bool {
	return o.rtvi != nil && o.rtvi.isLegacyClient()
}

func (o *Observer) handleLLMText(text string) {
	if o.params.LLMRawTextEnabled && o.clientAtLeast(2, 2, 0) {
		o.send(NewMessage(TypeLLMRawText, TextData{Text: text}))
	}
	if !o.params.BotLLMEnabled {
		return
	}
	o.send(NewMessage(TypeBotLLMText, TextData{Text: text}))
	o.botTranscription.WriteString(text)
	if endsSentence(o.botTranscription.String()) {
		o.send(NewMessage(TypeBotTranscription, TextData{Text: o.botTranscription.String()}))
		o.botTranscription.Reset()
	}
}

func endsSentence(s string) bool {
	s = strings.TrimRight(s, " \t\r\n\"')]}»”’")
	if s == "" {
		return false
	}
	switch s[len(s)-1] {
	case '.', '!', '?', ';':
		return true
	}
	for _, suf := range []string{"。", "！", "？", "…"} {
		if strings.HasSuffix(s, suf) {
			return true
		}
	}
	return false
}

func speaker(s *frames.TranscriptionSpeaker) *Speaker {
	if s == nil {
		return nil
	}
	return &Speaker{Label: s.Label, Number: s.Number}
}

func perception(signals []frames.PerceptionSignal) []PerceptionSignal {
	if len(signals) == 0 {
		return nil
	}
	out := make([]PerceptionSignal, len(signals))
	for i, s := range signals {
		labels := make([]PerceptionLabel, len(s.Labels))
		for j, l := range s.Labels {
			labels[j] = PerceptionLabel{Label: l.Label, Confidence: l.Confidence}
		}
		out[i] = PerceptionSignal{Name: s.Name, Labels: labels, Value: s.Value}
	}
	return out
}

func (o *Observer) handleTranscription(text, userID string, ts time.Time, final bool, sp *frames.TranscriptionSpeaker, signals []frames.PerceptionSignal) {
	stamp := isoTimestamp(ts)
	if o.params.UserInputEnabled {
		o.sendUserInput(text, "transcription", ts, final, strPtr(userID), nil, signals, sp)
	}
	if o.params.STTRawTextEnabled && o.clientAtLeast(2, 2, 0) {
		o.send(NewMessage(TypeSTTRawText, STTRawTextData{Text: text, UserID: userID, Timestamp: stamp, Final: final, Speaker: speaker(sp)}))
	}
	if o.params.UserTranscriptionEnabled {
		o.send(NewMessage(TypeUserTranscription, UserTranscriptionData{Text: text, UserID: userID, Timestamp: stamp, Final: final}))
	}
}

func (o *Observer) sendUserInput(text, inputType string, ts time.Time, final bool, userID, msgID *string, signals []frames.PerceptionSignal, sp *frames.TranscriptionSpeaker) {
	o.send(NewMessage(TypeUserInput, UserInputData{
		Text:       text,
		InputType:  inputType,
		Timestamp:  isoTimestamp(ts),
		Final:      final,
		UserID:     userID,
		MsgID:      msgID,
		Perception: perception(signals),
		Speaker:    speaker(sp),
	}))
}

func (o *Observer) handleContext(ctx *llm.Context) {
	msgs := ctx.Messages()
	if len(msgs) == 0 {
		return
	}
	last := msgs[len(msgs)-1]
	if last.Role != llm.RoleUser {
		return
	}
	if text := last.Text(); text != "" {
		o.send(NewMessage(TypeUserLLMText, TextData{Text: text}))
	}
}

func (o *Observer) skippedType(t frames.TextType) bool {
	switch t {
	case frames.TextTypeBackchannel:
		if !o.clientAtLeast(2, 2, 0) {
			return true
		}
	case frames.TextTypeExpressionSound, frames.TextTypeExpressionPause:
		if !o.clientAtLeast(2, 3, 0) {
			return true
		}
	}
	for _, s := range o.params.SkipTextTypes {
		if s == t {
			return true
		}
	}
	return false
}

func (o *Observer) handleAggregated(frame frames.Frame) {
	skipTTS := false
	if f, ok := frame.(*frames.AggregatedTextFrame); ok {
		skipTTS = f.SkipTTS
	}
	if o.botSpeaking || (skipTTS && len(o.queued) == 0) {
		o.sendAggregated(frame)
		return
	}
	o.queued = append(o.queued, frame)
}

func (o *Observer) sendAggregated(frame frames.Frame) {
	var text string
	var textType frames.TextType
	var willBeSpoken, isTTS bool
	var segmentID int
	var delivery []string
	switch f := frame.(type) {
	case *frames.TTSTextFrame:
		text, textType, willBeSpoken, isTTS = f.Text, f.TextType, f.WillBeSpoken, true
		segmentID = int(f.ID())
		if f.SegmentID != nil {
			segmentID = *f.SegmentID
		}
		delivery = f.Delivery
	case *frames.AggregatedTextFrame:
		text, textType, willBeSpoken = f.Text, f.TextType, f.WillBeSpoken
		segmentID = int(f.ID())
	default:
		return
	}
	if o.skippedType(textType) || strings.TrimSpace(text) == "" {
		return
	}
	if textType == "" {
		textType = frames.TextTypeSentence
	}
	suppress := !o.legacyClient() && (textType == frames.TextTypeWord || textType == frames.TextTypeToken)
	if o.params.BotOutputEnabled && !suppress {
		status := "new"
		var progress *SpokenProgress
		if willBeSpoken {
			if isTTS {
				status = "completed"
				progress = &SpokenProgress{AccumulatedText: text}
			} else {
				progress = &SpokenProgress{RemainingText: text}
			}
		}
		tt := string(textType)
		o.send(NewMessage(TypeBotOutput, BotOutputData{
			Text:           text,
			OutputType:     tt,
			TextType:       tt,
			AggregatedBy:   tt,
			SegmentID:      &segmentID,
			Spoken:         boolPtr(isTTS),
			WillBeSpoken:   boolPtr(willBeSpoken),
			SpokenStatus:   strPtr(status),
			SpokenProgress: progress,
			Delivery:       delivery,
		}))
	}
	spokenWords := isTTS && textType != frames.TextTypeExpressionSound && textType != frames.TextTypeExpressionPause
	if spokenWords && o.params.TTSRawTextEnabled && o.clientAtLeast(2, 2, 0) {
		o.send(NewMessage(TypeTTSRawText, TextData{Text: text}))
	}
	if spokenWords && o.params.BotTTSEnabled {
		o.send(NewMessage(TypeBotTTSText, TextData{Text: text}))
	}
}

func (o *Observer) handleProgress(f *frames.AggregatedTextProgressFrame) {
	if o.skippedType(f.TextType) || o.legacyClient() || !o.params.BotOutputEnabled {
		return
	}
	status := "in-progress"
	if f.RemainingText == "" {
		status = "completed"
	}
	tt := string(f.TextType)
	seg := f.SegmentID
	o.send(NewMessage(TypeBotOutput, BotOutputData{
		Text:           f.Text,
		OutputType:     tt,
		TextType:       tt,
		AggregatedBy:   tt,
		SegmentID:      &seg,
		WillBeSpoken:   boolPtr(true),
		SpokenStatus:   strPtr(status),
		SpokenProgress: &SpokenProgress{AccumulatedText: f.AccumulatedText, RemainingText: f.RemainingText},
		Delivery:       f.Delivery,
	}))
}

func (o *Observer) audioLevel(data []byte, last *time.Time, smoothed *float64, typ string) {
	*smoothed = pcm.ExpSmoothing(pcm.Volume(data), *smoothed, 0.2)
	now := time.Now()
	if now.Sub(*last) <= o.params.AudioLevelPeriod {
		return
	}
	*last = now
	o.send(NewMessage(typ, AudioLevelData{Value: *smoothed}))
}

type callPhase uint8

const (
	phaseStarted callPhase = iota
	phaseInProgress
	phaseStopped
)

func (o *Observer) reportLevel(name string) FunctionCallReportLevel {
	levels := o.params.FunctionCallReportLevel
	if l, ok := levels[name]; ok {
		return l
	}
	if l, ok := levels["*"]; ok {
		return l
	}
	return FunctionCallReportNone
}

func (o *Observer) reportFunctionCall(phase callPhase, name, toolCallID string, args map[string]any, result any, cancelled bool) {
	level := o.reportLevel(name)
	if level == FunctionCallReportDisabled {
		return
	}
	named := level == FunctionCallReportName || level == FunctionCallReportArguments || level == FunctionCallReportFull
	withArgs := level == FunctionCallReportArguments || level == FunctionCallReportFull
	full := level == FunctionCallReportFull
	var fn *string
	if named {
		fn = strPtr(name)
	}
	switch phase {
	case phaseStarted:
		o.send(NewMessage(TypeLLMFunctionCallStarted, FunctionCallStartedData{FunctionName: fn}))
	case phaseInProgress:
		data := FunctionCallInProgressData{ToolCallID: toolCallID, FunctionName: fn}
		if withArgs && args != nil {
			data.Arguments = args
		}
		o.send(NewMessage(TypeLLMFunctionCallInProgress, data))
	default:
		data := FunctionCallStoppedData{ToolCallID: toolCallID, Cancelled: cancelled, FunctionName: fn}
		if full && !cancelled {
			data.Result = result
		}
		o.send(NewMessage(TypeLLMFunctionCallStopped, data))
	}
}

func seconds(d time.Duration) float64 { return d.Seconds() }

func (o *Observer) handleMetrics(f *frames.MetricsFrame) {
	metrics := make(map[string][]any, 2)
	for _, d := range f.Data {
		switch m := d.(type) {
		case frames.TTFBMetricsData:
			metrics["ttfb"] = append(metrics["ttfb"], TimingMetric{Processor: m.Processor, Model: m.Model, Value: seconds(m.Value)})
		case frames.TTFAMetricsData:
			metrics["ttfa"] = append(metrics["ttfa"], TTFAMetric{Processor: m.Processor, Model: m.Model, TTFA: seconds(m.TTFA), TTFB: seconds(m.TTFB), LeadingSilence: seconds(m.LeadingSilence)})
		case frames.TTFATMetricsData:
			metrics["ttfat"] = append(metrics["ttfat"], TTFATMetric{Processor: m.Processor, Model: m.Model, TTFAT: seconds(m.TTFAT), TTFB: seconds(m.TTFB), ThinkingTime: seconds(m.ThinkingTime)})
		case frames.ProcessingMetricsData:
			metrics["processing"] = append(metrics["processing"], TimingMetric{Processor: m.Processor, Model: m.Model, Value: seconds(m.Value)})
		case frames.LLMUsageMetricsData:
			metrics["tokens"] = append(metrics["tokens"], tokenUsage(m.Value))
		case frames.STTUsageMetricsData:
			metrics["stt_usage"] = append(metrics["stt_usage"], STTUsageMetric{Processor: m.Processor, Model: m.Model, Value: STTUsageValue{AudioSeconds: m.Value.AudioSeconds}})
		case frames.TTSUsageMetricsData:
			metrics["characters"] = append(metrics["characters"], CharactersMetric{Processor: m.Processor, Model: m.Model, Value: m.Value})
		}
	}
	o.send(NewMessage(TypeMetrics, metrics))
}

func tokenUsage(u llm.Usage) TokenUsage {
	t := TokenUsage{PromptTokens: u.PromptTokens, CompletionTokens: u.CompletionTokens, TotalTokens: u.TotalTokens}
	if u.CacheReadTokens > 0 {
		t.CacheReadInputTokens = &u.CacheReadTokens
	}
	if u.CacheCreationTokens > 0 {
		t.CacheCreationInputTokens = &u.CacheCreationTokens
	}
	if u.ReasoningTokens > 0 {
		t.ReasoningTokens = &u.ReasoningTokens
	}
	return t
}
