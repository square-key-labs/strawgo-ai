package frames

type TTSSettings struct {
	Model    string
	Voice    string
	Language string
	Speed    *float64
	Extra    map[string]any
}

func (s TTSSettings) Merge(delta TTSSettings) TTSSettings {
	out := s
	if delta.Model != "" {
		out.Model = delta.Model
	}
	if delta.Voice != "" {
		out.Voice = delta.Voice
	}
	if delta.Language != "" {
		out.Language = delta.Language
	}
	if delta.Speed != nil {
		out.Speed = delta.Speed
	}
	out.Extra = mergeExtra(s.Extra, delta.Extra)
	return out
}

type STTSettings struct {
	Model    string
	Language string
	Extra    map[string]any
}

func (s STTSettings) Merge(delta STTSettings) STTSettings {
	out := s
	if delta.Model != "" {
		out.Model = delta.Model
	}
	if delta.Language != "" {
		out.Language = delta.Language
	}
	out.Extra = mergeExtra(s.Extra, delta.Extra)
	return out
}

func mergeExtra(base, delta map[string]any) map[string]any {
	if len(delta) == 0 {
		return base
	}
	merged := make(map[string]any, len(base)+len(delta))
	for k, v := range base {
		merged[k] = v
	}
	for k, v := range delta {
		merged[k] = v
	}
	return merged
}

type VADParams struct {
	Confidence     float32
	StartSecs      float32
	StopSecs       float32
	MinVolume      float32
	HoldConfidence float32
}

type TTSUpdateSettingsFrame struct {
	*ControlFrame
	Delta                 TTSSettings
	Service               ProcessorRef
	ReachInactiveServices bool
}

func NewTTSUpdateSettingsFrame(delta TTSSettings) *TTSUpdateSettingsFrame {
	return &TTSUpdateSettingsFrame{
		ControlFrame: newUninterruptibleControlFrame("TTSUpdateSettingsFrame"),
		Delta:        delta,
	}
}

type STTUpdateSettingsFrame struct {
	*ControlFrame
	Delta                 STTSettings
	Service               ProcessorRef
	ReachInactiveServices bool
}

func NewSTTUpdateSettingsFrame(delta STTSettings) *STTUpdateSettingsFrame {
	return &STTUpdateSettingsFrame{
		ControlFrame: newUninterruptibleControlFrame("STTUpdateSettingsFrame"),
		Delta:        delta,
	}
}

type VADParamsUpdateFrame struct {
	*ControlFrame
	Params VADParams
}

func NewVADParamsUpdateFrame(params VADParams) *VADParamsUpdateFrame {
	return &VADParamsUpdateFrame{
		ControlFrame: newControlFrame("VADParamsUpdateFrame"),
		Params:       params,
	}
}

type SpeechControlParamsFrame struct {
	*SystemFrame
	VADParams  *VADParams
	TurnParams any
}

func NewSpeechControlParamsFrame(vadParams *VADParams, turnParams any) *SpeechControlParamsFrame {
	return &SpeechControlParamsFrame{
		SystemFrame: newSystemFrame("SpeechControlParamsFrame"),
		VADParams:   vadParams,
		TurnParams:  turnParams,
	}
}
