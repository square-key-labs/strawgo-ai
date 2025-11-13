package frames

// ControlFrame is the base for control/configuration frames
type ControlFrame struct {
	*BaseFrame
}

func (f *ControlFrame) Category() FrameCategory {
	return ControlCategory
}

// LLMFullResponseStartFrame marks the beginning of an LLM response
type LLMFullResponseStartFrame struct {
	*ControlFrame
}

func NewLLMFullResponseStartFrame() *LLMFullResponseStartFrame {
	return &LLMFullResponseStartFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("LLMFullResponseStartFrame"),
		},
	}
}

// LLMFullResponseEndFrame marks the end of an LLM response
type LLMFullResponseEndFrame struct {
	*ControlFrame
}

func NewLLMFullResponseEndFrame() *LLMFullResponseEndFrame {
	return &LLMFullResponseEndFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("LLMFullResponseEndFrame"),
		},
	}
}

// TTSStartedFrame marks the beginning of TTS synthesis
type TTSStartedFrame struct {
	*ControlFrame
}

func NewTTSStartedFrame() *TTSStartedFrame {
	return &TTSStartedFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("TTSStartedFrame"),
		},
	}
}

// TTSStoppedFrame marks the end of TTS synthesis
type TTSStoppedFrame struct {
	*ControlFrame
}

func NewTTSStoppedFrame() *TTSStoppedFrame {
	return &TTSStoppedFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("TTSStoppedFrame"),
		},
	}
}

// HeartbeatFrame is used for pipeline health monitoring
type HeartbeatFrame struct {
	*ControlFrame
}

func NewHeartbeatFrame() *HeartbeatFrame {
	return &HeartbeatFrame{
		ControlFrame: &ControlFrame{
			BaseFrame: NewBaseFrame("HeartbeatFrame"),
		},
	}
}
