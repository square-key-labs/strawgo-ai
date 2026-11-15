package frames

type FilterControlFrame struct {
	*ControlFrame
}

func NewFilterControlFrame() *FilterControlFrame {
	return &FilterControlFrame{ControlFrame: newControlFrame("FilterControlFrame")}
}

type FilterUpdateSettingsFrame struct {
	*FilterControlFrame
	Settings map[string]any
}

func NewFilterUpdateSettingsFrame(settings map[string]any) *FilterUpdateSettingsFrame {
	return &FilterUpdateSettingsFrame{
		FilterControlFrame: &FilterControlFrame{ControlFrame: newControlFrame("FilterUpdateSettingsFrame")},
		Settings:           settings,
	}
}

type FilterEnableFrame struct {
	*FilterControlFrame
	Enable bool
}

func NewFilterEnableFrame(enable bool) *FilterEnableFrame {
	return &FilterEnableFrame{
		FilterControlFrame: &FilterControlFrame{ControlFrame: newControlFrame("FilterEnableFrame")},
		Enable:             enable,
	}
}

type MixerControlFrame struct {
	*ControlFrame
}

func NewMixerControlFrame() *MixerControlFrame {
	return &MixerControlFrame{ControlFrame: newControlFrame("MixerControlFrame")}
}

type MixerUpdateSettingsFrame struct {
	*MixerControlFrame
	Settings map[string]any
}

func NewMixerUpdateSettingsFrame(settings map[string]any) *MixerUpdateSettingsFrame {
	return &MixerUpdateSettingsFrame{
		MixerControlFrame: &MixerControlFrame{ControlFrame: newControlFrame("MixerUpdateSettingsFrame")},
		Settings:          settings,
	}
}

type MixerEnableFrame struct {
	*MixerControlFrame
	Enable bool
}

func NewMixerEnableFrame(enable bool) *MixerEnableFrame {
	return &MixerEnableFrame{
		MixerControlFrame: &MixerControlFrame{ControlFrame: newControlFrame("MixerEnableFrame")},
		Enable:            enable,
	}
}

type AudioBufferStartRecordingFrame struct {
	*ControlFrame
}

func NewAudioBufferStartRecordingFrame() *AudioBufferStartRecordingFrame {
	return &AudioBufferStartRecordingFrame{ControlFrame: newUninterruptibleControlFrame("AudioBufferStartRecordingFrame")}
}

type AudioBufferStopRecordingFrame struct {
	*ControlFrame
}

func NewAudioBufferStopRecordingFrame() *AudioBufferStopRecordingFrame {
	return &AudioBufferStopRecordingFrame{ControlFrame: newUninterruptibleControlFrame("AudioBufferStopRecordingFrame")}
}
