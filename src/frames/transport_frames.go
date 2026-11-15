package frames

type InputTransportMessageFrame struct {
	*SystemFrame
	Message any
}

func NewInputTransportMessageFrame(message any) *InputTransportMessageFrame {
	return &InputTransportMessageFrame{
		SystemFrame: newSystemFrame("InputTransportMessageFrame"),
		Message:     message,
	}
}

type OutputTransportMessageFrame struct {
	*DataFrame
	Message any
}

func NewOutputTransportMessageFrame(message any) *OutputTransportMessageFrame {
	return &OutputTransportMessageFrame{
		DataFrame: newDataFrame("OutputTransportMessageFrame"),
		Message:   message,
	}
}

type OutputTransportMessageUrgentFrame struct {
	*SystemFrame
	Message any
}

func NewOutputTransportMessageUrgentFrame(message any) *OutputTransportMessageUrgentFrame {
	return &OutputTransportMessageUrgentFrame{
		SystemFrame: newSystemFrame("OutputTransportMessageUrgentFrame"),
		Message:     message,
	}
}

type ClientDisconnectedFrame struct {
	*SystemFrame
}

func NewClientDisconnectedFrame() *ClientDisconnectedFrame {
	return &ClientDisconnectedFrame{SystemFrame: newSystemFrame("ClientDisconnectedFrame")}
}

type OutputTransportReadyFrame struct {
	*ControlFrame
}

func NewOutputTransportReadyFrame() *OutputTransportReadyFrame {
	return &OutputTransportReadyFrame{ControlFrame: newControlFrame("OutputTransportReadyFrame")}
}

type InputTransportStartAudioStreamingFrame struct {
	*ControlFrame
}

func NewInputTransportStartAudioStreamingFrame() *InputTransportStartAudioStreamingFrame {
	return &InputTransportStartAudioStreamingFrame{ControlFrame: newControlFrame("InputTransportStartAudioStreamingFrame")}
}

type VolumeFrame struct {
	*ControlFrame
	Volume float64
}

func NewVolumeFrame(volume float64) *VolumeFrame {
	return &VolumeFrame{
		ControlFrame: newUninterruptibleControlFrame("VolumeFrame"),
		Volume:       volume,
	}
}

type VolumeGainFrame struct {
	*ControlFrame
	Gain float64
}

func NewVolumeGainFrame(gain float64) *VolumeGainFrame {
	return &VolumeGainFrame{
		ControlFrame: newUninterruptibleControlFrame("VolumeGainFrame"),
		Gain:         gain,
	}
}

type FileSourceType string

const (
	FileSourceBytes FileSourceType = "bytes"
	FileSourceURL   FileSourceType = "url"
)

type InputFileRawFrame struct {
	*SystemFrame
	File     string
	Type     FileSourceType
	Filename string
	Format   string
}

func NewInputFileRawFrame(file string, sourceType FileSourceType, filename, format string) *InputFileRawFrame {
	return &InputFileRawFrame{
		SystemFrame: newSystemFrame("InputFileRawFrame"),
		File:        file,
		Type:        sourceType,
		Filename:    filename,
		Format:      format,
	}
}

type UserFileRawFrame struct {
	*SystemFrame
	File            string
	Type            FileSourceType
	Filename        string
	Format          string
	UserID          string
	Text            string
	AppendToContext *bool
	CustomOptions   map[string]any
	RunLLM          *bool
}

func NewUserFileRawFrame(userID, file string, sourceType FileSourceType, filename, format string) *UserFileRawFrame {
	return &UserFileRawFrame{
		SystemFrame: newSystemFrame("UserFileRawFrame"),
		File:        file,
		Type:        sourceType,
		Filename:    filename,
		Format:      format,
		UserID:      userID,
	}
}
