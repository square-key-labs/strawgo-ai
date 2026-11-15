package frames

type InputImageRawFrame struct {
	*SystemFrame
	Data   []byte
	Width  int
	Height int
	Format string
}

func NewInputImageRawFrame(data []byte, width, height int, format string) *InputImageRawFrame {
	return &InputImageRawFrame{
		SystemFrame: newSystemFrame("InputImageRawFrame"),
		Data:        data,
		Width:       width,
		Height:      height,
		Format:      format,
	}
}

type OutputImageRawFrame struct {
	*DataFrame
	Data          []byte
	Width         int
	Height        int
	Format        string
	SyncWithAudio bool
}

func NewOutputImageRawFrame(data []byte, width, height int, format string) *OutputImageRawFrame {
	return &OutputImageRawFrame{
		DataFrame: newDataFrame("OutputImageRawFrame"),
		Data:      data,
		Width:     width,
		Height:    height,
		Format:    format,
	}
}

type URLImageRawFrame struct {
	*DataFrame
	Data          []byte
	Width         int
	Height        int
	Format        string
	SyncWithAudio bool
	URL           string
}

func NewURLImageRawFrame(url string, data []byte, width, height int, format string) *URLImageRawFrame {
	return &URLImageRawFrame{
		DataFrame: newDataFrame("URLImageRawFrame"),
		Data:      data,
		Width:     width,
		Height:    height,
		Format:    format,
		URL:       url,
	}
}

type UserImageRequestFrame struct {
	*SystemFrame
	UserID          string
	Text            string
	AppendToContext *bool
	VideoSource     string
	FunctionName    string
	ToolCallID      string
	ResultCallback  any
}

func NewUserImageRequestFrame(userID string) *UserImageRequestFrame {
	return &UserImageRequestFrame{
		SystemFrame: newSystemFrame("UserImageRequestFrame"),
		UserID:      userID,
	}
}

type UserImageRawFrame struct {
	*SystemFrame
	Data            []byte
	Width           int
	Height          int
	Format          string
	UserID          string
	Text            string
	AppendToContext *bool
	Request         *UserImageRequestFrame
	RunLLM          *bool
}

func NewUserImageRawFrame(userID string, data []byte, width, height int, format string) *UserImageRawFrame {
	return &UserImageRawFrame{
		SystemFrame: newSystemFrame("UserImageRawFrame"),
		Data:        data,
		Width:       width,
		Height:      height,
		Format:      format,
		UserID:      userID,
	}
}
