package frames

type TransferCallFrame struct {
	*ControlFrame
	Destination  string
	CallerID     string
	Announcement string
	OnComplete   func(err error)
}

func NewTransferCallFrame(destination string) *TransferCallFrame {
	return &TransferCallFrame{
		ControlFrame: newUninterruptibleControlFrame("TransferCallFrame"),
		Destination:  destination,
	}
}

type TransferCallResultFrame struct {
	*SystemFrame
	Destination string
	Err         error
}

func NewTransferCallResultFrame(destination string, err error) *TransferCallResultFrame {
	return &TransferCallResultFrame{
		SystemFrame: newSystemFrame("TransferCallResultFrame"),
		Destination: destination,
		Err:         err,
	}
}

func (f *TransferCallResultFrame) Succeeded() bool { return f.Err == nil }
