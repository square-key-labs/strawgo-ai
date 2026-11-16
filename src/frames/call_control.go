package frames

import (
	"sync"
	"sync/atomic"
)

const (
	transferPending int32 = iota
	transferStarted
	transferCancelled
)

type TransferCallFrame struct {
	*ControlFrame
	Destination  string
	CallerID     string
	Announcement string
	OnComplete   func(err error)

	state    atomic.Int32
	complete sync.Once
}

func NewTransferCallFrame(destination string) *TransferCallFrame {
	return &TransferCallFrame{
		ControlFrame: newUninterruptibleControlFrame("TransferCallFrame"),
		Destination:  destination,
	}
}

func (f *TransferCallFrame) Cancel() bool {
	return f.state.CompareAndSwap(transferPending, transferCancelled) || f.state.Load() == transferCancelled
}

func (f *TransferCallFrame) Claim() bool {
	return f.state.CompareAndSwap(transferPending, transferStarted)
}

func (f *TransferCallFrame) Complete(err error) {
	f.complete.Do(func() {
		if f.OnComplete != nil {
			f.OnComplete(err)
		}
	})
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
