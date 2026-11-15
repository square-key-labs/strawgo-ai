package frames

type ProcessorRef interface {
	Name() string
}

func newSystemFrame(name string) *SystemFrame {
	return &SystemFrame{BaseFrame: NewBaseFrame(name)}
}

func newControlFrame(name string) *ControlFrame {
	return &ControlFrame{BaseFrame: NewBaseFrame(name)}
}

func newUninterruptibleControlFrame(name string) *ControlFrame {
	return &ControlFrame{BaseFrame: NewUninterruptibleBaseFrame(name)}
}

func newDataFrame(name string) *DataFrame {
	return &DataFrame{BaseFrame: NewBaseFrame(name)}
}

func newUninterruptibleDataFrame(name string) *DataFrame {
	return &DataFrame{BaseFrame: NewUninterruptibleBaseFrame(name)}
}
