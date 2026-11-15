package frames

import (
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/turns"
)

var flushCounter uint64

type FrameProcessorPauseFrame struct {
	*ControlFrame
	Processor ProcessorRef
}

func NewFrameProcessorPauseFrame(processor ProcessorRef) *FrameProcessorPauseFrame {
	return &FrameProcessorPauseFrame{
		ControlFrame: newControlFrame("FrameProcessorPauseFrame"),
		Processor:    processor,
	}
}

type FrameProcessorPauseUrgentFrame struct {
	*SystemFrame
	Processor ProcessorRef
}

func NewFrameProcessorPauseUrgentFrame(processor ProcessorRef) *FrameProcessorPauseUrgentFrame {
	return &FrameProcessorPauseUrgentFrame{
		SystemFrame: newSystemFrame("FrameProcessorPauseUrgentFrame"),
		Processor:   processor,
	}
}

type FrameProcessorResumeFrame struct {
	*SystemFrame
	Processor ProcessorRef
}

func NewFrameProcessorResumeFrame(processor ProcessorRef) *FrameProcessorResumeFrame {
	return &FrameProcessorResumeFrame{
		SystemFrame: newSystemFrame("FrameProcessorResumeFrame"),
		Processor:   processor,
	}
}

type FrameProcessorResumeUrgentFrame struct {
	*SystemFrame
	Processor ProcessorRef
}

func NewFrameProcessorResumeUrgentFrame(processor ProcessorRef) *FrameProcessorResumeUrgentFrame {
	return &FrameProcessorResumeUrgentFrame{
		SystemFrame: newSystemFrame("FrameProcessorResumeUrgentFrame"),
		Processor:   processor,
	}
}

type PipelineFlushFrame struct {
	*ControlFrame
	Event     chan struct{}
	Returning bool
	Origin    string
	FlushID   uint64
}

func NewPipelineFlushFrame(origin string) *PipelineFlushFrame {
	return &PipelineFlushFrame{
		ControlFrame: newUninterruptibleControlFrame("PipelineFlushFrame"),
		Event:        make(chan struct{}),
		Origin:       origin,
		FlushID:      atomic.AddUint64(&flushCounter, 1),
	}
}

type ServiceMetadataFrame struct {
	*SystemFrame
	ServiceName        string
	UserTurnStrategies *turns.UserTurnStrategies
}

func newServiceMetadataFrame(name, serviceName string) *ServiceMetadataFrame {
	return &ServiceMetadataFrame{
		SystemFrame: newSystemFrame(name),
		ServiceName: serviceName,
	}
}

func NewServiceMetadataFrame(serviceName string) *ServiceMetadataFrame {
	return newServiceMetadataFrame("ServiceMetadataFrame", serviceName)
}

type ServiceSwitcherRequestMetadataFrame struct {
	*ControlFrame
	Service ProcessorRef
}

func NewServiceSwitcherRequestMetadataFrame(service ProcessorRef) *ServiceSwitcherRequestMetadataFrame {
	return &ServiceSwitcherRequestMetadataFrame{
		ControlFrame: newControlFrame("ServiceSwitcherRequestMetadataFrame"),
		Service:      service,
	}
}

type ServiceSwitcherFrame struct {
	*ControlFrame
}

func NewServiceSwitcherFrame() *ServiceSwitcherFrame {
	return &ServiceSwitcherFrame{ControlFrame: newControlFrame("ServiceSwitcherFrame")}
}

type ManuallySwitchServiceFrame struct {
	*ServiceSwitcherFrame
	Service ProcessorRef
}

func NewManuallySwitchServiceFrame(service ProcessorRef) *ManuallySwitchServiceFrame {
	return &ManuallySwitchServiceFrame{
		ServiceSwitcherFrame: &ServiceSwitcherFrame{ControlFrame: newControlFrame("ManuallySwitchServiceFrame")},
		Service:              service,
	}
}
