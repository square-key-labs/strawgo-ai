package transports

import (
	"github.com/square-key-labs/strawgo-ai/src/audio/vad"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type VADAnalyzerHook struct {
	analyzer vad.VADAnalyzer
	rate     int
	buf      []byte
	speaking bool
}

func NewVADAnalyzerHook(analyzer vad.VADAnalyzer) *VADAnalyzerHook {
	return &VADAnalyzerHook{analyzer: analyzer}
}

func (h *VADAnalyzerHook) AnalyzeAudio(f *frames.AudioFrame, emit func(frames.Frame)) {
	if f.SampleRate != h.rate {
		if err := h.analyzer.SetSampleRate(f.SampleRate); err != nil {
			logger.Warn("[VADAnalyzerHook] set sample rate %d: %v", f.SampleRate, err)
			return
		}
		h.rate = f.SampleRate
		h.buf = h.buf[:0]
	}
	need := h.analyzer.NumFramesRequired() * 2
	if need <= 0 {
		return
	}
	h.buf = append(h.buf, f.Data...)
	off := 0
	for len(h.buf)-off >= need {
		state, err := h.analyzer.AnalyzeAudio(h.buf[off : off+need])
		off += need
		if err != nil {
			continue
		}
		switch {
		case state == vad.VADStateSpeaking && !h.speaking:
			h.speaking = true
			emit(frames.NewVADUserStartedSpeakingFrame(h.params().StartSecs))
		case state == vad.VADStateQuiet && h.speaking:
			h.speaking = false
			emit(frames.NewVADUserStoppedSpeakingFrame(h.params().StopSecs))
		}
	}
	h.buf = h.buf[:copy(h.buf, h.buf[off:])]
}

func (h *VADAnalyzerHook) params() struct{ StartSecs, StopSecs float64 } {
	var out struct{ StartSecs, StopSecs float64 }
	if pp, ok := h.analyzer.(interface{ GetParams() vad.VADParams }); ok {
		p := pp.GetParams()
		out.StartSecs, out.StopSecs = float64(p.StartSecs), float64(p.StopSecs)
	}
	return out
}
