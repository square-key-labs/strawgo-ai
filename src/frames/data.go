package frames

import "time"

type DataFrame struct {
	*BaseFrame
}

func (f *DataFrame) Category() FrameCategory {
	return DataCategory
}

type TextFrame struct {
	*DataFrame
	Text    string
	SkipTTS bool
}

func NewTextFrame(text string) *TextFrame {
	return &TextFrame{
		DataFrame: &DataFrame{
			BaseFrame: NewBaseFrame("TextFrame"),
		},
		Text: text,
	}
}

type AudioFrame struct {
	*DataFrame
	Data       []byte
	SampleRate int
	Channels   int
}

func NewAudioFrame(data []byte, sampleRate, channels int) *AudioFrame {
	return &AudioFrame{
		DataFrame: &DataFrame{
			BaseFrame: NewUninterruptibleBaseFrame("AudioFrame"),
		},
		Data:       data,
		SampleRate: sampleRate,
		Channels:   channels,
	}
}

type TranscriptionFrame struct {
	*DataFrame
	Text      string
	IsFinal   bool
	Language  string
	Timestamp time.Time
}

func NewTranscriptionFrame(text string, isFinal bool) *TranscriptionFrame {
	newBase := NewBaseFrame
	if isFinal {
		newBase = NewUninterruptibleBaseFrame
	}
	base := newBase("TranscriptionFrame")
	return &TranscriptionFrame{
		DataFrame: &DataFrame{
			BaseFrame: base,
		},
		Text:      text,
		IsFinal:   isFinal,
		Timestamp: time.Now(),
	}
}

func (f *TranscriptionFrame) IsTranscriptionFinal() bool {
	return f.IsFinal
}

type LLMTextFrame struct {
	*DataFrame
	Text    string
	SkipTTS bool
}

func NewLLMTextFrame(text string) *LLMTextFrame {
	return &LLMTextFrame{
		DataFrame: &DataFrame{
			BaseFrame: NewBaseFrame("LLMTextFrame"),
		},
		Text: text,
	}
}

type TTSAudioFrame struct {
	*DataFrame
	Data       []byte
	SampleRate int
	Channels   int
	ContextID  string
}

func NewTTSAudioFrame(data []byte, sampleRate, channels int) *TTSAudioFrame {
	return &TTSAudioFrame{
		DataFrame: &DataFrame{
			BaseFrame: NewBaseFrame("TTSAudioFrame"),
		},
		Data:       data,
		SampleRate: sampleRate,
		Channels:   channels,
	}
}

type STTMetadataFrame struct {
	*DataFrame
	Provider       string        // STT provider name (e.g., "deepgram", "whisper", "azure")
	TTFSP99Latency time.Duration // P99 time-to-final-segment latency
}

type TurnMetricsFrame struct {
	*DataFrame
	ProcessorName string
	StartTime     time.Time
	EndTime       time.Time
	Duration      time.Duration
}

func NewTurnMetricsFrame(processorName string, startTime, endTime time.Time, duration time.Duration) *TurnMetricsFrame {
	return &TurnMetricsFrame{
		DataFrame: &DataFrame{
			BaseFrame: NewBaseFrame("TurnMetricsFrame"),
		},
		ProcessorName: processorName,
		StartTime:     startTime,
		EndTime:       endTime,
		Duration:      duration,
	}
}

func NewSTTMetadataFrame(provider string, p99 time.Duration) *STTMetadataFrame {
	return &STTMetadataFrame{
		DataFrame: &DataFrame{
			BaseFrame: NewBaseFrame("STTMetadataFrame"),
		},
		Provider:       provider,
		TTFSP99Latency: p99,
	}
}

func (f *STTMetadataFrame) GetTTFSP99Latency() time.Duration {
	return f.TTFSP99Latency
}
