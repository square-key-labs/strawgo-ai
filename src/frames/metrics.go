package frames

import (
	"time"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

type MetricsData interface {
	Base() MetricsBase
	isMetricsData()
}

type MetricsBase struct {
	Processor string
	Model     string
}

func (b MetricsBase) Base() MetricsBase {
	return b
}

type TTFBMetricsData struct {
	MetricsBase
	Value time.Duration
}

type TTFAMetricsData struct {
	MetricsBase
	TTFA           time.Duration
	TTFB           time.Duration
	LeadingSilence time.Duration
}

type TTFATMetricsData struct {
	MetricsBase
	TTFAT        time.Duration
	TTFB         time.Duration
	ThinkingTime time.Duration
}

type ProcessingMetricsData struct {
	MetricsBase
	Value time.Duration
}

type LLMUsageMetricsData struct {
	MetricsBase
	Value llm.Usage
}

type LLMResponseMetricsData struct {
	MetricsBase
	ResponseID    string
	RequestID     string
	ServedModel   string
	StopReason    string
	RawStopReason string
}

type STTUsage struct {
	AudioSeconds float64
}

type STTUsageMetricsData struct {
	MetricsBase
	Value STTUsage
}

type TTSUsageMetricsData struct {
	MetricsBase
	Value int
}

type TextAggregationMetricsData struct {
	MetricsBase
	Value time.Duration
}

type TurnMetricsData struct {
	MetricsBase
	IsComplete        bool
	Probability       float64
	E2EProcessingTime time.Duration
}

func (TTFBMetricsData) isMetricsData()            {}
func (TTFAMetricsData) isMetricsData()            {}
func (TTFATMetricsData) isMetricsData()           {}
func (ProcessingMetricsData) isMetricsData()      {}
func (LLMUsageMetricsData) isMetricsData()        {}
func (LLMResponseMetricsData) isMetricsData()     {}
func (STTUsageMetricsData) isMetricsData()        {}
func (TTSUsageMetricsData) isMetricsData()        {}
func (TextAggregationMetricsData) isMetricsData() {}
func (TurnMetricsData) isMetricsData()            {}

type MetricsFrame struct {
	*SystemFrame
	Data []MetricsData
}

func NewMetricsFrame(data ...MetricsData) *MetricsFrame {
	return &MetricsFrame{
		SystemFrame: newSystemFrame("MetricsFrame"),
		Data:        data,
	}
}
