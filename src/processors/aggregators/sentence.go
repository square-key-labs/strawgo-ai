package aggregators

import (
	"context"
	"strings"
	"unicode"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

// SentenceAggregator buffers incoming text frames and emits complete sentences.
// This ensures interruptions occur at sentence boundaries, preventing the bot
// from repeating content after being interrupted during long responses.
//
// Supports both TextFrame and LLMTextFrame inputs.
// Outputs TextFrame for each complete sentence.
//
// Frame flow:
//
//	LLMTextFrame("Hello,") -> (buffered)
//	LLMTextFrame(" world.") -> TextFrame("Hello, world.")
type SentenceAggregator struct {
	*processors.BaseProcessor
	buffer strings.Builder
	mode   TextAggregationMode
}

// NewSentenceAggregator creates a new sentence aggregator processor
// By default, uses TextAggregationModeSentence (buffers until sentence boundaries)
// Pass TextAggregationModeToken to pass text through immediately without buffering
func NewSentenceAggregator(mode ...TextAggregationMode) *SentenceAggregator {
	aggMode := TextAggregationModeSentence // default
	if len(mode) > 0 {
		aggMode = mode[0]
	}
	sa := &SentenceAggregator{
		mode: aggMode,
	}
	sa.BaseProcessor = processors.NewBaseProcessor("SentenceAggregator", sa)
	return sa
}

func (s *SentenceAggregator) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// CRITICAL: Only process DOWNSTREAM frames (from LLM → TTS)
	// Upstream frames (like word timestamps from TTS) must pass through unchanged
	// to avoid mixing TTS metadata with LLM output (causes garbled sentences)
	if direction == frames.Upstream {
		return s.PushFrame(frame, direction)
	}

	// Handle LLMTextFrame (from LLM services) - primary input
	if llmFrame, ok := frame.(*frames.LLMTextFrame); ok {
		// SkipTTS frames (e.g. turn-completion markers from UserTurnCompletionProcessor)
		// must not be buffered into sentences — pass through so assistant aggregator
		// stores them in context and TTS services can check SkipTTS.
		if llmFrame.SkipTTS {
			return s.PushFrame(frame, direction)
		}
		return s.processText(llmFrame.Text)
	}

	// Handle TextFrame - only downstream (e.g., from user aggregator or other sources)
	// Note: Upstream TextFrames (like TTS word timestamps) are passed through above
	if textFrame, ok := frame.(*frames.TextFrame); ok {
		if textFrame.SkipTTS {
			return s.PushFrame(frame, direction)
		}
		return s.processText(textFrame.Text)
	}

	// Handle LLMFullResponseEndFrame - flush any remaining buffer
	if _, ok := frame.(*frames.LLMFullResponseEndFrame); ok {
		if err := s.flushBuffer(); err != nil {
			return err
		}
		return s.PushFrame(frame, direction)
	}

	// Handle InterruptionFrame - clear buffer to discard partial sentences
	// This prevents stale content from being emitted after interruption
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		if s.buffer.Len() > 0 {
			logger.Debug("[SentenceAggregator] Clearing buffer on interruption (%d bytes)", s.buffer.Len())
			s.buffer.Reset()
		}
		return s.PushFrame(frame, direction)
	}

	// Handle EndFrame - flush buffer before ending
	if _, ok := frame.(*frames.EndFrame); ok {
		if err := s.flushBuffer(); err != nil {
			return err
		}
		return s.PushFrame(frame, direction)
	}

	// Pass all other frames through unchanged
	return s.PushFrame(frame, direction)
}

// processText buffers text and emits complete sentences (or passes through in TOKEN mode)
func (s *SentenceAggregator) processText(text string) error {
	// TOKEN mode: pass text through immediately without buffering
	if s.mode == TextAggregationModeToken {
		if text != "" {
			logger.Debug("[SentenceAggregator] TOKEN mode: passing through text: %s", text)
			textFrame := frames.NewTextFrame(text)
			return s.PushFrame(textFrame, frames.Downstream)
		}
		return nil
	}

	// SENTENCE mode: buffer and extract complete sentences
	s.buffer.WriteString(text)

	// Extract complete sentences
	sentences, remainder := extractSentences(s.buffer.String())
	s.buffer.Reset()
	s.buffer.WriteString(remainder)

	// Push complete sentences downstream
	for _, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if sentence != "" {
			logger.Debug("[SentenceAggregator] Emitting sentence: %s", sentence)
			textFrame := frames.NewTextFrame(sentence + " ") // Add space for natural flow
			if err := s.PushFrame(textFrame, frames.Downstream); err != nil {
				return err
			}
		}
	}

	return nil
}

// flushBuffer emits any remaining text in the buffer
func (s *SentenceAggregator) flushBuffer() error {
	if s.buffer.Len() > 0 {
		remainder := strings.TrimSpace(s.buffer.String())
		s.buffer.Reset()
		if remainder != "" {
			logger.Debug("[SentenceAggregator] Flushing remainder: %s", remainder)
			textFrame := frames.NewTextFrame(remainder)
			return s.PushFrame(textFrame, frames.Downstream)
		}
	}
	return nil
}

// Multilingual sentence-ending punctuation
var sentenceEnders = map[rune]bool{
	// Latin script
	'.': true, '!': true, '?': true, ';': true, '…': true,
	// East Asian (Chinese, Japanese, Korean)
	'。': true, '？': true, '！': true, '；': true, '．': true, '｡': true,
	// Indic scripts (Hindi, Sanskrit)
	'।': true, '॥': true,
	// Arabic script
	'؟': true, '؛': true, '۔': true,
	// Thai, Myanmar, Khmer
	'။': true, '។': true,
	// Armenian, Ethiopic
	'։': true, '።': true, '፧': true,
}

// Common abbreviations that shouldn't end sentences
var abbreviations = map[string]bool{
	"Dr.": true, "Mr.": true, "Mrs.": true, "Ms.": true,
	"Jr.": true, "Sr.": true, "vs.": true, "etc.": true,
	"Inc.": true, "Ltd.": true, "Co.": true,
	"Prof.": true, "Gen.": true, "Col.": true,
	"St.": true, "Ave.": true, "Blvd.": true,
	"Jan.": true, "Feb.": true, "Mar.": true, "Apr.": true,
	"Jun.": true, "Jul.": true, "Aug.": true, "Sep.": true,
	"Oct.": true, "Nov.": true, "Dec.": true,
	"U.S.": true, "U.K.": true, "E.U.": true,
	"a.m.": true, "p.m.": true, "e.g.": true, "i.e.": true,
}

// extractSentences splits text into complete sentences and remainder.
// Handles common sentence-ending punctuation while avoiding false positives
// from abbreviations (Dr., Mr., etc.) and numbers ($29.95).
func extractSentences(text string) ([]string, string) {
	var sentences []string
	var currentSentence strings.Builder

	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		currentSentence.WriteRune(r)

		if sentenceEnders[r] {
			current := currentSentence.String()

			// Check for abbreviations
			isAbbreviation := false
			for abbr := range abbreviations {
				if strings.HasSuffix(current, abbr) {
					isAbbreviation = true
					break
				}
			}

			// Check for decimal numbers (e.g., "29.95")
			if r == '.' && i+1 < len(runes) && unicode.IsDigit(runes[i+1]) {
				continue // Not end of sentence
			}

			// Check for currency/number patterns (e.g., "$29.95")
			if r == '.' && i > 0 {
				prevIdx := i - 1
				hasDigitBefore := false
				for prevIdx >= 0 && (unicode.IsDigit(runes[prevIdx]) || runes[prevIdx] == ',' || runes[prevIdx] == '$' || runes[prevIdx] == '€' || runes[prevIdx] == '£') {
					if unicode.IsDigit(runes[prevIdx]) {
						hasDigitBefore = true
					}
					prevIdx--
				}
				if hasDigitBefore && i+1 < len(runes) && unicode.IsDigit(runes[i+1]) {
					continue // Part of a number
				}
			}

			if !isAbbreviation {
				// Check if at end of text or followed by space/newline
				if i == len(runes)-1 {
					// End of text - this is a complete sentence
					sentences = append(sentences, currentSentence.String())
					currentSentence.Reset()
				} else if i+1 < len(runes) && (unicode.IsSpace(runes[i+1]) || runes[i+1] == '\n') {
					// Followed by space - likely end of sentence
					sentences = append(sentences, currentSentence.String())
					currentSentence.Reset()
				}
			}
		}
	}

	// Return sentences and any remaining incomplete text
	return sentences, currentSentence.String()
}
