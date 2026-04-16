package azure

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

const (
	AzureTTSURLTemplate = "https://%s.tts.speech.microsoft.com/cognitiveservices/v1"
	DefaultVoice        = "en-US-JennyNeural"
	DefaultOutputFormat = "riff-16khz-16bit-mono-pcm"
)

// TTSService provides text-to-speech using Azure Cognitive Services
type TTSService struct {
	*processors.BaseProcessor

	subscriptionKey string
	region          string
	voice           string
	outputFormat    string
	httpClient      *http.Client

	started bool
}

// TTSConfig holds configuration for Azure TTS
type TTSConfig struct {
	SubscriptionKey string
	Region          string
	Voice           string
	OutputFormat    string
}

// NewTTSService creates a new Azure TTS service
func NewTTSService(config TTSConfig) *TTSService {
	region := config.Region
	if region == "" {
		region = DefaultRegion
	}

	voice := config.Voice
	if voice == "" {
		voice = DefaultVoice
	}

	outputFormat := config.OutputFormat
	if outputFormat == "" {
		outputFormat = DefaultOutputFormat
	}

	service := &TTSService{
		subscriptionKey: config.SubscriptionKey,
		region:          region,
		voice:           voice,
		outputFormat:    outputFormat,
		httpClient:      &http.Client{},
	}

	service.BaseProcessor = processors.NewBaseProcessor("AzureTTS", service)
	return service
}

func (s *TTSService) Initialize(ctx context.Context) error {
	logger.Debug("[AzureTTS] Service initialized")
	return nil
}

func (s *TTSService) Cleanup() error {
	logger.Debug("[AzureTTS] Service cleaned up")
	return nil
}

func (s *TTSService) SetVoice(voiceID string) {
	s.voice = voiceID
}

// SetModel sets the model (not used for Azure TTS)
// Azure TTS uses voice names which include the model type (e.g., Neural)
func (s *TTSService) SetModel(model string) {
}

func (s *TTSService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.StartFrame:
		s.started = true
		return s.PushFrame(frame, direction)

	case *frames.EndFrame:
		s.started = false
		return s.PushFrame(frame, direction)

	case *frames.InterruptionFrame:
		return s.PushFrame(frame, direction)

	case *frames.TextFrame:
		if f.SkipTTS {
			return s.PushFrame(frame, direction)
		}
		return s.synthesize(ctx, f.Text)

	case *frames.LLMTextFrame:
		if f.SkipTTS {
			return s.PushFrame(frame, direction)
		}
		return s.synthesize(ctx, f.Text)

	default:
		return s.PushFrame(frame, direction)
	}
}

func (s *TTSService) synthesize(ctx context.Context, text string) error {
	if text == "" {
		return nil
	}

	contextID := services.GenerateContextID()

	startFrame := frames.NewTTSStartedFrameWithContext(contextID)
	if err := s.PushFrame(startFrame, frames.Upstream); err != nil {
		return err
	}
	if err := s.PushFrame(startFrame, frames.Downstream); err != nil {
		return err
	}

	ssml := s.buildSSML(text)

	url := fmt.Sprintf(AzureTTSURLTemplate, s.region)
req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader([]byte(ssml)))
if err != nil {
errMsg := fmt.Sprintf("failed to create request: %v", err)
logger.Error("[AzureTTS] %s", errMsg)
s.PushFrame(frames.NewErrorFrame(errors.New(errMsg)), frames.Upstream)
return errors.New(errMsg)
}

	req.Header.Set("Ocp-Apim-Subscription-Key", s.subscriptionKey)
	req.Header.Set("Content-Type", "application/ssml+xml")
	req.Header.Set("X-Microsoft-OutputFormat", s.outputFormat)

resp, err := s.httpClient.Do(req)
if err != nil {
errMsg := fmt.Sprintf("failed to send request: %v", err)
logger.Error("[AzureTTS] %s", errMsg)
s.PushFrame(frames.NewErrorFrame(errors.New(errMsg)), frames.Upstream)
return errors.New(errMsg)
}
	defer resp.Body.Close()

if resp.StatusCode != http.StatusOK {
body, _ := io.ReadAll(resp.Body)
errMsg := fmt.Sprintf("Azure TTS API error (%d): %s", resp.StatusCode, string(body))
logger.Error("[AzureTTS] %s", errMsg)
s.PushFrame(frames.NewErrorFrame(errors.New(errMsg)), frames.Upstream)
return errors.New(errMsg)
}

audioData, err := io.ReadAll(resp.Body)
if err != nil {
errMsg := fmt.Sprintf("failed to read audio: %v", err)
logger.Error("[AzureTTS] %s", errMsg)
s.PushFrame(frames.NewErrorFrame(errors.New(errMsg)), frames.Upstream)
return errors.New(errMsg)
}

	logger.Debug("[AzureTTS] Received audio: %d bytes", len(audioData))

	sampleRate, channels := s.parseOutputFormat()

	audioFrame := frames.NewTTSAudioFrame(audioData, sampleRate, channels)
	audioFrame.SetMetadata("codec", s.getCodec())
	audioFrame.SetMetadata("context_id", contextID)
	if err := s.PushFrame(audioFrame, frames.Downstream); err != nil {
		return err
	}

	stopFrame := frames.NewTTSStoppedFrame()
	stopFrame.SetMetadata("context_id", contextID)
	if err := s.PushFrame(stopFrame, frames.Upstream); err != nil {
		return err
	}

	logger.Debug("[AzureTTS] Synthesis complete for context: %s", contextID)
	return nil
}

func (s *TTSService) buildSSML(text string) string {
	return fmt.Sprintf(`<speak version='1.0' xml:lang='en-US' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts'>
		<voice name='%s'>%s</voice>
	</speak>`, s.voice, text)
}

func (s *TTSService) parseOutputFormat() (sampleRate int, channels int) {
	sampleRate = 16000
	channels = 1

	switch s.outputFormat {
	case "riff-8khz-16bit-mono-pcm":
		sampleRate = 8000
	case "riff-16khz-16bit-mono-pcm":
		sampleRate = 16000
	case "riff-24khz-16bit-mono-pcm":
		sampleRate = 24000
	case "riff-48khz-16bit-mono-pcm":
		sampleRate = 48000
	}

	return sampleRate, channels
}

func (s *TTSService) getCodec() string {
	switch s.outputFormat {
	case "audio-16khz-32kbitrate-mono-mp3",
		"audio-16khz-64kbitrate-mono-mp3",
		"audio-16khz-128kbitrate-mono-mp3",
		"audio-24khz-48kbitrate-mono-mp3",
		"audio-24khz-96kbitrate-mono-mp3",
		"audio-24khz-160kbitrate-mono-mp3",
		"audio-48khz-96kbitrate-mono-mp3",
		"audio-48khz-192kbitrate-mono-mp3":
		return "mp3"
	case "webm-16khz-16bit-mono-opus",
		"webm-24khz-16bit-mono-opus":
		return "opus"
	case "ogg-16khz-16bit-mono-opus",
		"ogg-24khz-16bit-mono-opus",
		"ogg-48khz-16bit-mono-opus":
		return "opus"
	default:
		return "linear16"
	}
}
