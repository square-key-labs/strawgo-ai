package whisper

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func TestNewWhisperSTTService(t *testing.T) {
	service := NewWhisperSTTService("test-api-key")

	if service == nil {
		t.Fatal("Expected service to be created")
	}

	if service.apiKey != "test-api-key" {
		t.Errorf("Expected API key 'test-api-key', got '%s'", service.apiKey)
	}

	if service.model != DefaultModel {
		t.Errorf("Expected model %s, got %s", DefaultModel, service.model)
	}

	if service.sampleRate != DefaultSampleRate {
		t.Errorf("Expected sample rate %d, got %d", DefaultSampleRate, service.sampleRate)
	}

	if service.channels != DefaultChannels {
		t.Errorf("Expected channels %d, got %d", DefaultChannels, service.channels)
	}
}

func TestNewWhisperSTTServiceWithConfig(t *testing.T) {
	config := WhisperSTTConfig{
		APIKey:     "custom-key",
		Model:      "custom-model",
		Language:   "es",
		SampleRate: 24000,
		Channels:   2,
	}

	service := NewWhisperSTTServiceWithConfig(config)

	if service.apiKey != "custom-key" {
		t.Errorf("Expected API key 'custom-key', got '%s'", service.apiKey)
	}

	if service.model != "custom-model" {
		t.Errorf("Expected model 'custom-model', got '%s'", service.model)
	}

	if service.language != "es" {
		t.Errorf("Expected language 'es', got '%s'", service.language)
	}

	if service.sampleRate != 24000 {
		t.Errorf("Expected sample rate 24000, got %d", service.sampleRate)
	}

	if service.channels != 2 {
		t.Errorf("Expected channels 2, got %d", service.channels)
	}
}

func TestSetLanguage(t *testing.T) {
	service := NewWhisperSTTService("test-api-key")
	service.SetLanguage("fr")

	if service.language != "fr" {
		t.Errorf("Expected language 'fr', got '%s'", service.language)
	}
}

func TestSetModel(t *testing.T) {
	service := NewWhisperSTTService("test-api-key")
	service.SetModel("whisper-large")

	if service.model != "whisper-large" {
		t.Errorf("Expected model 'whisper-large', got '%s'", service.model)
	}
}

func TestAudioAccumulation(t *testing.T) {
	service := NewWhisperSTTService("test-api-key")
	ctx := context.Background()

	startFrame := frames.NewUserStartedSpeakingFrame()
	err := service.HandleFrame(ctx, startFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("Failed to handle UserStartedSpeakingFrame: %v", err)
	}

	if !service.accumulating {
		t.Error("Expected accumulation to start")
	}

	audio1 := frames.NewAudioFrame([]byte{1, 2, 3, 4}, 16000, 1)
	err = service.HandleFrame(ctx, audio1, frames.Downstream)
	if err != nil {
		t.Fatalf("Failed to handle AudioFrame: %v", err)
	}

	audio2 := frames.NewAudioFrame([]byte{5, 6, 7, 8}, 16000, 1)
	err = service.HandleFrame(ctx, audio2, frames.Downstream)
	if err != nil {
		t.Fatalf("Failed to handle AudioFrame: %v", err)
	}

	if len(service.audioBuffer) != 8 {
		t.Errorf("Expected buffer size 8, got %d", len(service.audioBuffer))
	}

	expectedData := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	for i, b := range service.audioBuffer {
		if b != expectedData[i] {
			t.Errorf("Expected byte %d at index %d, got %d", expectedData[i], i, b)
		}
	}
}

func TestInterruptionClearsBuffer(t *testing.T) {
	service := NewWhisperSTTService("test-api-key")
	ctx := context.Background()

	service.startAccumulation()
	service.audioBuffer = []byte{1, 2, 3, 4}

	interruptFrame := frames.NewInterruptionFrame()
	err := service.HandleFrame(ctx, interruptFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("Failed to handle InterruptionFrame: %v", err)
	}

	if len(service.audioBuffer) != 0 {
		t.Errorf("Expected buffer to be cleared, got size %d", len(service.audioBuffer))
	}

	if service.accumulating {
		t.Error("Expected accumulation to stop")
	}
}

func TestBufferTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := map[string]string{
			"text": "Timeout transcription",
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	service := NewWhisperSTTService("test-api-key")
	service.apiURL = server.URL
	ctx := context.Background()

	service.startAccumulation()
	service.bufferStart = time.Now().Add(-MaxBufferDuration - time.Second)
	service.audioBuffer = []byte{1, 2, 3, 4}

	audio := frames.NewAudioFrame([]byte{5, 6, 7, 8}, 16000, 1)
	err := service.HandleFrame(ctx, audio, frames.Downstream)
	if err != nil {
		t.Fatalf("Expected timeout to trigger transcription, got error: %v", err)
	}

	if len(service.audioBuffer) != 0 {
		t.Errorf("Expected buffer to be cleared after timeout, got size %d", len(service.audioBuffer))
	}
}

func TestWhisperTranscription(t *testing.T) {
	expectedText := "Hello world"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("Expected POST, got %s", r.Method)
		}

		authHeader := r.Header.Get("Authorization")
		if !strings.HasPrefix(authHeader, "Bearer ") {
			t.Errorf("Expected Bearer token, got %s", authHeader)
		}

		err := r.ParseMultipartForm(10 << 20)
		if err != nil {
			t.Errorf("Failed to parse multipart form: %v", err)
		}

		model := r.FormValue("model")
		if model != DefaultModel {
			t.Errorf("Expected model %s, got %s", DefaultModel, model)
		}

		file, header, err := r.FormFile("file")
		if err != nil {
			t.Errorf("Failed to get file: %v", err)
		}
		defer file.Close()

		if header.Filename != "audio.wav" {
			t.Errorf("Expected filename 'audio.wav', got '%s'", header.Filename)
		}

		wavData, err := io.ReadAll(file)
		if err != nil {
			t.Errorf("Failed to read WAV data: %v", err)
		}

		if len(wavData) < 44 {
			t.Errorf("WAV file too small: %d bytes", len(wavData))
		}

		if string(wavData[0:4]) != "RIFF" {
			t.Errorf("Expected RIFF header, got %s", string(wavData[0:4]))
		}

		if string(wavData[8:12]) != "WAVE" {
			t.Errorf("Expected WAVE format, got %s", string(wavData[8:12]))
		}

		response := map[string]string{
			"text": expectedText,
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	service := NewWhisperSTTService("test-api-key")
	service.apiURL = server.URL

	ctx := context.Background()

	startSpeaking := frames.NewUserStartedSpeakingFrame()
	err := service.HandleFrame(ctx, startSpeaking, frames.Downstream)
	if err != nil {
		t.Fatalf("Failed to handle UserStartedSpeakingFrame: %v", err)
	}

	audio := frames.NewAudioFrame([]byte{1, 2, 3, 4, 5, 6, 7, 8}, 16000, 1)
	err = service.HandleFrame(ctx, audio, frames.Downstream)
	if err != nil {
		t.Fatalf("Failed to handle AudioFrame: %v", err)
	}

	stopSpeaking := frames.NewUserStoppedSpeakingFrame()
	err = service.HandleFrame(ctx, stopSpeaking, frames.Downstream)
	if err != nil {
		t.Fatalf("Failed to handle UserStoppedSpeakingFrame: %v", err)
	}
}

func TestWhisperTranscriptionWithLanguage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		err := r.ParseMultipartForm(10 << 20)
		if err != nil {
			t.Errorf("Failed to parse multipart form: %v", err)
		}

		language := r.FormValue("language")
		if language != "es" {
			t.Errorf("Expected language 'es', got '%s'", language)
		}

		response := map[string]string{
			"text": "Hola mundo",
		}
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	config := WhisperSTTConfig{
		APIKey:   "test-api-key",
		Language: "es",
	}
	service := NewWhisperSTTServiceWithConfig(config)
	service.apiURL = server.URL

	ctx := context.Background()

	startSpeaking := frames.NewUserStartedSpeakingFrame()
	service.HandleFrame(ctx, startSpeaking, frames.Downstream)

	audio := frames.NewAudioFrame([]byte{1, 2, 3, 4}, 16000, 1)
	service.HandleFrame(ctx, audio, frames.Downstream)

	stopSpeaking := frames.NewUserStoppedSpeakingFrame()
	service.HandleFrame(ctx, stopSpeaking, frames.Downstream)
}

func TestEmptyAudioBuffer(t *testing.T) {
	service := NewWhisperSTTService("test-api-key")
	ctx := context.Background()

	startSpeaking := frames.NewUserStartedSpeakingFrame()
	err := service.HandleFrame(ctx, startSpeaking, frames.Downstream)
	if err != nil {
		t.Fatalf("Failed to handle UserStartedSpeakingFrame: %v", err)
	}

	stopSpeaking := frames.NewUserStoppedSpeakingFrame()
	err = service.HandleFrame(ctx, stopSpeaking, frames.Downstream)
	if err != nil {
		t.Fatalf("Failed to handle UserStoppedSpeakingFrame with empty buffer: %v", err)
	}
}

func TestAPIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error": "Invalid request"}`))
	}))
	defer server.Close()

	service := NewWhisperSTTService("test-api-key")
	service.apiURL = server.URL

	ctx := context.Background()

	startSpeaking := frames.NewUserStartedSpeakingFrame()
	service.HandleFrame(ctx, startSpeaking, frames.Downstream)

	audio := frames.NewAudioFrame([]byte{1, 2, 3, 4}, 16000, 1)
	service.HandleFrame(ctx, audio, frames.Downstream)

	stopSpeaking := frames.NewUserStoppedSpeakingFrame()
	service.HandleFrame(ctx, stopSpeaking, frames.Downstream)
}

func TestWAVFileGeneration(t *testing.T) {
	service := NewWhisperSTTService("test-api-key")

	audioData := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	wavData := service.createWAVFile(audioData)

	if len(wavData) < 44 {
		t.Fatalf("WAV file too small: %d bytes", len(wavData))
	}

	if string(wavData[0:4]) != "RIFF" {
		t.Errorf("Expected RIFF header, got %s", string(wavData[0:4]))
	}

	if string(wavData[8:12]) != "WAVE" {
		t.Errorf("Expected WAVE format, got %s", string(wavData[8:12]))
	}

	if string(wavData[12:16]) != "fmt " {
		t.Errorf("Expected 'fmt ' chunk, got %s", string(wavData[12:16]))
	}

	if string(wavData[36:40]) != "data" {
		t.Errorf("Expected 'data' chunk, got %s", string(wavData[36:40]))
	}

	actualAudioData := wavData[44:]
	if len(actualAudioData) != len(audioData) {
		t.Errorf("Expected %d audio bytes, got %d", len(audioData), len(actualAudioData))
	}

	for i, b := range actualAudioData {
		if b != audioData[i] {
			t.Errorf("Expected byte %d at index %d, got %d", audioData[i], i, b)
		}
	}
}

func TestStartFrameLazyInitialization(t *testing.T) {
	service := NewWhisperSTTService("test-api-key")
	ctx := context.Background()

	if service.ctx != nil {
		t.Error("Expected context to be nil before StartFrame")
	}

	startFrame := frames.NewStartFrame()
	err := service.HandleFrame(ctx, startFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("Failed to handle StartFrame: %v", err)
	}

	if !service.started {
		t.Error("Expected service to be started")
	}

	if service.ctx == nil {
		t.Error("Expected context to be initialized after StartFrame")
	}
}

func TestEndFrameCleanup(t *testing.T) {
	service := NewWhisperSTTService("test-api-key")
	ctx := context.Background()

	service.Initialize(ctx)
	service.started = true
	service.audioBuffer = []byte{1, 2, 3, 4}

	endFrame := frames.NewEndFrame()
	err := service.HandleFrame(ctx, endFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("Failed to handle EndFrame: %v", err)
	}

	if service.started {
		t.Error("Expected service to be stopped")
	}

	if len(service.audioBuffer) != 0 {
		t.Errorf("Expected buffer to be cleared, got size %d", len(service.audioBuffer))
	}
}

func TestPassThroughFrames(t *testing.T) {
	service := NewWhisperSTTService("test-api-key")
	ctx := context.Background()

	textFrame := frames.NewTextFrame("test")
	err := service.HandleFrame(ctx, textFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("Failed to handle TextFrame: %v", err)
	}
}
