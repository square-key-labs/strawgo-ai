package mixer

import (
	"encoding/binary"
	"fmt"
	"sort"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/audio/pcm"
	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/audio/wav"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type Mixer interface {
	Start(sampleRate int) error
	Stop() error
	ProcessFrame(frame frames.Frame) error
	Mix(dst, audio []byte) []byte
}

type SoundfileOption func(*SoundfileMixer)

func WithVolume(v float64) SoundfileOption { return func(m *SoundfileMixer) { m.volume = v } }
func WithLoop(loop bool) SoundfileOption   { return func(m *SoundfileMixer) { m.loop = loop } }
func WithMixing(on bool) SoundfileOption   { return func(m *SoundfileMixer) { m.mixing = on } }

type SoundfileMixer struct {
	files map[string]string

	mu      sync.Mutex
	sounds  map[string][]int16
	current string
	pos     int
	volume  float64
	loop    bool
	mixing  bool
	rate    int
}

func NewSoundfileMixer(soundFiles map[string]string, defaultSound string, opts ...SoundfileOption) *SoundfileMixer {
	files := make(map[string]string, len(soundFiles))
	for k, v := range soundFiles {
		files[k] = v
	}
	m := &SoundfileMixer{
		files:   files,
		sounds:  map[string][]int16{},
		current: defaultSound,
		volume:  0.4,
		loop:    true,
		mixing:  true,
	}
	for _, o := range opts {
		o(m)
	}
	return m
}

func (m *SoundfileMixer) Start(sampleRate int) error {
	if sampleRate <= 0 {
		return fmt.Errorf("mixer: invalid sample rate %d", sampleRate)
	}
	names := make([]string, 0, len(m.files))
	for name := range m.files {
		names = append(names, name)
	}
	sort.Strings(names)
	loaded := make(map[string][]int16, len(names))
	var firstErr error
	for _, name := range names {
		s, err := loadSound(m.files[name], sampleRate)
		if err != nil {
			logger.Error("[SoundfileMixer] unable to load %q from %s: %v", name, m.files[name], err)
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		loaded[name] = s
	}
	m.mu.Lock()
	m.rate = sampleRate
	m.sounds = loaded
	m.pos = 0
	m.mu.Unlock()
	if len(loaded) == 0 && firstErr != nil {
		return firstErr
	}
	return nil
}

func loadSound(path string, rate int) ([]int16, error) {
	a, err := wav.ReadFile(path)
	if err != nil {
		return nil, err
	}
	mono := pcm.ToMono(nil, a.Data, a.Channels)
	samples := pcm.ToInt16(nil, mono)
	if a.SampleRate != rate {
		return resample.Resample(samples, a.SampleRate, rate)
	}
	return samples, nil
}

func (m *SoundfileMixer) Stop() error { return nil }

func (m *SoundfileMixer) ProcessFrame(frame frames.Frame) error {
	switch f := frame.(type) {
	case *frames.MixerEnableFrame:
		m.mu.Lock()
		m.mixing = f.Enable
		m.mu.Unlock()
	case *frames.MixerUpdateSettingsFrame:
		m.mu.Lock()
		defer m.mu.Unlock()
		for k, v := range f.Settings {
			switch k {
			case "sound":
				name, ok := v.(string)
				if !ok {
					return fmt.Errorf("mixer: sound must be a string")
				}
				if _, ok := m.files[name]; !ok {
					return fmt.Errorf("mixer: sound %q is not available", name)
				}
				m.current = name
				m.pos = 0
			case "volume":
				switch n := v.(type) {
				case float64:
					m.volume = n
				case float32:
					m.volume = float64(n)
				case int:
					m.volume = float64(n)
				default:
					return fmt.Errorf("mixer: volume must be a number")
				}
			case "loop":
				b, ok := v.(bool)
				if !ok {
					return fmt.Errorf("mixer: loop must be a bool")
				}
				m.loop = b
			}
		}
	}
	return nil
}

func (m *SoundfileMixer) Mix(dst, audio []byte) []byte {
	n := len(audio) / 2
	l := len(dst)
	dst = pcm.Grow(dst, 2*n)
	out := dst[l:]
	copy(out, audio[:2*n])
	m.mu.Lock()
	defer m.mu.Unlock()
	sound := m.sounds[m.current]
	if !m.mixing || len(sound) == 0 || n == 0 {
		return dst
	}
	if m.pos+n > len(sound) && !m.loop {
		return dst
	}
	g := int32(m.volume * 65536)
	for i := 0; i < n; i++ {
		if m.pos >= len(sound) {
			m.pos = 0
		}
		v := int32(int16(binary.LittleEndian.Uint16(out[2*i:]))) + int32((int64(sound[m.pos])*int64(g))>>16)
		m.pos++
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		binary.LittleEndian.PutUint16(out[2*i:], uint16(int16(v)))
	}
	return dst
}
