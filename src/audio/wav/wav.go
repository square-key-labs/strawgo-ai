package wav

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

type Audio struct {
	Data       []byte
	SampleRate int
	Channels   int
}

var ErrFormat = errors.New("wav: unsupported format")

func Decode(r io.Reader) (*Audio, error) {
	var hdr [12]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, fmt.Errorf("wav: header: %w", err)
	}
	if string(hdr[0:4]) != "RIFF" || string(hdr[8:12]) != "WAVE" {
		return nil, fmt.Errorf("%w: not RIFF/WAVE", ErrFormat)
	}
	var (
		a         Audio
		haveFmt   bool
		chunkHead [8]byte
	)
	for {
		if _, err := io.ReadFull(r, chunkHead[:]); err != nil {
			return nil, fmt.Errorf("wav: missing data chunk: %w", err)
		}
		id := string(chunkHead[0:4])
		size := int64(binary.LittleEndian.Uint32(chunkHead[4:8]))
		switch id {
		case "fmt ":
			if size < 16 || size > 1<<16 {
				return nil, fmt.Errorf("%w: bad fmt chunk size %d", ErrFormat, size)
			}
			buf := make([]byte, size)
			if _, err := io.ReadFull(r, buf); err != nil {
				return nil, fmt.Errorf("wav: fmt: %w", err)
			}
			format := binary.LittleEndian.Uint16(buf[0:2])
			a.Channels = int(binary.LittleEndian.Uint16(buf[2:4]))
			a.SampleRate = int(binary.LittleEndian.Uint32(buf[4:8]))
			bits := binary.LittleEndian.Uint16(buf[14:16])
			if format == 0xFFFE && size >= 26 {
				format = binary.LittleEndian.Uint16(buf[24:26])
			}
			if format != 1 || bits != 16 || a.Channels < 1 || a.SampleRate <= 0 {
				return nil, fmt.Errorf("%w: format=%d bits=%d channels=%d", ErrFormat, format, bits, a.Channels)
			}
			haveFmt = true
			if size%2 == 1 {
				if _, err := io.CopyN(io.Discard, r, 1); err != nil {
					return nil, err
				}
			}
		case "data":
			if !haveFmt {
				return nil, fmt.Errorf("%w: data before fmt", ErrFormat)
			}
			data, err := io.ReadAll(io.LimitReader(r, size))
			if err != nil {
				return nil, fmt.Errorf("wav: data: %w", err)
			}
			frame := 2 * a.Channels
			a.Data = data[:len(data)/frame*frame]
			return &a, nil
		default:
			if _, err := io.CopyN(io.Discard, r, size+size%2); err != nil {
				return nil, fmt.Errorf("wav: skip %q: %w", id, err)
			}
		}
	}
}

func ReadFile(path string) (*Audio, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Decode(f)
}

func Header(dataLen, sampleRate, channels int) [44]byte {
	var h [44]byte
	copy(h[0:4], "RIFF")
	binary.LittleEndian.PutUint32(h[4:8], uint32(36+dataLen))
	copy(h[8:12], "WAVE")
	copy(h[12:16], "fmt ")
	binary.LittleEndian.PutUint32(h[16:20], 16)
	binary.LittleEndian.PutUint16(h[20:22], 1)
	binary.LittleEndian.PutUint16(h[22:24], uint16(channels))
	binary.LittleEndian.PutUint32(h[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(h[28:32], uint32(sampleRate*channels*2))
	binary.LittleEndian.PutUint16(h[32:34], uint16(channels*2))
	binary.LittleEndian.PutUint16(h[34:36], 16)
	copy(h[36:40], "data")
	binary.LittleEndian.PutUint32(h[40:44], uint32(dataLen))
	return h
}

func Encode(pcm []byte, sampleRate, channels int) []byte {
	h := Header(len(pcm), sampleRate, channels)
	out := make([]byte, 0, len(h)+len(pcm))
	out = append(out, h[:]...)
	return append(out, pcm...)
}

func Write(w io.Writer, pcm []byte, sampleRate, channels int) error {
	h := Header(len(pcm), sampleRate, channels)
	if _, err := w.Write(h[:]); err != nil {
		return err
	}
	_, err := w.Write(pcm)
	return err
}

func WriteFile(path string, pcm []byte, sampleRate, channels int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := Write(f, pcm, sampleRate, channels); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

type Writer struct {
	ws         io.WriteSeeker
	sampleRate int
	channels   int
	n          int64
	closed     bool
}

func NewWriter(ws io.WriteSeeker, sampleRate, channels int) (*Writer, error) {
	h := Header(0, sampleRate, channels)
	if _, err := ws.Write(h[:]); err != nil {
		return nil, err
	}
	return &Writer{ws: ws, sampleRate: sampleRate, channels: channels}, nil
}

func (w *Writer) Write(pcm []byte) (int, error) {
	if w.closed {
		return 0, os.ErrClosed
	}
	n, err := w.ws.Write(pcm)
	w.n += int64(n)
	return n, err
}

func (w *Writer) Close() error {
	if w.closed {
		return nil
	}
	w.closed = true
	if _, err := w.ws.Seek(0, io.SeekStart); err != nil {
		return err
	}
	h := Header(int(w.n), w.sampleRate, w.channels)
	if _, err := w.ws.Write(h[:]); err != nil {
		return err
	}
	_, err := w.ws.Seek(0, io.SeekEnd)
	return err
}
