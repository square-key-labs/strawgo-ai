package tts

import (
	"bufio"
	"encoding/binary"
	"io"
)

type wavSkipper struct {
	r       *bufio.Reader
	skipped bool
}

func SkipWAVHeader(r io.Reader) io.Reader {
	return &wavSkipper{r: bufio.NewReaderSize(r, 4096)}
}

func (w *wavSkipper) Read(p []byte) (int, error) {
	if !w.skipped {
		w.skipped = true
		if err := w.skip(); err != nil {
			return 0, err
		}
	}
	return w.r.Read(p)
}

func (w *wavSkipper) skip() error {
	head, err := w.r.Peek(12)
	if err != nil || string(head[0:4]) != "RIFF" || string(head[8:12]) != "WAVE" {
		return nil
	}
	if _, err := w.r.Discard(12); err != nil {
		return err
	}
	for {
		chunk, err := w.r.Peek(8)
		if err != nil {
			return err
		}
		id := string(chunk[0:4])
		size := int(binary.LittleEndian.Uint32(chunk[4:8]))
		if _, err := w.r.Discard(8); err != nil {
			return err
		}
		if id == "data" {
			return nil
		}
		if _, err := w.r.Discard(size + size%2); err != nil {
			return err
		}
	}
}

type readCloser struct {
	io.Reader
	io.Closer
}

func NewReadCloser(r io.Reader, c io.Closer) io.ReadCloser {
	return readCloser{Reader: r, Closer: c}
}
