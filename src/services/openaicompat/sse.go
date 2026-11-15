package openaicompat

import (
	"bufio"
	"bytes"
	"errors"
	"io"
)

type SSEReader struct {
	r     *bufio.Reader
	line  []byte
	data  []byte
	event []byte
}

func NewSSEReader(r io.Reader) *SSEReader {
	return &SSEReader{r: bufio.NewReaderSize(r, 32<<10)}
}

func (s *SSEReader) readLine() ([]byte, error) {
	s.line = s.line[:0]
	for {
		chunk, err := s.r.ReadSlice('\n')
		if errors.Is(err, bufio.ErrBufferFull) {
			s.line = append(s.line, chunk...)
			continue
		}
		var line []byte
		if len(s.line) > 0 {
			s.line = append(s.line, chunk...)
			line = s.line
		} else {
			line = chunk
		}
		if err != nil && len(line) == 0 {
			return nil, err
		}
		line = bytes.TrimSuffix(line, []byte("\n"))
		line = bytes.TrimSuffix(line, []byte("\r"))
		return line, nil
	}
}

func (s *SSEReader) Next() (event, data []byte, err error) {
	s.data = s.data[:0]
	s.event = s.event[:0]
	hasData := false
	for {
		line, err := s.readLine()
		if err != nil {
			if errors.Is(err, io.EOF) && hasData {
				return s.event, s.data, nil
			}
			return nil, nil, err
		}
		if len(line) == 0 {
			if hasData {
				return s.event, s.data, nil
			}
			continue
		}
		if line[0] == ':' {
			continue
		}
		field, value := line, []byte(nil)
		if i := bytes.IndexByte(line, ':'); i >= 0 {
			field, value = line[:i], line[i+1:]
			if len(value) > 0 && value[0] == ' ' {
				value = value[1:]
			}
		}
		switch string(field) {
		case "data":
			if hasData {
				s.data = append(s.data, '\n')
			}
			s.data = append(s.data, value...)
			hasData = true
		case "event":
			s.event = append(s.event[:0], value...)
		}
	}
}
