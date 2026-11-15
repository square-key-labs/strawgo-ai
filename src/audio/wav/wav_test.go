package wav

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func pcmData(n int) []byte {
	b := make([]byte, 2*n)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(int16(i*37-500)))
	}
	return b
}

func TestEncodeDecodeRoundTrip(t *testing.T) {
	data := pcmData(321)
	enc := Encode(data, 16000, 1)
	if len(enc) != 44+len(data) {
		t.Fatalf("len %d", len(enc))
	}
	if string(enc[0:4]) != "RIFF" || string(enc[8:16]) != "WAVEfmt " || string(enc[36:40]) != "data" {
		t.Fatal("bad chunk ids")
	}
	if binary.LittleEndian.Uint32(enc[4:8]) != uint32(36+len(data)) || binary.LittleEndian.Uint32(enc[40:44]) != uint32(len(data)) {
		t.Fatal("bad sizes")
	}
	if binary.LittleEndian.Uint32(enc[28:32]) != 32000 || binary.LittleEndian.Uint16(enc[32:34]) != 2 {
		t.Fatal("bad byte rate or block align")
	}
	a, err := Decode(bytes.NewReader(enc))
	if err != nil {
		t.Fatal(err)
	}
	if a.SampleRate != 16000 || a.Channels != 1 || !bytes.Equal(a.Data, data) {
		t.Fatalf("decoded %d Hz %d ch %d bytes", a.SampleRate, a.Channels, len(a.Data))
	}
}

func TestDecodeStereoTruncatesPartialFrame(t *testing.T) {
	data := pcmData(5)
	a, err := Decode(bytes.NewReader(Encode(data, 8000, 2)))
	if err != nil {
		t.Fatal(err)
	}
	if a.Channels != 2 || len(a.Data) != 8 {
		t.Fatalf("channels %d len %d", a.Channels, len(a.Data))
	}
}

func TestDecodeSkipsUnknownChunksAndOddFmt(t *testing.T) {
	data := pcmData(10)
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(0))
	b.WriteString("WAVE")
	b.WriteString("LIST")
	binary.Write(&b, binary.LittleEndian, uint32(3))
	b.Write([]byte{1, 2, 3, 0})
	b.WriteString("fmt ")
	binary.Write(&b, binary.LittleEndian, uint32(18))
	binary.Write(&b, binary.LittleEndian, uint16(1))
	binary.Write(&b, binary.LittleEndian, uint16(1))
	binary.Write(&b, binary.LittleEndian, uint32(24000))
	binary.Write(&b, binary.LittleEndian, uint32(48000))
	binary.Write(&b, binary.LittleEndian, uint16(2))
	binary.Write(&b, binary.LittleEndian, uint16(16))
	binary.Write(&b, binary.LittleEndian, uint16(0))
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(len(data)))
	b.Write(data)
	a, err := Decode(&b)
	if err != nil {
		t.Fatal(err)
	}
	if a.SampleRate != 24000 || !bytes.Equal(a.Data, data) {
		t.Fatalf("decoded %d Hz %d bytes", a.SampleRate, len(a.Data))
	}
}

func TestDecodeRejectsUnsupported(t *testing.T) {
	good := Encode(pcmData(4), 8000, 1)
	notRiff := append([]byte("RIFX"), good[4:]...)
	eightBit := append([]byte(nil), good...)
	binary.LittleEndian.PutUint16(eightBit[34:36], 8)
	float := append([]byte(nil), good...)
	binary.LittleEndian.PutUint16(float[20:22], 3)
	for name, in := range map[string][]byte{"riff": notRiff, "8bit": eightBit, "float": float} {
		if _, err := Decode(bytes.NewReader(in)); !errors.Is(err, ErrFormat) {
			t.Errorf("%s: err %v", name, err)
		}
	}
	if _, err := Decode(bytes.NewReader(good[:20])); err == nil {
		t.Error("truncated header accepted")
	}
	if _, err := Decode(bytes.NewReader(good[:36])); err == nil {
		t.Error("missing data chunk accepted")
	}
}

func TestWriteFileReadFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a.wav")
	data := pcmData(100)
	if err := WriteFile(path, data, 44100, 1); err != nil {
		t.Fatal(err)
	}
	a, err := ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if a.SampleRate != 44100 || !bytes.Equal(a.Data, data) {
		t.Fatal("file round trip mismatch")
	}
	if _, err := ReadFile(filepath.Join(t.TempDir(), "missing.wav")); err == nil {
		t.Fatal("missing file accepted")
	}
}

func TestStreamingWriterPatchesHeader(t *testing.T) {
	path := filepath.Join(t.TempDir(), "s.wav")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	w, err := NewWriter(f, 8000, 2)
	if err != nil {
		t.Fatal(err)
	}
	data := pcmData(40)
	for i := 0; i < len(data); i += 16 {
		if _, err := w.Write(data[i:min(i+16, len(data))]); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal("second close failed")
	}
	if _, err := w.Write(data); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("write after close: %v", err)
	}
	f.Close()
	a, err := ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if a.Channels != 2 || a.SampleRate != 8000 || !bytes.Equal(a.Data, data) {
		t.Fatalf("streamed file decoded %d bytes", len(a.Data))
	}
}
