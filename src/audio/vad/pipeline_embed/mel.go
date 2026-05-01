// Package pipeline_embed — Whisper-style mel feature extractor for smart-turn.
// Mirrors onnx-worker/src/features.rs exactly.
package pipeline_embed

import (
	"math"
)

// Whisper feature constants (must match Rust impl in onnx-worker/src/features.rs).
const (
	whisperSampleRate = 16000
	whisperNFFT       = 400
	whisperHopLength  = 160
	whisperNMels      = 80
	fftPadSize        = 512 // next power of 2 above WHISPER_N_FFT

	slaneyFMin     = 0.0
	slaneyFSp      = 200.0 / 3.0
	slaneyMinLogHz = 1000.0
)

var (
	slaneyMinLogMel = (slaneyMinLogHz - slaneyFMin) / slaneyFSp // ~15.0
	slaneyLogStep   = math.Log(6.4) / 27.0                      // ~0.06875
)

// MelExtractor produces log-mel spectrogram features for smart-turn.
type MelExtractor struct {
	window     []float32
	melFilters [][]float32 // [nMels][nFreqs]
}

// NewMelExtractor allocates filterbanks once. Reuse across frames.
func NewMelExtractor() *MelExtractor {
	return &MelExtractor{
		window:     hannWindow(whisperNFFT),
		melFilters: melFilterbank(whisperNMels, whisperNFFT, whisperSampleRate),
	}
}

// Extract returns a flat row-major [nMels × nFrames] float32 log-mel
// spectrogram matching the Rust extractor.
//
// audio is the f32-normalized [-1, 1] waveform at 16 kHz.
// maxLengthSamples is the target padded length (e.g. 8 s × 16 kHz = 128000).
func (m *MelExtractor) Extract(audio []float32, maxLengthSamples int) []float32 {
	// 1. Pad/truncate to exact maxLengthSamples (real audio at END, zeros at START).
	audioPadded := make([]float32, maxLengthSamples)
	if len(audio) >= maxLengthSamples {
		start := len(audio) - maxLengthSamples
		copy(audioPadded, audio[start:start+maxLengthSamples])
	} else {
		padding := maxLengthSamples - len(audio)
		copy(audioPadded[padding:], audio)
	}

	// 2. Reflect-pad nFFT/2 on each side.
	halfFFT := whisperNFFT / 2 // 200
	paddedLen := maxLengthSamples + 2*halfFFT
	paddedAudio := make([]float32, paddedLen)

	for i := 0; i < halfFFT; i++ {
		idx := halfFFT - i
		if idx < maxLengthSamples {
			paddedAudio[i] = audioPadded[idx]
		}
	}
	copy(paddedAudio[halfFFT:halfFFT+maxLengthSamples], audioPadded)
	for i := 0; i < halfFFT; i++ {
		srcIdx := maxLengthSamples - 2 - i
		if srcIdx >= 0 {
			paddedAudio[halfFFT+maxLengthSamples+i] = audioPadded[srcIdx]
		}
	}

	// 3. STFT (skip leading silence frames).
	nFreqs := whisperNFFT/2 + 1 // 201
	nFrames := maxLengthSamples / whisperHopLength
	if nFrames < 1 {
		nFrames = 1
	}
	audioUsed := len(audio)
	if audioUsed > maxLengthSamples {
		audioUsed = maxLengthSamples
	}
	paddingSamples := maxLengthSamples - audioUsed
	firstRealFrame := 0
	if paddingSamples+halfFFT > whisperNFFT {
		firstRealFrame = (paddingSamples + halfFFT - whisperNFFT) / whisperHopLength
	}

	// magnitudes[freq][frame]
	magnitudes := make([][]float32, nFreqs)
	for i := range magnitudes {
		magnitudes[i] = make([]float32, nFrames)
	}

	scratch := make([]complex128, fftPadSize)

	for frameIdx := firstRealFrame; frameIdx < nFrames; frameIdx++ {
		startIdx := frameIdx * whisperHopLength
		// Reset scratch.
		for i := range scratch {
			scratch[i] = 0
		}
		for i := 0; i < whisperNFFT; i++ {
			sampleIdx := startIdx + i
			if sampleIdx < paddedLen {
				scratch[i] = complex(float64(paddedAudio[sampleIdx]*m.window[i]), 0)
			}
		}
		// In-place radix-2 FFT (fftPadSize=512).
		fft512(scratch)
		for freqIdx := 0; freqIdx < nFreqs; freqIdx++ {
			c := scratch[freqIdx]
			re := real(c)
			im := imag(c)
			magnitudes[freqIdx][frameIdx] = float32(re*re + im*im)
		}
	}

	// 4. Apply mel filterbank.
	melSpec := make([][]float32, whisperNMels)
	for i := range melSpec {
		melSpec[i] = make([]float32, nFrames)
	}
	for melIdx := 0; melIdx < whisperNMels; melIdx++ {
		for frameIdx := firstRealFrame; frameIdx < nFrames; frameIdx++ {
			var sum float32
			for freqIdx := 0; freqIdx < nFreqs; freqIdx++ {
				sum += m.melFilters[melIdx][freqIdx] * magnitudes[freqIdx][frameIdx]
			}
			melSpec[melIdx][frameIdx] = sum
		}
	}

	// 5. Log-mel + Whisper normalization.
	logMelSpec := make([][]float32, whisperNMels)
	for i := range logMelSpec {
		logMelSpec[i] = make([]float32, nFrames)
	}
	maxVal := float32(math.Log10(1e-10)) // -10.0 default for all-silence

	for melIdx := 0; melIdx < whisperNMels; melIdx++ {
		for frameIdx := firstRealFrame; frameIdx < nFrames; frameIdx++ {
			val := melSpec[melIdx][frameIdx]
			if val < 1e-10 {
				val = 1e-10
			}
			logVal := float32(math.Log10(float64(val)))
			logMelSpec[melIdx][frameIdx] = logVal
			if logVal > maxVal {
				maxVal = logVal
			}
		}
	}

	minVal := maxVal - 8.0
	for melIdx := 0; melIdx < whisperNMels; melIdx++ {
		for frameIdx := firstRealFrame; frameIdx < nFrames; frameIdx++ {
			v := logMelSpec[melIdx][frameIdx]
			if v < minVal {
				v = minVal
			}
			logMelSpec[melIdx][frameIdx] = (v + 4.0) / 4.0
		}
	}

	if firstRealFrame > 0 {
		silenceLog := float32(-10.0)
		if silenceLog < minVal {
			silenceLog = minVal
		}
		silenceVal := (silenceLog + 4.0) / 4.0
		for melIdx := 0; melIdx < whisperNMels; melIdx++ {
			for frameIdx := 0; frameIdx < firstRealFrame; frameIdx++ {
				logMelSpec[melIdx][frameIdx] = silenceVal
			}
		}
	}

	// 6. Flatten row-major.
	out := make([]float32, whisperNMels*nFrames)
	for melIdx := 0; melIdx < whisperNMels; melIdx++ {
		copy(out[melIdx*nFrames:(melIdx+1)*nFrames], logMelSpec[melIdx])
	}
	return out
}

func hannWindow(size int) []float32 {
	w := make([]float32, size)
	for i := 0; i < size; i++ {
		w[i] = float32(0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(size))))
	}
	return w
}

func hzToMel(hz float32) float32 {
	if hz < slaneyMinLogHz {
		return (hz - slaneyFMin) / slaneyFSp
	}
	return float32(slaneyMinLogMel) + float32(math.Log(float64(hz)/slaneyMinLogHz)/slaneyLogStep)
}

func melToHz(mel float32) float32 {
	if mel < float32(slaneyMinLogMel) {
		return slaneyFMin + slaneyFSp*mel
	}
	return slaneyMinLogHz * float32(math.Exp(slaneyLogStep*float64(mel-float32(slaneyMinLogMel))))
}

func melFilterbank(nMels, nFFT, sampleRate int) [][]float32 {
	nFreqs := nFFT/2 + 1
	fMin := float32(0.0)
	fMax := float32(sampleRate) / 2.0
	melMin := hzToMel(fMin)
	melMax := hzToMel(fMax)

	nPoints := nMels + 2
	melPoints := make([]float32, nPoints)
	for i := 0; i < nPoints; i++ {
		melPoints[i] = melMin + float32(i)*(melMax-melMin)/float32(nMels+1)
	}
	hzPoints := make([]float32, nPoints)
	for i, m := range melPoints {
		hzPoints[i] = melToHz(m)
	}
	binPoints := make([]int, nPoints)
	for i, hz := range hzPoints {
		bin := int(math.Floor(float64(float32(nFFT+1)*hz/float32(sampleRate))))
		if bin >= nFreqs {
			bin = nFreqs - 1
		}
		binPoints[i] = bin
	}

	filters := make([][]float32, nMels)
	for i := range filters {
		filters[i] = make([]float32, nFreqs)
	}
	for i := 0; i < nMels; i++ {
		left := binPoints[i]
		center := binPoints[i+1]
		right := binPoints[i+2]
		if center != left {
			endJ := center
			if endJ > nFreqs {
				endJ = nFreqs
			}
			for j := left; j < endJ; j++ {
				filters[i][j] = float32(j-left) / float32(center-left)
			}
		}
		if right != center {
			endJ := right
			if endJ > nFreqs {
				endJ = nFreqs
			}
			for j := center; j < endJ; j++ {
				filters[i][j] = float32(right-j) / float32(right-center)
			}
		}
		melBandWidth := melPoints[i+2] - melPoints[i]
		if melBandWidth > 0.0 {
			enorm := 2.0 / melBandWidth
			for j := 0; j < nFreqs; j++ {
				filters[i][j] *= enorm
			}
		}
	}
	return filters
}

// fft512 is an in-place radix-2 Cooley-Tukey FFT for n=512. Generalised to any
// power-of-two n actually — we only call it with 512 in this package.
func fft512(x []complex128) {
	n := len(x)
	// Bit-reversal permutation.
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			x[i], x[j] = x[j], x[i]
		}
	}
	// Cooley-Tukey butterflies.
	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
		// Forward FFT: angle = -2π/size.
		angleStep := -2 * math.Pi / float64(size)
		for start := 0; start < n; start += size {
			for k := 0; k < half; k++ {
				angle := angleStep * float64(k)
				w := complex(math.Cos(angle), math.Sin(angle))
				t := w * x[start+k+half]
				u := x[start+k]
				x[start+k] = u + t
				x[start+k+half] = u - t
			}
		}
	}
}
