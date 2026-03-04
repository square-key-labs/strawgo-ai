package turn

import (
	"math"
	"math/cmplx"
)

// Whisper feature extraction constants
const (
	WhisperSampleRate = 16000
	WhisperNFFT       = 400
	WhisperHopLength  = 160
	WhisperNMels      = 80
)

// WhisperFeatureExtractor extracts log mel spectrograms compatible with Whisper models
type WhisperFeatureExtractor struct {
	sampleRate int
	nFFT       int
	hopLength  int
	nMels      int
	window     []float64
	melFilters [][]float64
}

// NewWhisperFeatureExtractor creates a new feature extractor with Whisper parameters
func NewWhisperFeatureExtractor() *WhisperFeatureExtractor {
	fe := &WhisperFeatureExtractor{
		sampleRate: WhisperSampleRate,
		nFFT:       WhisperNFFT,
		hopLength:  WhisperHopLength,
		nMels:      WhisperNMels,
	}

	// Pre-compute Hann window
	fe.window = hannWindow(WhisperNFFT)

	// Pre-compute mel filterbank
	fe.melFilters = melFilterbank(WhisperNMels, WhisperNFFT, WhisperSampleRate)

	return fe
}

// Extract computes log mel spectrogram features from audio samples
// Input: audio samples normalized to [-1, 1], at 16kHz (REQUIRED - input must be 16kHz)
// Output: log mel spectrogram with shape (n_mels, n_frames) flattened to []float32
// NOTE: Caller must resample input audio to 16kHz before calling this function
// Input: audio samples normalized to [-1, 1], at 16kHz
// Output: log mel spectrogram with shape (n_mels, n_frames) flattened to []float32
func (fe *WhisperFeatureExtractor) Extract(audio []float32, maxLengthSamples int) []float32 {
	// Pad or truncate to exact length
	audioF64 := make([]float64, maxLengthSamples)
	if len(audio) >= maxLengthSamples {
		// Truncate - keep the last maxLengthSamples
		startIdx := len(audio) - maxLengthSamples
		for i := 0; i < maxLengthSamples; i++ {
			audioF64[i] = float64(audio[startIdx+i])
		}
	} else {
		// Pad with zeros at the beginning
		padding := maxLengthSamples - len(audio)
		for i := 0; i < len(audio); i++ {
			audioF64[padding+i] = float64(audio[i])
		}
	}

	// Compute STFT
	stft := fe.stft(audioF64)

	// Compute magnitude squared
	magnitudes := make([][]float64, len(stft))
	for i := range stft {
		magnitudes[i] = make([]float64, len(stft[i]))
		for j := range stft[i] {
			mag := cmplx.Abs(stft[i][j])
			magnitudes[i][j] = mag * mag
		}
	}

	// Apply mel filterbank: mel_spec = filters @ magnitudes
	// magnitudes shape: (n_freqs, n_frames)
	// filters shape: (n_mels, n_freqs)
	// result shape: (n_mels, n_frames)
	nFrames := len(magnitudes[0])
	nFreqs := len(magnitudes)
	melSpec := make([][]float64, fe.nMels)
	for i := 0; i < fe.nMels; i++ {
		melSpec[i] = make([]float64, nFrames)
		for j := 0; j < nFrames; j++ {
			var sum float64
			for k := 0; k < nFreqs && k < len(fe.melFilters[i]); k++ {
				sum += fe.melFilters[i][k] * magnitudes[k][j]
			}
			melSpec[i][j] = sum
		}
	}

	// Log mel spectrogram with Whisper normalization
	// log_spec = torch.clamp(mel_spec, min=1e-10).log10()
	// log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
	// log_spec = (log_spec + 4.0) / 4.0
	logMelSpec := make([][]float64, fe.nMels)
	maxVal := math.Inf(-1)

	for i := 0; i < fe.nMels; i++ {
		logMelSpec[i] = make([]float64, nFrames)
		for j := 0; j < nFrames; j++ {
			val := melSpec[i][j]
			if val < 1e-10 {
				val = 1e-10
			}
			logVal := math.Log10(val)
			logMelSpec[i][j] = logVal
			if logVal > maxVal {
				maxVal = logVal
			}
		}
	}

	// Apply max normalization and scaling
	minVal := maxVal - 8.0
	for i := 0; i < fe.nMels; i++ {
		for j := 0; j < nFrames; j++ {
			val := logMelSpec[i][j]
			if val < minVal {
				val = minVal
			}
			logMelSpec[i][j] = (val + 4.0) / 4.0
		}
	}

	// Flatten to []float32 for ONNX input (n_mels, n_frames)
	result := make([]float32, fe.nMels*nFrames)
	for i := 0; i < fe.nMels; i++ {
		for j := 0; j < nFrames; j++ {
			result[i*nFrames+j] = float32(logMelSpec[i][j])
		}
	}

	return result
}

// stft computes Short-Time Fourier Transform with center padding
// Returns complex spectrogram with shape (n_freqs, n_frames)
// where n_freqs = n_fft/2 + 1
// Uses center=True padding (like torch.stft) and discards the last frame
// to match transformers WhisperFeatureExtractor behavior
func (fe *WhisperFeatureExtractor) stft(audio []float64) [][]complex128 {
	nFreqs := fe.nFFT/2 + 1

	// Apply center padding like torch.stft with center=True
	// Pad n_fft//2 on each side using reflect padding
	halfFFT := fe.nFFT / 2
	paddedLen := len(audio) + 2*halfFFT
	paddedAudio := make([]float64, paddedLen)

	// Reflect padding on the left
	for i := 0; i < halfFFT; i++ {
		idx := halfFFT - i
		if idx < len(audio) {
			paddedAudio[i] = audio[idx]
		}
	}

	// Copy original audio
	copy(paddedAudio[halfFFT:], audio)

	// Reflect padding on the right
	for i := 0; i < halfFFT; i++ {
		srcIdx := len(audio) - 2 - i
		if srcIdx >= 0 {
			paddedAudio[halfFFT+len(audio)+i] = audio[srcIdx]
		}
	}

	// Target number of frames = n_samples // hop_length (like transformers)
	// This matches: nb_max_frames = n_samples // hop_length
	// The STFT produces more frames with padding, but we take exactly this many
	nFrames := len(audio) / fe.hopLength
	if nFrames < 1 {
		nFrames = 1
	}

	result := make([][]complex128, nFreqs)
	for i := range result {
		result[i] = make([]complex128, nFrames)
	}

	// Process each frame
	for frameIdx := 0; frameIdx < nFrames; frameIdx++ {
		startIdx := frameIdx * fe.hopLength

		// Extract windowed frame
		frame := make([]float64, fe.nFFT)
		for i := 0; i < fe.nFFT && startIdx+i < paddedLen; i++ {
			frame[i] = paddedAudio[startIdx+i] * fe.window[i]
		}

		// Compute FFT
		fftResult := fft(frame)

		// Store positive frequencies only
		for freqIdx := 0; freqIdx < nFreqs; freqIdx++ {
			result[freqIdx][frameIdx] = fftResult[freqIdx]
		}
	}

	return result
}

// hannWindow generates a Hann window of given size
func hannWindow(size int) []float64 {
	window := make([]float64, size)
	for i := 0; i < size; i++ {
		window[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(size)))
	}
	return window
}

// melFilterbank creates mel filterbank matrix with Slaney normalization
// Returns matrix of shape (n_mels, n_freqs) where n_freqs = n_fft/2 + 1
// Uses Slaney-style area normalization to achieve approximately constant energy per channel
func melFilterbank(nMels, nFFT, sampleRate int) [][]float64 {
	nFreqs := nFFT/2 + 1
	fMin := 0.0
	fMax := float64(sampleRate) / 2.0

	// Convert to mel scale
	melMin := hzToMel(fMin)
	melMax := hzToMel(fMax)

	// Create mel points (nMels + 2 points for left, center, right of each filter)
	melPoints := make([]float64, nMels+2)
	for i := 0; i < nMels+2; i++ {
		melPoints[i] = melMin + float64(i)*(melMax-melMin)/float64(nMels+1)
	}

	// Convert back to Hz
	hzPoints := make([]float64, nMels+2)
	for i := range hzPoints {
		hzPoints[i] = melToHz(melPoints[i])
	}

	// Convert to FFT bin indices (using floor like librosa)
	binPoints := make([]int, nMels+2)
	for i := range binPoints {
		binPoints[i] = int(math.Floor((float64(nFFT) + 1) * hzPoints[i] / float64(sampleRate)))
	}

	// Create filterbank with triangular filters
	filters := make([][]float64, nMels)
	for i := 0; i < nMels; i++ {
		filters[i] = make([]float64, nFreqs)

		left := binPoints[i]
		center := binPoints[i+1]
		right := binPoints[i+2]

		// Rising edge (left to center)
		for j := left; j < center && j < nFreqs; j++ {
			if center != left {
				filters[i][j] = float64(j-left) / float64(center-left)
			}
		}

		// Falling edge (center to right)
		for j := center; j < right && j < nFreqs; j++ {
			if right != center {
				filters[i][j] = float64(right-j) / float64(right-center)
			}
		}

		// Apply Slaney-style area normalization: 2.0 / (mel_band_width)
		// This divides triangular mel weights by the width of the mel band
		melBandWidth := melPoints[i+2] - melPoints[i]
		if melBandWidth > 0 {
			enorm := 2.0 / melBandWidth
			for j := 0; j < nFreqs; j++ {
				filters[i][j] *= enorm
			}
		}
	}

	return filters
}

// Slaney mel scale constants
const (
	slaneyFMin      = 0.0
	slaneyFSp       = 200.0 / 3.0 // ~66.67 Hz per mel in linear region
	slaneyMinLogHz  = 1000.0
	slaneyMinLogMel = (slaneyMinLogHz - slaneyFMin) / slaneyFSp // ~15.0
	slaneyLogStep   = 0.06875177742094912                       // math.Log(6.4) / 27.0
)

// hzToMel converts frequency in Hz to mel scale using Slaney formula
// This matches librosa's default (htk=False) and transformers WhisperFeatureExtractor
func hzToMel(hz float64) float64 {
	if hz < slaneyMinLogHz {
		// Linear region below 1000 Hz
		return (hz - slaneyFMin) / slaneyFSp
	}
	// Logarithmic region above 1000 Hz
	return slaneyMinLogMel + math.Log(hz/slaneyMinLogHz)/slaneyLogStep
}

// melToHz converts mel scale to frequency in Hz using Slaney formula
func melToHz(mel float64) float64 {
	if mel < slaneyMinLogMel {
		// Linear region
		return slaneyFMin + slaneyFSp*mel
	}
	// Logarithmic region
	return slaneyMinLogHz * math.Exp(slaneyLogStep*(mel-slaneyMinLogMel))
}

// fft computes the Fast Fourier Transform using Cooley-Tukey algorithm
func fft(x []float64) []complex128 {
	n := len(x)

	// Pad to power of 2 if necessary
	paddedN := 1
	for paddedN < n {
		paddedN *= 2
	}

	// Create complex array with padding
	data := make([]complex128, paddedN)
	for i := 0; i < n; i++ {
		data[i] = complex(x[i], 0)
	}

	// Bit reversal permutation
	j := 0
	for i := 0; i < paddedN-1; i++ {
		if i < j {
			data[i], data[j] = data[j], data[i]
		}
		k := paddedN / 2
		for k <= j {
			j -= k
			k /= 2
		}
		j += k
	}

	// Cooley-Tukey FFT
	for size := 2; size <= paddedN; size *= 2 {
		halfSize := size / 2
		step := -2.0 * math.Pi / float64(size)

		for i := 0; i < paddedN; i += size {
			for k := 0; k < halfSize; k++ {
				angle := step * float64(k)
				w := complex(math.Cos(angle), math.Sin(angle))

				even := data[i+k]
				odd := data[i+k+halfSize] * w

				data[i+k] = even + odd
				data[i+k+halfSize] = even - odd
			}
		}
	}

	return data[:n]
}
