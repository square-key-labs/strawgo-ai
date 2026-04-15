use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;

// Whisper feature extraction constants
pub const WHISPER_SAMPLE_RATE: usize = 16000;
pub const WHISPER_N_FFT: usize = 400;
pub const WHISPER_HOP_LENGTH: usize = 160;
pub const WHISPER_N_MELS: usize = 80;

// Slaney mel scale constants (must match Go exactly)
const SLANEY_F_MIN: f32 = 0.0;
const SLANEY_F_SP: f32 = 200.0 / 3.0; // ~66.667 Hz per mel in linear region
const SLANEY_MIN_LOG_HZ: f32 = 1000.0;
const SLANEY_MIN_LOG_MEL: f32 = (SLANEY_MIN_LOG_HZ - SLANEY_F_MIN) / SLANEY_F_SP; // ~15.0
// math.Log(6.4) / 27.0 — using the exact constant from Go source
const SLANEY_LOG_STEP: f32 = 0.068_751_777_420_949_12;

// FFT padded size: next power of 2 above WHISPER_N_FFT (400 -> 512)
const FFT_PAD_SIZE: usize = 512;

pub struct WhisperFeatureExtractor {
    window: Vec<f32>,
    mel_filters: Vec<Vec<f32>>, // shape: [WHISPER_N_MELS][n_freqs], n_freqs=201
    fft_planner: FftPlanner<f32>,
    scratch: Vec<Complex<f32>>, // FFT_PAD_SIZE elements, pre-alloc
}

impl WhisperFeatureExtractor {
    pub fn new() -> Self {
        let window = hann_window(WHISPER_N_FFT);
        let mel_filters = mel_filterbank(WHISPER_N_MELS, WHISPER_N_FFT, WHISPER_SAMPLE_RATE);
        let fft_planner = FftPlanner::new();
        let scratch = vec![Complex { re: 0.0, im: 0.0 }; FFT_PAD_SIZE];

        WhisperFeatureExtractor {
            window,
            mel_filters,
            fft_planner,
            scratch,
        }
    }

    /// Extract log mel spectrogram features from audio samples.
    ///
    /// Input: audio samples normalized to [-1, 1], at 16 kHz.
    /// Output: log mel spectrogram (n_mels × n_frames) flattened row-major to Vec<f32>.
    pub fn extract(&mut self, audio: &[f32], max_length_samples: usize) -> Vec<f32> {
        // --- 1. Pad or truncate to exact max_length_samples ---
        // Mirrors Go's Extract: if longer keep the LAST max_length_samples;
        // if shorter zero-pad at the BEGINNING.
        let mut audio_padded = vec![0.0f32; max_length_samples];
        if audio.len() >= max_length_samples {
            let start_idx = audio.len() - max_length_samples;
            audio_padded.copy_from_slice(&audio[start_idx..start_idx + max_length_samples]);
        } else {
            let padding = max_length_samples - audio.len();
            audio_padded[padding..].copy_from_slice(audio);
        }

        // --- 2. Center padding (reflect) — pad n_fft/2 on each side ---
        let half_fft = WHISPER_N_FFT / 2; // 200
        let padded_len = max_length_samples + 2 * half_fft;
        let mut padded_audio = vec![0.0f32; padded_len];

        // Reflect on the left
        for i in 0..half_fft {
            let idx = half_fft - i;
            if idx < max_length_samples {
                padded_audio[i] = audio_padded[idx];
            }
        }

        // Copy original audio
        padded_audio[half_fft..half_fft + max_length_samples].copy_from_slice(&audio_padded);

        // Reflect on the right
        for i in 0..half_fft {
            let src_idx = max_length_samples as isize - 2 - i as isize;
            if src_idx >= 0 {
                padded_audio[half_fft + max_length_samples + i] =
                    audio_padded[src_idx as usize];
            }
        }

        // --- 3. STFT ---
        let n_freqs = WHISPER_N_FFT / 2 + 1; // 201
        let n_frames = (max_length_samples / WHISPER_HOP_LENGTH).max(1);

        // result[freq_idx][frame_idx]
        let mut magnitudes = vec![vec![0.0f32; n_frames]; n_freqs];

        // Prepare FFT of fixed size 512
        let fft = self.fft_planner.plan_fft_forward(FFT_PAD_SIZE);

        for frame_idx in 0..n_frames {
            let start_idx = frame_idx * WHISPER_HOP_LENGTH;

            // Fill scratch with windowed frame + zero padding
            for s in self.scratch.iter_mut() {
                *s = Complex { re: 0.0, im: 0.0 };
            }
            for i in 0..WHISPER_N_FFT {
                let sample_idx = start_idx + i;
                if sample_idx < padded_len {
                    self.scratch[i] = Complex {
                        re: padded_audio[sample_idx] * self.window[i],
                        im: 0.0,
                    };
                }
            }
            // Elements [WHISPER_N_FFT..FFT_PAD_SIZE] are already zero.

            fft.process(&mut self.scratch);

            // Go returns data[:n] (first 400 of the 512 output), then takes [:201].
            // So we use scratch[0..201] for the positive frequencies.
            for freq_idx in 0..n_freqs {
                let c = self.scratch[freq_idx];
                magnitudes[freq_idx][frame_idx] = c.re * c.re + c.im * c.im;
            }
        }

        // --- 4. Apply mel filterbank: mel_spec = filters @ magnitudes ---
        // filters shape: [n_mels][n_freqs]
        // magnitudes shape: [n_freqs][n_frames]
        // mel_spec shape: [n_mels][n_frames]
        let mut mel_spec = vec![vec![0.0f32; n_frames]; WHISPER_N_MELS];
        for mel_idx in 0..WHISPER_N_MELS {
            for frame_idx in 0..n_frames {
                let mut sum = 0.0f32;
                for freq_idx in 0..n_freqs {
                    if freq_idx < self.mel_filters[mel_idx].len() {
                        sum += self.mel_filters[mel_idx][freq_idx]
                            * magnitudes[freq_idx][frame_idx];
                    }
                }
                mel_spec[mel_idx][frame_idx] = sum;
            }
        }

        // --- 5. Log mel + Whisper normalization ---
        let mut log_mel_spec = vec![vec![0.0f32; n_frames]; WHISPER_N_MELS];
        let mut max_val = f32::NEG_INFINITY;

        for mel_idx in 0..WHISPER_N_MELS {
            for frame_idx in 0..n_frames {
                let val = mel_spec[mel_idx][frame_idx].max(1e-10);
                let log_val = val.log10();
                log_mel_spec[mel_idx][frame_idx] = log_val;
                if log_val > max_val {
                    max_val = log_val;
                }
            }
        }

        let min_val = max_val - 8.0;
        for mel_idx in 0..WHISPER_N_MELS {
            for frame_idx in 0..n_frames {
                let val = log_mel_spec[mel_idx][frame_idx].max(min_val);
                log_mel_spec[mel_idx][frame_idx] = (val + 4.0) / 4.0;
            }
        }

        // --- 6. Flatten row-major (n_mels, n_frames) ---
        let mut result = Vec::with_capacity(WHISPER_N_MELS * n_frames);
        for mel_idx in 0..WHISPER_N_MELS {
            for frame_idx in 0..n_frames {
                result.push(log_mel_spec[mel_idx][frame_idx]);
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Hann window
// ---------------------------------------------------------------------------
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos()))
        .collect()
}

// ---------------------------------------------------------------------------
// Slaney mel scale helpers
// ---------------------------------------------------------------------------
fn hz_to_mel(hz: f32) -> f32 {
    if hz < SLANEY_MIN_LOG_HZ {
        (hz - SLANEY_F_MIN) / SLANEY_F_SP
    } else {
        SLANEY_MIN_LOG_MEL + (hz / SLANEY_MIN_LOG_HZ).ln() / SLANEY_LOG_STEP
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    if mel < SLANEY_MIN_LOG_MEL {
        SLANEY_F_MIN + SLANEY_F_SP * mel
    } else {
        SLANEY_MIN_LOG_HZ * (SLANEY_LOG_STEP * (mel - SLANEY_MIN_LOG_MEL)).exp()
    }
}

// ---------------------------------------------------------------------------
// Mel filterbank
// ---------------------------------------------------------------------------
fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: usize) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1; // 201
    let f_min = 0.0f32;
    let f_max = sample_rate as f32 / 2.0; // 8000.0

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // n_mels + 2 equally spaced mel points
    let n_points = n_mels + 2;
    let mel_points: Vec<f32> = (0..n_points)
        .map(|i| mel_min + i as f32 * (mel_max - mel_min) / (n_mels + 1) as f32)
        .collect();

    // Convert mel points back to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz points to FFT bin indices: floor((n_fft+1) * hz / sample_rate)
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| {
            (((n_fft as f32 + 1.0) * hz / sample_rate as f32).floor() as usize)
                .min(n_freqs - 1)
        })
        .collect();

    // Build triangular filters with Slaney area normalization
    let mut filters = vec![vec![0.0f32; n_freqs]; n_mels];
    for i in 0..n_mels {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];

        // Rising edge (left to center)
        if center != left {
            for j in left..center.min(n_freqs) {
                filters[i][j] = (j - left) as f32 / (center - left) as f32;
            }
        }

        // Falling edge (center to right)
        if right != center {
            for j in center..right.min(n_freqs) {
                filters[i][j] = (right - j) as f32 / (right - center) as f32;
            }
        }

        // Slaney area normalization: 2.0 / mel_band_width
        // mel_band_width = mel_points[i+2] - mel_points[i]  (in mel space, not Hz)
        let mel_band_width = mel_points[i + 2] - mel_points[i];
        if mel_band_width > 0.0 {
            let enorm = 2.0 / mel_band_width;
            for j in 0..n_freqs {
                filters[i][j] *= enorm;
            }
        }
    }

    filters
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// Load a binary file of little-endian float32 values.
    fn load_f32_bin(path: &str) -> Vec<f32> {
        let bytes = fs::read(path).expect("failed to read binary fixture file");
        assert_eq!(bytes.len() % 4, 0, "file length not multiple of 4");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Generate a 440 Hz sine wave at 16 kHz for `duration_samples` samples,
    /// normalized to [-1, 1] (matching the Go fixture generator).
    fn sine_440hz(n_samples: usize) -> Vec<f32> {
        (0..n_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / WHISPER_SAMPLE_RATE as f32).sin())
            .collect()
    }

    #[test]
    fn test_mel_sine_440_100ms_parity() {
        // Resolve testdata path relative to crate root
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let fixture_path = manifest_dir
            .join("../testdata/mel_sine_440_100ms.bin")
            .canonicalize()
            .expect("testdata/mel_sine_440_100ms.bin not found");

        let expected = load_f32_bin(fixture_path.to_str().unwrap());
        // 100ms at 16kHz = 1600 samples
        let n_samples = 1600usize;
        assert_eq!(
            expected.len(),
            WHISPER_N_MELS * (n_samples / WHISPER_HOP_LENGTH),
            "fixture size mismatch"
        );

        let audio = sine_440hz(n_samples);

        let mut fe = WhisperFeatureExtractor::new();
        let got = fe.extract(&audio, n_samples);

        assert_eq!(got.len(), expected.len(), "output length mismatch");

        let mut max_diff = 0.0f32;
        let mut fail_count = 0usize;
        for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let diff = (g - e).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff >= 0.01 {
                fail_count += 1;
                if fail_count <= 5 {
                    eprintln!("  idx={idx}: rust={g:.6} go={e:.6} diff={diff:.6}");
                }
            }
        }

        eprintln!("max_diff = {max_diff:.6}  fail_count = {fail_count}");
        assert!(
            fail_count == 0,
            "{fail_count} values exceed tolerance 0.01 (max_diff={max_diff:.6})"
        );
    }
}
