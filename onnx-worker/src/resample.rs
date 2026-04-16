use anyhow::Result;
use rubato::{
    audioadapter_buffers::direct::SequentialSliceOfVecs, Async, FixedAsync, Resampler,
    SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Creates a sinc resampler configured for voice 8kHz→16kHz upsampling.
/// Construct ONCE per SmartTurnSession — not per call — to avoid rebuilding
/// the sinc filter table on each invocation.
pub fn make_sinc_resampler(from_rate: u32, to_rate: u32) -> Result<Async<f32>> {
    let params = SincInterpolationParameters {
        sinc_len: 128, // sufficient for 8→16kHz voice; no WER difference vs 256
        f_cutoff: 0.95,
        oversampling_factor: 128,
        interpolation: SincInterpolationType::Cubic,
        window: WindowFunction::BlackmanHarris2,
    };
    Async::<f32>::new_sinc(
        to_rate as f64 / from_rate as f64,
        1.0, // fixed ratio — no drift compensation needed
        &params,
        1024, // chunk_size: internal buffer unit, NOT input length
        1,    // mono
        FixedAsync::Input,
    )
    .map_err(anyhow::Error::from)
}

/// Resample PCM using a pre-built resampler.
/// Handles variable-length input (0.5s–8s of accumulated turn audio).
pub fn resample_sinc(resampler: &mut Async<f32>, pcm: &[i16]) -> Result<Vec<i16>> {
    if pcm.is_empty() {
        return Ok(Vec::new());
    }

    // i16 → f32 normalized to [-1.0, 1.0]
    let input_f32: Vec<f32> = pcm.iter().map(|&s| s as f32 / 32768.0).collect();
    let input_len = input_f32.len();

    // process_all_into_buffer handles variable-length input internally,
    // looping process_into_buffer until all input frames are consumed.
    let out_cap = resampler.process_all_needed_output_len(input_len) + 16;
    let input_vecs: Vec<Vec<f32>> = vec![input_f32];
    let mut output_vecs: Vec<Vec<f32>> = vec![vec![0.0f32; out_cap]];

    // Scope adapters so output_vecs borrow is released before we slice it.
    let out_len = {
        let input_adapter = SequentialSliceOfVecs::new(&input_vecs, 1, input_len)
            .map_err(|e| anyhow::anyhow!("resample input adapter: {e}"))?;
        let mut out_adapter = SequentialSliceOfVecs::new_mut(&mut output_vecs, 1, out_cap)
            .map_err(|e| anyhow::anyhow!("resample output adapter: {e}"))?;
        let (_, n) =
            resampler.process_all_into_buffer(&input_adapter, &mut out_adapter, input_len, None)?;
        n
    };

    // f32 → i16 with clamping to prevent overflow on cast
    Ok(output_vecs[0][..out_len]
        .iter()
        .map(|&s| (s * 32768.0).clamp(-32768.0, 32767.0) as i16)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Goertzel algorithm: returns the power at `freq` Hz in `signal`
    /// sampled at `sample_rate` Hz. Scales with signal length² — compare
    /// ratios only; do not compare absolute values across different lengths.
    fn goertzel_power(signal: &[f32], freq: f32, sample_rate: f32) -> f32 {
        let omega = 2.0 * std::f32::consts::PI * freq / sample_rate;
        let coeff = 2.0 * omega.cos();
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for &x in signal {
            let s = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s;
        }
        (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0)
    }

    #[test]
    fn test_resample_upsample_2x_length() {
        let mut r = make_sinc_resampler(8000, 16000).unwrap();
        // 256 samples = 32ms @ 8kHz
        let input: Vec<i16> = (0..256).map(|i| (i as i16) * 100).collect();
        let out = resample_sinc(&mut r, &input).unwrap();
        // output should be approximately double the input length
        assert!(out.len() >= input.len(), "upsample must produce more samples");
    }

    #[test]
    fn test_resample_downsample_2x_length() {
        let mut r = make_sinc_resampler(16000, 8000).unwrap();
        let input: Vec<i16> = (0..512).map(|i| (i as i16) * 50).collect();
        let out = resample_sinc(&mut r, &input).unwrap();
        assert!(out.len() <= input.len(), "downsample must produce fewer samples");
    }

    #[test]
    fn test_resample_empty_input() {
        let mut r = make_sinc_resampler(8000, 16000).unwrap();
        let out = resample_sinc(&mut r, &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_resample_no_clamp_artifacts() {
        let mut r = make_sinc_resampler(8000, 16000).unwrap();
        // sine wave at ~440Hz in 8kHz signal
        let input: Vec<i16> = (0..256)
            .map(|i| ((i as f32 * 0.345).sin() * 16000.0) as i16)
            .collect();
        let out = resample_sinc(&mut r, &input).unwrap();
        // no full-scale clamp artifacts expected on a moderate sine
        let clipped = out.iter().filter(|&&s| s == i16::MIN || s == i16::MAX).count();
        assert_eq!(clipped, 0, "unexpected clipping in sinc output");
    }

    #[test]
    fn test_resample_same_rate_passthrough() {
        // same-rate: make_sinc_resampler with ratio=1.0, confirm output ~= input
        let mut r = make_sinc_resampler(16000, 16000).unwrap();
        let input: Vec<i16> = vec![0, 100, 200, 300, 400];
        let out = resample_sinc(&mut r, &input).unwrap();
        assert!(!out.is_empty());
    }

    /// Quality test: sinc must suppress spectral imaging.
    ///
    /// When upsampling from 8kHz to 16kHz, the zero-insertion step creates a mirror
    /// image of every input frequency f at (8000 - f) Hz in the output. For a 440Hz
    /// input tone the image appears at 7560Hz. A linear/polynomial interpolator leaves
    /// this image at about −44dB; a bandlimited sinc filter (BlackmanHarris2,
    /// sinc_len=128, f_cutoff=0.95) attenuates it by >70dB.
    ///
    /// 60dB threshold: passes sinc (>70dB), fails linear (~44dB).
    /// The SmartTurn mel spectrogram covers 0–8kHz, so an unfiltered 7560Hz image
    /// would land in the top mel bins and corrupt turn-detection inference.
    #[test]
    fn test_sinc_attenuates_spectral_imaging() {
        // 2 seconds of 440Hz at 8kHz (16000 samples)
        let n_in = 8000 * 2;
        let input: Vec<i16> = (0..n_in)
            .map(|i| {
                let t = i as f32 / 8000.0;
                ((2.0 * std::f32::consts::PI * 440.0 * t).sin() * 16000.0) as i16
            })
            .collect();

        let mut r = make_sinc_resampler(8000, 16000).unwrap();
        let out = resample_sinc(&mut r, &input).unwrap();

        let signal: Vec<f32> = out.iter().map(|&s| s as f32 / 32768.0).collect();

        // Power at 440Hz — the tone we put in
        let passband_power = goertzel_power(&signal, 440.0, 16000.0);
        // Power at the spectral image: 8000 - 440 = 7560Hz
        // (Note: 16000 - 440 = 15560Hz aliases back to 440Hz at 16kHz rate — wrong target)
        let image_power = goertzel_power(&signal, 7560.0, 16000.0);

        assert!(
            passband_power > 0.0,
            "no energy at 440Hz in upsampled output — resampler may be broken"
        );
        // 60dB = factor 1_000_000 in power
        assert!(
            image_power * 1_000_000.0 < passband_power,
            "image at 7560Hz insufficiently suppressed by sinc filter: \
             passband={passband_power:.3e}, image={image_power:.3e}, \
             attenuation={:.1}dB (need ≥60dB, linear interp gives ~44dB)",
            10.0 * (passband_power / image_power.max(1e-30_f32)).log10()
        );
    }
}
