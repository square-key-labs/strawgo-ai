use anyhow::Result;
use ort::{inputs, session::Session, session::builder::GraphOptimizationLevel, value::TensorRef};

pub struct SmartTurnSession {
    session: Session,
    feature_extractor: crate::features::WhisperFeatureExtractor,
    /// Sinc resampler — lazy-initialized on first non-16kHz audio.
    /// Constructed once per session to avoid rebuilding the sinc filter table.
    resampler: Option<rubato::Async<f32>>,
}

impl SmartTurnSession {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let feature_extractor = crate::features::WhisperFeatureExtractor::new();

        Ok(SmartTurnSession {
            session,
            feature_extractor,
            resampler: None,
        })
    }

    /// Returns turn completion probability (0.0 = incomplete, 1.0 = complete).
    /// audio_pcm: raw int16 LE samples, sample_rate: original sample rate,
    /// _speech_start_ms: available for future use (not used in v3.1 inference).
    pub fn run(
        &mut self,
        audio_pcm: &[i16],
        sample_rate: u32,
        _speech_start_ms: u32,
    ) -> Result<f32> {
        // 1. Validate and resample to 16kHz if needed
        if sample_rate == 0 {
            return Err(anyhow::anyhow!("smart_turn: sample_rate must not be 0"));
        }
        let pcm_16k = if sample_rate != 16000 {
            if self.resampler.is_none() {
                self.resampler = Some(crate::resample::make_sinc_resampler(sample_rate, 16000)?);
            }
            crate::resample::resample_sinc(self.resampler.as_mut().unwrap(), audio_pcm)?
        } else {
            audio_pcm.to_vec()
        };

        // 2. Convert to f32 normalized to [-1, 1]
        let audio_f32: Vec<f32> = pcm_16k.iter().map(|&s| s as f32 / 32768.0).collect();

        // 3. Extract mel spectrogram (8s window = 128000 samples)
        const MAX_SAMPLES: usize = 8 * 16000; // 128000
        let mel = self.feature_extractor.extract(&audio_f32, MAX_SAMPLES);

        // 4. Zero-pad mel to exactly 80*800 = 64000 floats
        let mut mel_padded = vec![0.0f32; 80 * 800];
        let copy_len = mel.len().min(64000);
        mel_padded[..copy_len].copy_from_slice(&mel[..copy_len]);

        // 5. Create input tensor with shape [1, 80, 800]
        let tensor =
            TensorRef::from_array_view(([1usize, 80, 800], mel_padded.as_slice()))?;

        // 6. Run inference
        let outputs = self.session.run(inputs!["input_features" => tensor])?;
        let (_dims, data) = outputs["logits"].try_extract_tensor::<f32>()?;

        Ok(data[0])
    }
}
