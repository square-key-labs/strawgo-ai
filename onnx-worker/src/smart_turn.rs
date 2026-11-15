use anyhow::{anyhow, Result};
use ort::{inputs, session::builder::GraphOptimizationLevel, session::Session, value::TensorRef};
use rubato::Resampler;

const MAX_SAMPLES: usize = 8 * 16000;
const MEL_LEN: usize = 80 * 800;

pub struct SmartTurnModel {
    session: Session,
    feature_extractor: crate::features::WhisperFeatureExtractor,
    resampler: Option<(u32, rubato::Async<f32>)>,
    audio_f32: Vec<f32>,
    mel_padded: Vec<f32>,
}

impl SmartTurnModel {
    pub fn from_memory(model: &[u8]) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow!("{}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("{}", e))?
            .with_intra_threads(1)
            .map_err(|e| anyhow!("{}", e))?
            .commit_from_memory(model)
            .map_err(|e| anyhow!("{}", e))?;
        Ok(SmartTurnModel {
            session,
            feature_extractor: crate::features::WhisperFeatureExtractor::new(),
            resampler: None,
            audio_f32: Vec::with_capacity(MAX_SAMPLES),
            mel_padded: vec![0.0f32; MEL_LEN],
        })
    }

    pub fn run(
        &mut self,
        audio_pcm: &[i16],
        sample_rate: u32,
        _speech_start_ms: u32,
    ) -> Result<f32> {
        if sample_rate == 0 {
            return Err(anyhow!("smart_turn: sample_rate must not be 0"));
        }
        self.audio_f32.clear();
        if sample_rate != 16000 {
            let stale = !matches!(&self.resampler, Some((sr, _)) if *sr == sample_rate);
            if stale {
                self.resampler = Some((
                    sample_rate,
                    crate::resample::make_sinc_resampler(sample_rate, 16000)?,
                ));
            }
            let (_, rs) = self
                .resampler
                .as_mut()
                .expect("resampler initialised above");
            rs.reset();
            let pcm_16k = crate::resample::resample_sinc(rs, audio_pcm)?;
            self.audio_f32
                .extend(pcm_16k.iter().map(|&s| s as f32 / 32768.0));
        } else {
            self.audio_f32
                .extend(audio_pcm.iter().map(|&s| s as f32 / 32768.0));
        }

        let mel = self.feature_extractor.extract(&self.audio_f32, MAX_SAMPLES);
        let copy_len = mel.len().min(MEL_LEN);
        self.mel_padded[..copy_len].copy_from_slice(&mel[..copy_len]);
        self.mel_padded[copy_len..]
            .iter_mut()
            .for_each(|v| *v = 0.0);

        let tensor = TensorRef::from_array_view(([1usize, 80, 800], self.mel_padded.as_slice()))?;
        let outputs = self.session.run(inputs!["input_features" => tensor])?;
        let (_dims, data) = outputs["logits"].try_extract_tensor::<f32>()?;
        Ok(data[0])
    }
}
