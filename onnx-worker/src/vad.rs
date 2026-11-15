use anyhow::{anyhow, Result};
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

pub struct VadState {
    hidden_state: Vec<f32>,
    context: Vec<f32>,
    last_sr: u32,
}

impl Default for VadState {
    fn default() -> Self {
        Self {
            hidden_state: vec![0.0f32; 256],
            context: Vec::new(),
            last_sr: 0,
        }
    }
}

pub struct SileroModel {
    session: Session,
    input: Vec<f32>,
}

impl SileroModel {
    pub fn from_memory(model: &[u8]) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow!("{}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("{}", e))?
            .with_intra_threads(1)
            .map_err(|e| anyhow!("{}", e))?
            .commit_from_memory(model)
            .map_err(|e| anyhow!("{}", e))?;
        Ok(Self {
            session,
            input: Vec::with_capacity(64 + 512),
        })
    }

    pub fn run(
        &mut self,
        state: &mut VadState,
        audio_pcm: &[i16],
        sample_rate: u32,
    ) -> Result<f32> {
        let (ctx_size, expected_samples) = match sample_rate {
            16000 => (64usize, 512usize),
            8000 => (32usize, 256usize),
            other => {
                return Err(anyhow!(
                    "unsupported sample_rate {}: must be 8000 or 16000",
                    other
                ))
            }
        };
        if audio_pcm.len() != expected_samples {
            return Err(anyhow!(
                "num_samples {} does not match expected {} for {}Hz",
                audio_pcm.len(),
                expected_samples,
                sample_rate
            ));
        }

        if state.last_sr != 0 && state.last_sr != sample_rate {
            state.hidden_state.iter_mut().for_each(|v| *v = 0.0);
            state.context.clear();
        }
        if state.context.len() != ctx_size {
            state.context.clear();
            state.context.resize(ctx_size, 0.0);
        }

        self.input.clear();
        self.input.extend_from_slice(&state.context);
        self.input
            .extend(audio_pcm.iter().map(|&s| s as f32 / 32768.0));
        let input_len = self.input.len();

        let input_t = TensorRef::from_array_view(([1usize, input_len], self.input.as_slice()))?;
        let state_t =
            TensorRef::from_array_view(([2usize, 1, 128], state.hidden_state.as_slice()))?;
        let sr_data = [sample_rate as i64];
        let sr_t = TensorRef::from_array_view(([1usize], &sr_data[..]))?;

        let outputs = self
            .session
            .run(inputs!["input" => input_t, "state" => state_t, "sr" => sr_t])?;

        let (_dims, conf_data) = outputs["output"].try_extract_tensor::<f32>()?;
        let confidence = conf_data[0];
        let (_sdims, new_state) = outputs["stateN"].try_extract_tensor::<f32>()?;
        state.hidden_state.clear();
        state.hidden_state.extend_from_slice(new_state);

        state.context.clear();
        state
            .context
            .extend_from_slice(&self.input[input_len - ctx_size..]);
        state.last_sr = sample_rate;

        Ok(confidence)
    }
}
