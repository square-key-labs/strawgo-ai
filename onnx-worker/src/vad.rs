use anyhow::{anyhow, Result};
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

/// Per-connection Silero VAD session.
/// Hidden state accumulates across calls for the lifetime of the connection.
pub struct SileroSession {
    session: Session,
    /// Hidden state tensor data: shape [2, 1, 128] = 256 f32 values.
    hidden_state: Vec<f32>,
    /// Last ctx_size samples fed as context; empty until first call.
    context: Vec<f32>,
    /// Sample rate from last call; 0 means not yet set.
    last_sr: u32,
}

impl SileroSession {
    /// Create a new session loading the model from `model_path`.
    /// Hidden state is zeroed; context is empty (filled on first call).
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow!("{}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("{}", e))?
            .with_intra_threads(1)
            .map_err(|e| anyhow!("{}", e))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("{}", e))?;

        Ok(Self {
            session,
            hidden_state: vec![0.0f32; 256], // [2, 1, 128]
            context: vec![],                 // populated on first call
            last_sr: 0,
        })
    }

    /// Run VAD inference on `audio_pcm` (raw i16 samples, not bytes).
    ///
    /// Returns the speech-probability confidence in [0.0, 1.0].
    pub fn run(&mut self, audio_pcm: &[i16], sample_rate: u32) -> Result<f32> {
        // --- Validate sample rate ---
        if sample_rate != 8000 && sample_rate != 16000 {
            return Err(anyhow!(
                "unsupported sample_rate {}: must be 8000 or 16000",
                sample_rate
            ));
        }

        let (ctx_size, expected_samples) = if sample_rate == 16000 {
            (64usize, 512usize)
        } else {
            (32usize, 256usize)
        };

        // --- Validate num_samples ---
        if audio_pcm.len() != expected_samples {
            return Err(anyhow!(
                "num_samples {} does not match expected {} for {}Hz",
                audio_pcm.len(),
                expected_samples,
                sample_rate
            ));
        }

        // --- Reset state if sample rate changed ---
        if self.last_sr != 0 && self.last_sr != sample_rate {
            self.hidden_state = vec![0.0f32; 256];
            self.context.clear();
        }

        // --- Initialise context on first call ---
        if self.context.is_empty() {
            self.context = vec![0.0f32; ctx_size];
        }

        // --- Normalize i16 PCM → f32 ---
        let normalized: Vec<f32> = audio_pcm.iter().map(|&s| s as f32 / 32768.0).collect();

        // --- Build input tensor data: [context..., audio...] ---
        let input_len = ctx_size + expected_samples;
        let mut input_data: Vec<f32> = Vec::with_capacity(input_len);
        input_data.extend_from_slice(&self.context);
        input_data.extend_from_slice(&normalized);

        // --- Create TensorRefs ---
        let input_shape = [1usize, input_len];
        let input_t = TensorRef::from_array_view((input_shape, input_data.as_slice()))?;

        let state_shape = [2usize, 1, 128];
        let state_t =
            TensorRef::from_array_view((state_shape, self.hidden_state.as_slice()))?;

        let sr_data: Vec<i64> = vec![sample_rate as i64];
        let sr_shape = [1usize];
        let sr_t = TensorRef::from_array_view((sr_shape, sr_data.as_slice()))?;

        // --- Run inference ---
        let outputs =
            self.session
                .run(inputs!["input" => input_t, "state" => state_t, "sr" => sr_t])?;

        // --- Extract confidence ---
        let (_dims, conf_data) = outputs["output"].try_extract_tensor::<f32>()?;
        let confidence = conf_data[0];

        // --- Extract updated hidden state ---
        let (_sdims, new_state) = outputs["stateN"].try_extract_tensor::<f32>()?;
        self.hidden_state.clear();
        self.hidden_state.extend_from_slice(&new_state);

        // --- Update context: last ctx_size samples of input_data ---
        let context_start = input_data.len() - ctx_size;
        self.context.clear();
        self.context
            .extend_from_slice(&input_data[context_start..]);

        self.last_sr = sample_rate;

        Ok(confidence)
    }
}
