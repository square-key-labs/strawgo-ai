use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

/// Shared, process-wide Silero ORT session.
///
/// Built once at process start and shared across every connection (`Arc<Mutex<...>>`).
/// The Silero maintainer has explicitly confirmed the model can be shared across
/// independent streams as long as each stream keeps its own LSTM state — see
/// <https://github.com/snakers4/silero-vad/discussions/744> and
/// <https://github.com/pipecat-ai/pipecat/issues/2050>.
///
/// Why a mutex even though `Session: Send + Sync`?  The `pykeio/ort` 2.0.0-rc.12
/// public `Session::run` API takes `&mut self`, although the underlying
/// `run_inner(&self, ...)` and the C++ `OrtSession::Run` are thread-safe. We
/// therefore serialise access at the Rust level. The hold time is the kernel of
/// inference (~1 ms with int8, ~2 ms with fp32 on a 4-core VM), so for the
/// concurrency levels this worker targets (≤ 200 streams × 32 ms cadence) the
/// mutex is not the bottleneck — but it cuts RSS by ~50× because ORT's memory
/// arena, weight tensors, and graph optimiser state are loaded exactly once.
pub type SharedSileroSession = Arc<Mutex<Session>>;

/// Build the shared Silero session. Called once from `main`.
pub fn build_shared_session(model_path: &str) -> Result<SharedSileroSession> {
    let session = Session::builder()
        .map_err(|e| anyhow!("{}", e))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{}", e))?
        // Each session uses 1 intra-op thread. When a global thread pool is
        // configured on the env, this is overridden by `DisablePerSessionThreads`
        // (see `ort::session::builder::impl_commit::pre_commit`); we keep it
        // here so the worker still behaves sensibly if the env was not set up.
        .with_intra_threads(1)
        .map_err(|e| anyhow!("{}", e))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow!("{}", e))?;

    // Validate the model has the I/O surface this worker speaks. We support
    // both the standard `silero_vad.onnx` and the int8 `model_int8.onnx` from
    // <https://huggingface.co/onnx-community/silero-vad>; both expose
    // inputs {input, state, sr} and outputs {output, stateN}.
    validate_silero_io(&session)?;

    Ok(Arc::new(Mutex::new(session)))
}

fn validate_silero_io(session: &Session) -> Result<()> {
    let want_inputs = ["input", "state", "sr"];
    let want_outputs = ["output", "stateN"];

    let got_inputs: Vec<&str> = session.inputs().iter().map(|o| o.name()).collect();
    let got_outputs: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();

    for n in want_inputs {
        if !got_inputs.iter().any(|g| *g == n) {
            return Err(anyhow!(
                "silero model missing required input '{}' (got inputs: {:?})",
                n,
                got_inputs
            ));
        }
    }
    for n in want_outputs {
        if !got_outputs.iter().any(|g| *g == n) {
            return Err(anyhow!(
                "silero model missing required output '{}' (got outputs: {:?})",
                n,
                got_outputs
            ));
        }
    }
    Ok(())
}

/// Per-connection Silero VAD state.
///
/// Holds only the LSTM hidden state (`[2, 1, 128]` f32 = 256 values) and a
/// 32- or 64-sample context buffer. The model itself is shared across
/// connections via `SharedSileroSession`.
pub struct SileroSession {
    session: SharedSileroSession,
    /// LSTM hidden state tensor data: shape [2, 1, 128] = 256 f32 values.
    hidden_state: Vec<f32>,
    /// Last `ctx_size` samples fed as context; empty until first call.
    context: Vec<f32>,
    /// Sample rate from last call; 0 means not yet set.
    last_sr: u32,
}

impl SileroSession {
    /// Create a new per-connection state attached to the shared session.
    pub fn new(session: SharedSileroSession) -> Self {
        Self {
            session,
            hidden_state: vec![0.0f32; 256], // [2, 1, 128]
            context: vec![],                 // populated on first call
            last_sr: 0,
        }
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

        // --- Run inference under the shared-session mutex ---
        let confidence;
        {
            let mut guard = self
                .session
                .lock()
                .map_err(|e| anyhow!("shared silero session mutex poisoned: {e}"))?;

            let outputs = guard.run(
                inputs!["input" => input_t, "state" => state_t, "sr" => sr_t],
            )?;

            // --- Extract confidence ---
            let (_dims, conf_data) = outputs["output"].try_extract_tensor::<f32>()?;
            confidence = conf_data[0];

            // --- Extract updated hidden state ---
            let (_sdims, new_state) = outputs["stateN"].try_extract_tensor::<f32>()?;
            self.hidden_state.clear();
            self.hidden_state.extend_from_slice(new_state);
        }

        // --- Update context: last ctx_size samples of input_data ---
        let context_start = input_data.len() - ctx_size;
        self.context.clear();
        self.context
            .extend_from_slice(&input_data[context_start..]);

        self.last_sr = sample_rate;

        Ok(confidence)
    }
}
