mod features;
mod protocol;
mod resample;
mod server;
mod smart_turn;
mod vad;

use std::thread::available_parallelism;

use anyhow::{anyhow, Result};
use ort::environment::GlobalThreadPoolOptions;
use tokio::net::UnixListener;
use tracing::{error, info, warn};

/// Parse a named argument from the command-line args list.
/// Looks for `--flag value` pairs; returns None if not found.
fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == flag {
            return iter.next().cloned();
        }
    }
    None
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialise tracing (respects RUST_LOG env var)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();

    let vad_model_fp32 = parse_arg(&args, "--vad-model")
        .ok_or_else(|| anyhow!("missing required argument: --vad-model"))?;

    // Optional int8-quantised VAD: pass `--vad-model-int8 <path>` (or set env
    // `ONNX_WORKER_VAD_INT8=<path>`). Source:
    // <https://huggingface.co/onnx-community/silero-vad/tree/main/onnx>
    // (`model_int8.onnx`, 639 KB vs the fp32 2.24 MB). The int8 build exposes
    // the same input/output names ({input,state,sr} → {output,stateN}) and
    // the same shapes, validated by `vad::build_shared_session`. If
    // validation fails we fall back to the fp32 model rather than refusing
    // to start, so a wrong-shape model file can never take the worker down.
    let vad_model_int8 = parse_arg(&args, "--vad-model-int8")
        .or_else(|| std::env::var("ONNX_WORKER_VAD_INT8").ok());

    let turn_model = parse_arg(&args, "--turn-model")
        .ok_or_else(|| anyhow!("missing required argument: --turn-model"))?;

    let socket_path = parse_arg(&args, "--socket")
        .ok_or_else(|| anyhow!("missing required argument: --socket"))?;

    info!(
        vad_model = %vad_model_fp32,
        vad_model_int8 = ?vad_model_int8,
        turn_model = %turn_model,
        "onnx-worker starting"
    );

    // Configure ORT to use a *single* global intra-op thread pool sized to the
    // physical core count. All sessions created after this `commit()` will
    // share the pool (ORT calls `DisablePerSessionThreads` when a session is
    // built under an env that has a global pool).
    //
    // Without this, each `Session::builder().commit_*` allocates its own pool;
    // 100 concurrent VAD streams × `nproc` pools = catastrophic context-switch
    // contention. See <https://onnxruntime.ai/docs/performance/tune-performance/threading.html>.
    //
    // `commit()` returns `false` if another caller already configured the env
    // (e.g. in tests or multi-call scenarios) — we log and continue, since the
    // first commit wins and our settings are the same anyway.
    let cpus = available_parallelism().map(|n| n.get()).unwrap_or(1);
    let pool_opts = GlobalThreadPoolOptions::default()
        .with_intra_threads(cpus)
        .map_err(|e| anyhow!("global threading: with_intra_threads: {e}"))?
        .with_inter_threads(1)
        .map_err(|e| anyhow!("global threading: with_inter_threads: {e}"))?;
    let env_committed = ort::init()
        .with_name("onnx-worker")
        .with_global_thread_pool(pool_opts)
        .commit();
    if env_committed {
        info!(intra_threads = cpus, "ORT global thread pool committed");
    } else {
        warn!("ORT environment was already committed elsewhere — global thread pool may not apply");
    }

    // Build the shared, process-wide Silero ORT session BEFORE we accept any
    // connections. Connections only hold per-stream LSTM state from here on.
    //
    // If an int8 model was requested, try to load it first; on failure (file
    // missing / wrong I/O surface / corrupt) fall back to the fp32 model and
    // log a warning — production correctness > startup speed.
    let shared_vad = if let Some(int8_path) = vad_model_int8.as_deref() {
        match vad::build_shared_session(int8_path) {
            Ok(s) => {
                info!(model_path = %int8_path, "shared silero int8 VAD session ready");
                s
            }
            Err(e) => {
                warn!(
                    model_path = %int8_path,
                    error = %e,
                    "int8 VAD model failed to load; falling back to fp32"
                );
                let s = vad::build_shared_session(&vad_model_fp32)?;
                info!(model_path = %vad_model_fp32, "shared silero fp32 VAD session ready (fallback)");
                s
            }
        }
    } else {
        let s = vad::build_shared_session(&vad_model_fp32)?;
        info!(model_path = %vad_model_fp32, "shared silero fp32 VAD session ready");
        s
    };

    // Remove stale socket file so bind() doesn't fail
    let _ = std::fs::remove_file(&socket_path);

    // Bind the Unix domain socket.
    // After bind() the socket file exists — Go supervisor polls os.Stat(sockPath).
    let listener = UnixListener::bind(&socket_path)?;

    info!(socket = %socket_path, "onnx-worker ready, socket: {}", socket_path);

    // Accept loop
    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let vad_session = shared_vad.clone();
                let turn_model_path = turn_model.clone();
                tokio::spawn(async move {
                    if let Err(e) =
                        server::handle_connection(stream, vad_session, &turn_model_path).await
                    {
                        error!(error = %e, "handle_connection returned error");
                    }
                });
            }
            Err(e) => {
                error!(error = %e, "accept() failed");
            }
        }
    }
}
