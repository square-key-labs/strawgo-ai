mod features;
mod pool;
mod protocol;
mod resample;
mod server;
mod smart_turn;
mod vad;

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use tokio::net::UnixListener;
use tracing::{error, info};

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == flag {
            return iter.next().cloned();
        }
    }
    None
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .clamp(1, 4)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let vad_model = parse_arg(&args, "--vad-model")
        .ok_or_else(|| anyhow!("missing required argument: --vad-model"))?;
    let turn_model = parse_arg(&args, "--turn-model")
        .ok_or_else(|| anyhow!("missing required argument: --turn-model"))?;
    let socket_path = parse_arg(&args, "--socket")
        .ok_or_else(|| anyhow!("missing required argument: --socket"))?;
    let threads = match parse_arg(&args, "--threads") {
        Some(v) => v
            .parse::<usize>()
            .with_context(|| format!("invalid --threads value {v:?}"))?,
        None => default_threads(),
    };

    info!(vad_model = %vad_model, turn_model = %turn_model, threads, "onnx-worker starting");

    let vad_bytes: Arc<[u8]> = std::fs::read(&vad_model)
        .with_context(|| format!("read {vad_model}"))?
        .into();
    let turn_bytes: Arc<[u8]> = std::fs::read(&turn_model)
        .with_context(|| format!("read {turn_model}"))?
        .into();
    let pool = tokio::task::spawn_blocking(move || {
        pool::InferencePool::start(vad_bytes, turn_bytes, threads)
    })
    .await??;

    let _ = std::fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path)?;
    info!(socket = %socket_path, "onnx-worker ready");

    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let pool = Arc::clone(&pool);
                tokio::spawn(async move {
                    if let Err(e) = server::handle_connection(stream, pool).await {
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
