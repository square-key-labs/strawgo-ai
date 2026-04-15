mod features;
mod protocol;
mod resample;
mod server;
mod smart_turn;
mod vad;

use anyhow::{anyhow, Result};
use tokio::net::UnixListener;
use tracing::{error, info};

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

    let vad_model = parse_arg(&args, "--vad-model")
        .ok_or_else(|| anyhow!("missing required argument: --vad-model"))?;

    let turn_model = parse_arg(&args, "--turn-model")
        .ok_or_else(|| anyhow!("missing required argument: --turn-model"))?;

    let socket_path = parse_arg(&args, "--socket")
        .ok_or_else(|| anyhow!("missing required argument: --socket"))?;

    info!(vad_model = %vad_model, turn_model = %turn_model, "onnx-worker starting");

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
                let vad_model_path = vad_model.clone();
                let turn_model_path = turn_model.clone();
                tokio::spawn(async move {
                    if let Err(e) =
                        server::handle_connection(stream, &vad_model_path, &turn_model_path).await
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
