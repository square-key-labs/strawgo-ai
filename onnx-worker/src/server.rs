use std::io::ErrorKind;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use tokio::net::UnixStream;
use tracing::{debug, error, info};

use crate::pool::{InferencePool, TurnJob, VadJob};
use crate::protocol::{read_frame, write_response};
use crate::vad::VadState;

const MSG_VAD: u8 = 0x01;
const MSG_SMART_TURN: u8 = 0x02;

fn read_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn decode_samples(audio: &[u8], out: &mut Vec<i16>) -> Result<()> {
    if !audio.len().is_multiple_of(2) {
        return Err(anyhow!(
            "audio bytes have odd length {}: must be even (i16 samples)",
            audio.len()
        ));
    }
    out.clear();
    out.extend(
        audio
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes(*b)),
    );
    Ok(())
}

fn parse_vad_payload(payload: &[u8], samples: &mut Vec<i16>) -> Result<u32> {
    if payload.len() < 4 {
        return Err(anyhow!(
            "VAD payload too short: {} bytes (need at least 4 for sample_rate)",
            payload.len()
        ));
    }
    decode_samples(&payload[4..], samples)?;
    Ok(read_u32(payload, 0))
}

fn parse_smart_turn_payload(payload: &[u8], samples: &mut Vec<i16>) -> Result<(u32, u32)> {
    if payload.len() < 8 {
        return Err(anyhow!(
            "SmartTurn payload too short: {} bytes (need at least 8)",
            payload.len()
        ));
    }
    decode_samples(&payload[8..], samples)?;
    Ok((read_u32(payload, 0), read_u32(payload, 4)))
}

pub async fn handle_connection(mut stream: UnixStream, pool: Arc<InferencePool>) -> Result<()> {
    info!("client connected");

    let mut vad_state = Some(VadState::default());
    let mut payload: Vec<u8> = Vec::new();
    let mut samples: Vec<i16> = Vec::new();

    loop {
        let msg_type = match read_frame(&mut stream, &mut payload).await {
            Ok(t) => t,
            Err(e) => {
                if let Some(io_err) = e.downcast_ref::<std::io::Error>() {
                    if io_err.kind() == ErrorKind::UnexpectedEof {
                        info!("client disconnected (EOF)");
                        return Ok(());
                    }
                }
                error!(error = %e, "read_frame failed");
                return Ok(());
            }
        };
        debug!(msg_type, payload_len = payload.len(), "frame received");

        let value = match msg_type {
            MSG_VAD => {
                let sample_rate = match parse_vad_payload(&payload, &mut samples) {
                    Ok(sr) => sr,
                    Err(e) => {
                        error!(error = %e, "parse_vad_payload failed");
                        return Ok(());
                    }
                };
                let state = vad_state.take().unwrap_or_default();
                let job = VadJob {
                    samples: std::mem::take(&mut samples),
                    sample_rate,
                    state,
                };
                let outcome = pool.vad(job).await?;
                samples = outcome.samples;
                vad_state = Some(outcome.state);
                match outcome.result {
                    Ok(c) => c,
                    Err(e) => {
                        error!(error = %e, "vad inference failed");
                        return Ok(());
                    }
                }
            }
            MSG_SMART_TURN => {
                let (sample_rate, speech_start_ms) =
                    match parse_smart_turn_payload(&payload, &mut samples) {
                        Ok(v) => v,
                        Err(e) => {
                            error!(error = %e, "parse_smart_turn_payload failed");
                            return Ok(());
                        }
                    };
                let job = TurnJob {
                    samples: std::mem::take(&mut samples),
                    sample_rate,
                    speech_start_ms,
                };
                let outcome = pool.turn(job).await?;
                samples = outcome.samples;
                match outcome.result {
                    Ok(p) => p,
                    Err(e) => {
                        error!(error = %e, "smart turn inference failed");
                        return Ok(());
                    }
                }
            }
            other => {
                error!(msg_type = other, "unknown msg_type, closing connection");
                return Ok(());
            }
        };

        if let Err(e) = write_response(&mut stream, value).await {
            error!(error = %e, "write_response failed");
            return Ok(());
        }
    }
}
