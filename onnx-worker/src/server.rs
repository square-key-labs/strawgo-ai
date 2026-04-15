use std::io::ErrorKind;

use anyhow::{anyhow, Result};
use tokio::net::UnixStream;
use tracing::{debug, error, info};

use crate::protocol::{read_frame, write_response};
use crate::smart_turn::SmartTurnSession;
use crate::vad::SileroSession;

const MSG_VAD: u8 = 0x01;
const MSG_SMART_TURN: u8 = 0x02;

/// Parse the VAD payload.
///
/// Wire layout:
///   [u32 LE sample_rate][i16 LE samples...]
///
/// Returns (sample_rate, Vec<i16>).
fn parse_vad_payload(payload: &[u8]) -> Result<(u32, Vec<i16>)> {
    if payload.len() < 4 {
        return Err(anyhow!(
            "VAD payload too short: {} bytes (need at least 4 for sample_rate)",
            payload.len()
        ));
    }

    let sample_rate = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);

    let audio_bytes = &payload[4..];
    if audio_bytes.len() % 2 != 0 {
        return Err(anyhow!(
            "VAD audio bytes have odd length {}: must be even (i16 samples)",
            audio_bytes.len()
        ));
    }

    let samples: Vec<i16> = audio_bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();

    Ok((sample_rate, samples))
}

/// Parse the SmartTurn payload.
///
/// Wire layout:
///   [u32 LE sample_rate][u32 LE speech_start_ms][i16 LE samples...]
///
/// Returns (sample_rate, speech_start_ms, Vec<i16>).
fn parse_smart_turn_payload(payload: &[u8]) -> Result<(u32, u32, Vec<i16>)> {
    if payload.len() < 8 {
        return Err(anyhow!(
            "SmartTurn payload too short: {} bytes (need at least 8 for sample_rate + speech_start_ms)",
            payload.len()
        ));
    }

    let sample_rate = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let speech_start_ms = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);

    let audio_bytes = &payload[8..];
    if audio_bytes.len() % 2 != 0 {
        return Err(anyhow!(
            "SmartTurn audio bytes have odd length {}: must be even (i16 samples)",
            audio_bytes.len()
        ));
    }

    let samples: Vec<i16> = audio_bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();

    Ok((sample_rate, speech_start_ms, samples))
}

/// Handle a single client connection.
///
/// Each connection owns its own `SileroSession` so hidden state accumulates
/// across VAD calls for the lifetime of the connection. A `SmartTurnSession`
/// is also created per connection for turn-completion inference.
pub async fn handle_connection(
    mut stream: UnixStream,
    vad_model_path: &str,
    turn_model_path: &str,
) -> Result<()> {
    info!("client connected");

    let mut vad_session = SileroSession::new(vad_model_path)?;
    let mut smart_turn_session = SmartTurnSession::new(turn_model_path)?;

    loop {
        match read_frame(&mut stream).await {
            Ok((msg_type, payload)) => {
                let label = match msg_type {
                    MSG_VAD => "VAD",
                    MSG_SMART_TURN => "SmartTurn",
                    other => {
                        error!(msg_type = other, "unknown msg_type, closing connection");
                        break;
                    }
                };

                debug!(
                    msg_type = label,
                    payload_len = payload.len(),
                    "frame received"
                );

                match msg_type {
                    MSG_VAD => {
                        let confidence = match parse_vad_payload(&payload) {
                            Ok((sr, samples)) => match vad_session.run(&samples, sr) {
                                Ok(c) => c,
                                Err(e) => {
                                    error!(error = %e, "vad_session.run failed");
                                    break;
                                }
                            },
                            Err(e) => {
                                error!(error = %e, "parse_vad_payload failed");
                                break;
                            }
                        };

                        if let Err(e) = write_response(&mut stream, confidence).await {
                            error!(error = %e, "write_response failed");
                            break;
                        }
                    }
                    MSG_SMART_TURN => {
                        let probability = match parse_smart_turn_payload(&payload) {
                            Ok((sr, speech_start_ms, samples)) => {
                                match smart_turn_session.run(&samples, sr, speech_start_ms) {
                                    Ok(p) => p,
                                    Err(e) => {
                                        error!(error = %e, "smart_turn_session.run failed");
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                error!(error = %e, "parse_smart_turn_payload failed");
                                break;
                            }
                        };

                        if let Err(e) = write_response(&mut stream, probability).await {
                            error!(error = %e, "write_response failed");
                            break;
                        }
                    }
                    _ => unreachable!(),
                }
            }
            Err(e) => {
                // Unwrap underlying IO error to detect clean EOF
                if let Some(io_err) = e.downcast_ref::<std::io::Error>() {
                    if io_err.kind() == ErrorKind::UnexpectedEof {
                        info!("client disconnected (EOF)");
                        return Ok(());
                    }
                }
                error!(error = %e, "read_frame failed");
                break;
            }
        }
    }

    Ok(())
}
