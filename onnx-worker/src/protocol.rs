use anyhow::{anyhow, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

/// Read one frame from the stream.
/// Frame format: [u8 msg_type][u32 payload_len LE][payload bytes]
/// Returns (msg_type, payload) or an error.
/// An UnexpectedEof error from the stream indicates clean client disconnect.
pub async fn read_frame(stream: &mut UnixStream) -> Result<(u8, Vec<u8>)> {
    // Read message type (1 byte)
    let msg_type = stream.read_u8().await?;

    // Read payload length (4 bytes, little-endian)
    let payload_len = stream.read_u32_le().await?;

    // Sanity cap: 16 MiB
    if payload_len > 16 * 1024 * 1024 {
        return Err(anyhow!(
            "payload_len {} exceeds maximum allowed size",
            payload_len
        ));
    }

    // Read payload
    let mut payload = vec![0u8; payload_len as usize];
    if payload_len > 0 {
        stream.read_exact(&mut payload).await?;
    }

    Ok((msg_type, payload))
}

/// Write a 4-byte little-endian f32 response.
pub async fn write_response(stream: &mut UnixStream, value: f32) -> Result<()> {
    let bytes = value.to_le_bytes();
    stream.write_all(&bytes).await?;
    Ok(())
}
