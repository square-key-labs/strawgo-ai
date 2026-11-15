use anyhow::{anyhow, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

const MAX_PAYLOAD: u32 = 16 * 1024 * 1024;

pub async fn read_frame(stream: &mut UnixStream, payload: &mut Vec<u8>) -> Result<u8> {
    let msg_type = stream.read_u8().await?;
    let payload_len = stream.read_u32_le().await?;
    if payload_len > MAX_PAYLOAD {
        return Err(anyhow!(
            "payload_len {} exceeds maximum allowed size",
            payload_len
        ));
    }
    payload.clear();
    payload.resize(payload_len as usize, 0);
    if payload_len > 0 {
        stream.read_exact(payload).await?;
    }
    Ok(msg_type)
}

pub async fn write_response(stream: &mut UnixStream, value: f32) -> Result<()> {
    stream.write_all(&value.to_le_bytes()).await?;
    Ok(())
}
