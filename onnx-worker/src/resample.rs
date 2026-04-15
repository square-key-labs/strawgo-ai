/// Simple linear-interpolation resampler, ported directly from Go's Resample().
///
/// This preserves exact numerical parity with the Go implementation.
/// For higher-quality production resampling, consider switching to rubato's
/// sinc interpolator once parity requirements are relaxed.
pub fn resample_linear(pcm: &[i16], from_rate: u32, to_rate: u32) -> Vec<i16> {
    if from_rate == to_rate {
        return pcm.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (pcm.len() as f64 / ratio) as usize;
    let mut output = vec![0i16; output_len];

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;

        output[i] = if src_idx + 1 < pcm.len() {
            let s1 = pcm[src_idx] as f64;
            let s2 = pcm[src_idx + 1] as f64;
            (s1 * (1.0 - frac) + s2 * frac) as i16
        } else if src_idx < pcm.len() {
            pcm[src_idx]
        } else {
            0
        };
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_no_op() {
        let pcm: Vec<i16> = (0..16).map(|i| i * 100).collect();
        let out = resample_linear(&pcm, 16000, 16000);
        assert_eq!(out, pcm);
    }

    #[test]
    fn test_resample_downsample_2x() {
        // Downsample by 2: every other sample via linear interpolation
        let pcm: Vec<i16> = vec![0, 100, 200, 300, 400, 500, 600, 700];
        let out = resample_linear(&pcm, 16000, 8000);
        // ratio = 2.0; output_len = 8/2 = 4
        // i=0: src_pos=0.0 → pcm[0]=0
        // i=1: src_pos=2.0 → pcm[2]=200
        // i=2: src_pos=4.0 → pcm[4]=400
        // i=3: src_pos=6.0 → pcm[6]=600
        assert_eq!(out, vec![0, 200, 400, 600]);
    }

    #[test]
    fn test_resample_upsample_2x() {
        // Upsample by 2: linear interpolation between samples
        let pcm: Vec<i16> = vec![0, 200, 400, 600];
        let out = resample_linear(&pcm, 8000, 16000);
        // ratio = 0.5; output_len = 4/0.5 = 8
        // i=0: src_pos=0.0  → pcm[0]=0
        // i=1: src_pos=0.5  → 0*0.5 + 200*0.5 = 100
        // i=2: src_pos=1.0  → pcm[1]=200
        // i=3: src_pos=1.5  → 200*0.5 + 400*0.5 = 300
        // i=4: src_pos=2.0  → pcm[2]=400
        // i=5: src_pos=2.5  → 400*0.5 + 600*0.5 = 500
        // i=6: src_pos=3.0  → pcm[3]=600  (src_idx+1=4 >= len=4, use pcm[3])
        // i=7: src_pos=3.5  → src_idx=3, src_idx+1=4 >= len → pcm[3]=600
        assert_eq!(out, vec![0, 100, 200, 300, 400, 500, 600, 600]);
    }
}
