/// Benchmarks for the audio pipeline stages that run on every turn.
///
/// Run with:
///   cargo bench                          # all benchmarks
///   cargo bench -- resample             # just resample group
///   cargo bench -- mel                  # just mel group
///   cargo bench --bench audio -- --save-baseline main   # save baseline
///
/// Key question: is the resampler or mel extractor the bottleneck vs ONNX?
/// Expected answer: ONNX >> mel > resample (resample should be <1ms for any input).
///
/// Silence-skip optimization effect on mel/extract:
///   0.5s input (8000/128000 samples): ~16× speedup (748/800 frames skipped)
///   2.0s input (32000/128000 samples): ~4× speedup (598/800 frames skipped)
///   8.0s input (128000/128000 samples): no change (first_real_frame=0)
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use onnx_worker::{
    features::WhisperFeatureExtractor,
    resample::{make_sinc_resampler, resample_sinc},
};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Shared test signals
// ---------------------------------------------------------------------------

/// 440Hz sine at 8kHz for the given duration (seconds).
fn sine_8k(secs: f32) -> Vec<i16> {
    let n = (8000.0 * secs) as usize;
    (0..n)
        .map(|i| ((2.0 * PI * 440.0 * i as f32 / 8000.0).sin() * 16000.0) as i16)
        .collect()
}

/// 440Hz sine at 16kHz for the given duration (seconds), already in f32.
fn sine_16k_f32(secs: f32) -> Vec<f32> {
    let n = (16000.0 * secs) as usize;
    (0..n)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark: sinc resampler construction
// ---------------------------------------------------------------------------
//
// This measures how long make_sinc_resampler() takes — i.e. the cost that is
// amortised by constructing once per SmartTurnSession rather than per turn.
// If this is hundreds of ms, the lazy-init strategy is load-bearing.

fn bench_resampler_construction(c: &mut Criterion) {
    c.bench_function("resample/construction_8k→16k", |b| {
        b.iter(|| make_sinc_resampler(black_box(8000), black_box(16000)).unwrap())
    });
}

// ---------------------------------------------------------------------------
// Benchmark: resample_sinc at realistic turn durations
// ---------------------------------------------------------------------------
//
// SmartTurn accumulates audio from speech_start to turn_end, typically 0.5–8s.
// Measures how long one call to resample_sinc takes at each duration.
// Uses iter_batched so setup (resampler construction) is excluded from timing.

fn bench_resample(c: &mut Criterion) {
    let mut group = c.benchmark_group("resample/8k→16k");

    for secs in [0.5f32, 2.0, 8.0] {
        let input = sine_8k(secs);
        group.bench_with_input(
            BenchmarkId::new("sinc_len=128", format!("{secs}s")),
            &input,
            |b, input| {
                b.iter_batched(
                    || make_sinc_resampler(8000, 16000).unwrap(),
                    |mut r| resample_sinc(&mut r, black_box(input)).unwrap(),
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: mel spectrogram extraction
// ---------------------------------------------------------------------------
//
// WhisperFeatureExtractor::extract() is called on every SmartTurn inference.
// Input is always padded to 8s (128000 samples), but audio length varies.
// With silence-skip optimization, shorter inputs are proportionally faster:
//   0.5s → ~16× speedup (skips 748/800 STFT frames)
//   2.0s → ~4× speedup (skips 598/800 STFT frames)
//   8.0s → no change (all frames computed)

fn bench_mel_extract(c: &mut Criterion) {
    let mut group = c.benchmark_group("mel/extract");

    for secs in [0.5f32, 2.0, 8.0] {
        let audio = sine_16k_f32(secs);
        const MAX_SAMPLES: usize = 8 * 16000; // 128000

        // One extractor per benchmark parameter, reused across iterations.
        let mut extractor = WhisperFeatureExtractor::new();

        group.bench_with_input(
            BenchmarkId::new("whisper_80mel", format!("{secs}s_input")),
            &audio,
            |b, audio| {
                b.iter(|| extractor.extract(black_box(audio), black_box(MAX_SAMPLES)))
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_resampler_construction,
    bench_resample,
    bench_mel_extract,
);
criterion_main!(benches);
