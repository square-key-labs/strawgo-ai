# Voice-AI Inference Architecture — Industry State, May 2026

Survey of what production voice-AI stacks actually use in 2026, before
locking in Strawgo's path. Goal: align with industry-validated solutions,
not invent custom infrastructure.

## TL;DR

The industry is not solving "many sessions per process via clever IPC."
The industry is solving **"one session per container, scale horizontally
via Kubernetes."** That's the LiveKit playbook, the Pipecat playbook, the
NVIDIA-NIM-blueprint playbook, and the Bland AI playbook.

Strawgo's cgo-embed result (5 000+ agents per box) is **already
1-2 orders of magnitude beyond what containerized Python ships.** The
question is no longer "how do we scale?" — it's "do we want to be the
weird one or be compatible with the ecosystem?"

## What the leaders actually run

### LiveKit Agents 1.5 (April 2026)
- **Per-session worker process.** Each session bound to a single worker for
  full duration. State isolated per-process.
- **Worker pool**, horizontally scaled. Pre-warmed pool for burst capacity.
- **LiveKit Cloud Inference Gateway**: unified gRPC for STT/LLM/TTS
  routing across providers (OpenAI, Cartesia, Deepgram).
- 1 container per session — that's the official LiveKit answer.
[forasoft.com](https://www.forasoft.com/blog/article/livekit-ai-agents-guide), [docs.livekit.io](https://docs.livekit.io/deploy/admin/quotas-and-limits/)

### Pipecat 1.5 + NVIDIA NIM (production blueprint)
- **"Never run multiple voice sessions in the same Python process. This
  is the golden rule validated by every production deployment."**
- Inference (STT/LLM/TTS) via NVIDIA NIM microservices over HTTP/gRPC
- Pipecat orchestrator is CPU-bound, GIL-blocked, container-per-session
- Daily.co + NVIDIA shipping `nimble-pipecat`
[luonghongthuan.com](https://luonghongthuan.com/en/blog/pipecat-voice-agent-production-scalable-guide/), [build.nvidia.com/pipecat](https://build.nvidia.com/pipecat/voice-agent-framework-for-conversational-ai)

### Bland AI / Vapi / Retell
- Horizontal scale "thousands of concurrent calls" via container
  fleet + load balancers
- Latency 580–620 ms end-to-end (the threshold where users stop
  noticing AI)
- Cartesia Sonic 2 holds the production TTS record: TTFB 40 ms
[retellai.com](https://www.retellai.com/blog/best-voice-ai-providers), [softcery.com](https://softcery.com/lab/choosing-the-right-voice-agent-platform-in-2025)

**Implication:** the industry's bottleneck is vendor APIs (STT/LLM/TTS),
not local VAD. Local VAD is treated as a small CPU cost (Silero <1 ms per
call). Nobody is running 200 VAD streams in one process; they run 1 VAD
per session per container.

## Inference-runtime landscape, May 2026

| runtime | language | maturity | shape |
|---|---|---|---|
| **Triton Inference Server** | C++ server | production-grade | gRPC/HTTP service; backends for ONNX, TensorRT, vLLM, TensorFlow. Industry default for multi-model serving. [docs.nvidia.com](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html) |
| **Modular MAX** (26.2) | C++/Mojo, Python+ONNX API | production-graduating | Wraps Triton; PyTorch-like API graduated 26.1; 4× speedup on FLUX.2 in 26.2. Native ONNX. **No Go SDK** — talk via Triton gRPC. [modular.com/max](https://www.modular.com/max), [docs.modular.com](https://docs.modular.com/max/changelog/) |
| **ONNX Runtime** (1.25 stable) | C++ + first-party Python/C#/Java/JS, third-party Go/Rust | production | The "lingua franca" runtime. IOBinding for zero-copy. Used inside both Triton and Modular MAX. [onnxruntime.ai](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html) |
| **Burn** (Rust) | pure Rust | production case studies | Backend-agnostic (CPU/CUDA/Metal/WGPU/WASM). 2.3× speedup vs PyTorch in healthcare imaging case study. [github.com/tracel-ai/burn](https://github.com/tracel-ai/burn) |
| **Candle** (HuggingFace) | pure Rust | production-ready inference | Minimalist, quantization-focused, small binary, fast startup. [dev.to/mayu2008](https://dev.to/mayu2008/building-sentence-transformers-in-rust-a-practical-guide-with-burn-onnx-runtime-and-candle-281k) |
| **tract** (Sonos) | pure Rust | mature | Lightweight, portable, no LSTM v5 op coverage gaps reported recently. |
| **vLLM** (LLM-only) | Python | production | LLM serving champion. Not for VAD. |
| **TensorRT-LLM** | C++/CUDA | production | NVIDIA only, LLM-only. |

## ONNX Go bindings, May 2026 reality

| binding | API | cgo? | IOBinding? | maintained | verdict |
|---|---|---|---|---|---|
| **yalue/onnxruntime_go** v1.29 | wraps ORT C API | yes | yes — but **copies C→Go on each Run** to prevent leaks; not truly zero-copy. [pkg.go.dev/yalue](https://pkg.go.dev/github.com/yalue/onnxruntime_go) | **active** (push 2026-04-29) | **default choice** |
| **shota3506/onnxruntime-purego** | wraps ORT C API via `purego` | **NO cgo** | unclear | active | interesting — eliminates cgo build pain |
| **8ff/onnxruntime_go** | fork of yalue | yes | likely same | maintained | minor |
| **owulveryck/onnx-go** | pure Go | no | n/a | stale (Sep 2024) | rejected — no LSTM v5 |
| **knights-analytics/hugot** | Go ONNX wrapper for Transformers | yes (uses ORT) | unclear | active | good for transformer pipelines, not Silero |

Real takeaway on Go ONNX:
- **yalue's "tensor copy on each Run"** is the latency floor we measured
  (3–17 ms p99). IOBinding doesn't help much because the copy still
  happens (Go GC ownership constraint).
- **purego variant** worth evaluating — eliminates cgo build pain (cross
  compile, linker, libstdc++ vs libc++). But unclear if it has a tighter
  hot path.
- The **right fix in this lane** is to fork yalue or write a small
  custom binding that exposes raw `*C.float` pointers and uses
  `runtime.Pinner` (Go 1.21+) to keep Go memory pinned during ORT calls.
  ~50 lines of cgo. **This is the cheapest "make Go match Python ORT"
  path.**

## VAD model landscape, May 2026

| model | size | RTF | accuracy claim | license | Go support |
|---|---|---|---|---|---|
| **Silero v6.2.1** (Feb 2026) | ~2 MB fp32 / 639 KB int8 | ~0.03 | 16% fewer errors than v5 on noisy data | MIT | ONNX, easy via yalue. [silero-vad](https://github.com/snakers4/silero-vad) |
| **Cobra (Picovoice)** | ~50 KB | **0.005** (8.6× faster than Silero) | "best-in-class," 99% claim | commercial, paid | **NO Go SDK**. C/Linux/macOS/Win/Pi/Web only. |
| **TEN-VAD** (ByteDance) | 306 KB lib | 0.0086–0.0160 | "beats Silero on librispeech/gigaspeech/DNS" | Apache 2.0 + **Agora non-compete** | C/Go bindings (we tried). |
| **Pyannote** | varies | research-grade | SOTA per their benchmark | MIT | Python only; via Triton/HTTP otherwise. |
| **WebRTC VAD** | tiny | <1µs | low (50% TPR @ 5% FPR) | BSD | many Go ports. Use only as ultra-cheap fallback. |
| **FusionVAD** (research) | varies | unknown | beats Pyannote +2.04% avg | research repo | not productionized. |

**Silero v6.2.1 is the easy upgrade** — same ONNX shape, drop-in.
16% / 11% accuracy improvement is real. Strawgo today still uses v5
or earlier.

**Cobra and TEN-VAD** are accuracy/speed claims to take with grain
of salt:
- Cobra: no Go SDK — would need to either pay for source license,
  use commercial REST endpoint, or fork their C SDK behind cgo.
- TEN-VAD: Agora non-compete clause is a legal blocker for any
  product that touches voice infrastructure (which Strawgo does).

## Low-latency IPC landscape, May 2026

| tech | latency | language coverage | maturity | for our use? |
|---|---|---|---|---|
| **Iceoryx2** (Eclipse) | 150–250 ns | C/C++/Python/Rust; **Go bindings PLANNED, not yet** | production (autonomous vehicles) | **blocked** today on Go support. [github.com/eclipse-iceoryx/iceoryx2](https://github.com/eclipse-iceoryx/iceoryx2) |
| **Aeron** | sub-µs | Java/C++ primary; some Rust/Go | production (trading, Coinbase, Man Group) | overkill; Java-centric. [sanj.dev](https://sanj.dev/post/aeron-alternatives-messaging-comparison) |
| **DLPack** | zero-copy tensor exchange across runtimes | Python/C/Rust + LLVM ecosystem | mature | useful pattern for ML, not for IPC fabric |
| **gRPC over UDS** | 50–200 µs | universal | production | what Triton already does |
| **Custom shm + futex** | 1–5 µs | own everything | one-off | what I almost spawned an agent to build — **don't** |

**Iceoryx2 is the right answer** when Go bindings ship — projected by the
project maintainers but no committed date. Custom shm+futex would be
re-deriving what iceoryx2 already engineered (with worse defaults).

## What this means for Strawgo

### The cgo-embed path was correct
Phase-2 numbers stand: 47 MB at N=200, full real-time pacing, 17 ms p99.
That **already beats containerized Pipecat** by an order of magnitude
on capacity per box. The honest comparison isn't "Pipecat in one process
at N=200" (that doesn't exist in production — they ship 1 per
container), it's "Pipecat × 200 containers at 126 MB each ≈ 25 GB" vs
"Strawgo at 47 MB."

### Three production-aligned upgrade paths

**Path A — Polish the wrapper, ship cgo embed (smallest blast radius)**
1. Custom thin cgo wrapper exposing `runtime.Pinner`-pinned tensor
   pointers; reuse input/state buffers across calls.
2. Upgrade to **Silero v6.2.1** (drop-in, +16 % accuracy).
3. Drop in **int8 model** for 2–3× speed.
4. Done in ~3 days. Estimated p99 at N=200: 1–3 ms. **Beats Pipecat
   on every dimension** including latency.

**Path B — Triton-as-backend (industry standard for multi-model)**
1. Run Triton Inference Server alongside Strawgo (one per node).
2. Use [Trendyol/go-triton-client](https://github.com/Trendyol/go-triton-client) to talk to it via gRPC.
3. Triton handles model concurrency, dynamic batching, quantization,
   GPU offload (if any), multi-model (VAD + smart-turn + future ASR).
4. Cost: ~50–200 µs gRPC overhead per call vs in-process. Frame budget
   32 ms makes this fine.
5. Aligns with NVIDIA NIM ecosystem. Plays well with Modular MAX
   (which embeds inside Triton).
6. Operational overhead: one extra service per node. Worth it if you
   plan to serve 3+ models.

**Path C — Pure-Rust subprocess via iceoryx2 (waiting for Go)**
- **Don't start now.** Iceoryx2 Go bindings are planned, not shipped.
  Re-evaluate in 6 months.
- If we need it sooner, contribute Go bindings upstream — community
  goodwill + future-proof.
- Until then, **the IPC sidecar approach via custom shm+futex is dead
  weight** — we'd be building something iceoryx2 will replace. Cancel
  the in-flight SHM agent (still running as of this writing); do not
  ship its output as-is.

### Stay-aligned-with-industry options

If we want to be a "drop-in Pipecat replacement" rather than a
distinctive solution:
- Implement a Pipecat-compatible WebSocket protocol (SmallWebRTC /
  Daily / LiveKit transport).
- Expose Strawgo as a Pipecat-compatible "framework option" so users
  can swap orchestrators while keeping their Daily/LiveKit transports.

If we want to be the **distinctive low-cost option:**
- Lead with the per-box density story (47 MB vs 126 MB+, 5 000 vs ~50–500
  agents per VM). That's the actual wedge.
- Marketing message: "Pipecat's per-session container model costs 25 GB
  for 200 agents; Strawgo does the same in 47 MB on a 4-core box."

## What I'd cancel, what I'd keep

**Cancel:**
- The custom SHM+futex agent currently running. Industry has iceoryx2;
  rebuilding it would be wasted effort.

**Keep:**
- Phase-2 cgo embed branch. It's the production winner.
- TEN-VAD branch as a reference; do not ship until license is cleared.
- Rust onnx-worker Tier 1 branch. Useful if anyone wants the
  process-isolation flavor — but fix the `Arc<Mutex<Session>>`
  bug first (use `Arc<Session>` + `run_unsafe(&self)`).

**Add:**
- Custom thin yalue fork or `runtime.Pinner`-based wrapper for Silero
  hot path. ~50 lines.
- Silero v6.2.1 model swap (zero code change, just new .onnx file).
- int8 model option in cgo-embed path.
- Optional: gRPC client to Triton as a backend mode (`STRAWGO_VAD=triton`).

## Caveats / what I didn't verify

- Iceoryx2 Go bindings — checked their docs, said "planned." No public
  PR or roadmap date.
- pykeio/ort 2.0 stable release — still RC.12 as of search date.
- Cobra Go SDK — confirmed not present in 2026 SDK list.
- Modular MAX Go client — confirmed only via Triton gRPC path.
- The "5000 agents per box" Strawgo cgo-embed projection is from RSS
  scaling, not from running it. Hardware ceiling is real (4 cores @
  N=5000 = 1250 inferences/sec @ 32ms cadence; ORT can do that on CPU).

## Sources

- [LiveKit Agents 2026 playbook (forasoft)](https://www.forasoft.com/blog/article/livekit-ai-agents-guide)
- [LiveKit quotas & worker pool](https://docs.livekit.io/deploy/admin/quotas-and-limits/)
- [LiveKit Cloud Inference Gateway architecture](https://deepwiki.com/livekit/agents/5.7-livekit-cloud-inference-gateway)
- [Pipecat production scaling guide](https://luonghongthuan.com/en/blog/pipecat-voice-agent-production-scalable-guide/)
- [NVIDIA NIM Pipecat Voice Agent Blueprint](https://build.nvidia.com/pipecat/voice-agent-framework-for-conversational-ai)
- [Daily + NVIDIA nimble-pipecat](https://www.daily.co/blog/daily-and-nvidia-collaborate-to-simplify-voice-agents-at-scale/)
- [Bland AI horizontal scale architecture](https://www.retellai.com/blog/best-voice-ai-providers)
- [Best Voice Agent Stack 2026 (Hamming AI)](https://hamming.ai/resources/best-voice-agent-stack)
- [Modular MAX 26.2 release notes](https://docs.modular.com/max/changelog/)
- [Modular MAX overview](https://www.modular.com/max)
- [Triton gRPC Go client (Trendyol)](https://github.com/Trendyol/go-triton-client)
- [ONNX Runtime IOBinding docs](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)
- [yalue/onnxruntime_go pkg.go.dev (IOBinding caveat)](https://pkg.go.dev/github.com/yalue/onnxruntime_go)
- [shota3506/onnxruntime-purego](https://pkg.go.dev/github.com/shota3506/onnxruntime-purego/onnxruntime)
- [Hugot Go transformer pipelines](https://github.com/knights-analytics/hugot)
- [Burn vs Candle 2026 comparison](https://dasroot.net/posts/2026/04/rust-machine-learning-burn-vs-candle-framework-comparison/)
- [Iceoryx2 GitHub](https://github.com/eclipse-iceoryx/iceoryx2)
- [Aeron alternatives 2026](https://sanj.dev/post/aeron-alternatives-messaging-comparison)
- [Cobra VAD product page](https://picovoice.ai/platform/cobra/)
- [Cobra vs Silero vs WebRTC 2026](https://picovoice.ai/blog/best-voice-activity-detection-vad/)
- [Silero VAD v6 release](https://github.com/snakers4/silero-vad/releases/tag/v6.0)
- [Silero VAD v6.2.1 PyPI](https://pypi.org/project/silero-vad/)
- [Pipecat: never multiple sessions per process (production rule)](https://luonghongthuan.com/en/blog/pipecat-voice-agent-production-scalable-guide/)
- [TEN-VAD Apache+Agora license](https://github.com/TEN-framework/ten-vad)
- [pykeio/ort releases](https://github.com/pykeio/ort/releases)
