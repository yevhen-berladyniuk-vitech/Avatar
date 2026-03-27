# Mentorship Project
## AI Futuristic Holographic Speaking Avatar (Face Only)

**Track:** Open-Source Avatar Pipeline Research
**Phase:** Phase 1 — Offline Prototype

---

## Objective

Design and implement a system that generates a **speaking avatar (face only)** from:
- a single photo, OR
- a short video (30–120 sec)

The avatar must:
- resemble the original person
- speak provided text (via TTS or audio input)
- be rendered in a **futuristic / holographic style**

---

## Scope

### Phase 1 — Offline Pipeline (Required)

The system must:

1. **Accept input:**
   - image OR short video clip of a face
   - text string (to be spoken aloud)

2. **Generate output:**
   - an MP4 video with:
     - synchronized lip movement
     - basic facial animation
     - consistent identity across frames

3. **Remove or replace the background:**
   - isolate the face/head from the original background
   - replace with a neutral dark background OR a synthetic environment (e.g. space, grid, abstract)
   - background must not bleed through or flicker at face edges

4. **Apply a visual style layer:**
   - at least one futuristic effect (hologram glow, scanlines, neon edges, semi-transparency)

### Phase 2 — Real-Time Streaming (Optional / Bonus)

Integrate the pipeline as a **real-time video stream** delivered over WebRTC (e.g. via LiveKit), enabling the avatar to stream live inside a video call room.

Target architecture:

```
TTS audio output
        ↓
Real-time talking head model
        ↓
Style shader (GPU)
        ↓
WebRTC VideoSource → published to room
```

Target latency: ≤ 500ms end-to-end (audio in → first frame out).

---

## Technical Constraints

| Constraint | Requirement |
|---|---|
| Hardware target | Apple Silicon (Mac M1–M4) |
| PyTorch backend | MPS preferred |
| Cloud GPU dependency | None for Phase 1 |
| Phase 1 runtime target | ≤ 2 min per 10-second output clip |
| Phase 2 latency target | ≤ 500ms end-to-end |
| Output format | MP4 (Phase 1), raw frames / H.264 stream (Phase 2) |

---

## Recommended Model Stack

### Talking Head / Animation

| Model | Notes |
|---|---|
| **SadTalker** | Primary baseline. Accepts image + audio. MPS-compatible. |
| First Order Motion Model | Optional comparison baseline. Video-to-video. |
| DiffTalk / SyncTalk | Newer diffusion-based alternatives to evaluate. |

### Lip Sync Refinement

| Model | Notes |
|---|---|
| **Wav2Lip** | Industry standard post-processing. High lip-sync accuracy. |

### TTS (Audio Generation)

| Model | Notes |
|---|---|
| **Kokoro TTS** | Fast, high quality, MPS-friendly. |
| Coqui XTTS v2 | Voice cloning from reference audio. |
| Parler TTS | Lightweight alternative. |

### Background Removal / Matting

| Model | Notes |
|---|---|
| **rembg (u2net / u2net_human_seg)** | Fast, CPU/MPS-friendly. Best starting point for static images. |
| **BackgroundMattingV2** | High-quality video matting with temporal consistency. Requires a known background frame. |
| **RobustVideoMatting (RVM)** | Real-time capable (~30fps on M2). Human-specific, handles fine hair detail. Best option for video input. |
| MediaPipe Selfie Segmentation | Lightweight, no model download. Lower quality edges but very fast. |

> **Recommendation:** Use RobustVideoMatting for video input (best quality + temporal stability). Use rembg for single-image input. Evaluate edge quality carefully — poor matting degrades the holographic effect.

### Style / Effects

| Tool | Notes |
|---|---|
| ComfyUI | Flexible, node-based. Good for hologram post-processing. |
| OpenCV + FFmpeg | Lightweight shader pipeline. Easier to integrate into CLI. |

---

## System Architecture (Baseline)

```
Input (image/video + text)
        ↓
[TTS module]         →  audio (.wav)
        ↓
[SadTalker]          →  raw talking-head video (original background)
        ↓
[Background removal] →  face/head on transparent or dark background (RVM / rembg)
        ↓
[Wav2Lip]            →  lip-sync refined video
        ↓
[Style layer]        →  hologram effect applied
        ↓
Output: final.mp4
```

> Note: background removal should happen **before** Wav2Lip so that the lip-sync model operates on a clean face region without background clutter confusing its face detector.

---

## Visual Requirements

The avatar must:
- be **face-focused** (no full body required)
- maintain **identity consistency** across frames (same person, no flickering identity)
- have **clean background separation** — no background bleed, no edge artifacts at hair/face boundaries
- include **at least one** futuristic visual style:
  - hologram glow (cyan/blue tint + bloom)
  - scanlines overlay
  - neon edge detection (Sobel / Canny on face mask)
  - semi-transparent ghost effect

Style must **not** destroy lip sync legibility. The face must remain readable.

---

## Evaluation Criteria

### 1. Identity Preservation
- Does the output still look like the input person?
- Metric: FaceNet cosine similarity between input and output frames (threshold ≥ 0.6)

### 2. Lip Sync Accuracy
- Does mouth movement match the speech audio?
- Metric: Wav2Lip confidence score; manual A/B review

### 3. Temporal Stability
- Are there flickering faces, identity drift between frames, or jitter?
- Metric: frame-to-frame L2 distance on face landmarks (MediaPipe)

### 4. Performance
- What is the runtime on Mac M-series?
- Metric: seconds of processing per second of output (target: ≤ 12× slower than realtime for Phase 1)

### 5. Visual Quality
- Does the holographic style enhance or distract?
- Metric: subjective 1–5 rating from a panel of 3+ reviewers

---

## Research Tasks (Optional)

Mentee may document findings for each:

1. **Model comparison** — SadTalker vs at least one alternative. Compare quality, speed, and MPS compatibility.
2. **Input modality comparison** — image vs 5-sec video vs 30-sec video as input. Which produces better temporal consistency?
3. **TTS quality impact** — test with multiple TTS backends. Does synthetic voice quality affect perceived realism of lip sync?
4. **Style exploration** — test at least 3 holographic style variations. Document the realism vs stylization tradeoff.
5. **Identity drift analysis** — measure whether identity consistency degrades over longer clips (30s+).
6. **Background matting quality** — compare rembg vs RobustVideoMatting on the same input. Evaluate edge quality at hair boundaries, temporal flicker, and processing speed.

---

## Deliverables

| # | Deliverable | Notes |
|---|---|---|
| 1 | Working CLI prototype | `python run.py --input face.jpg --text "Hello" --style hologram` |
| 2 | Sample outputs (video) | 3+ subjects, 10–30 sec each |
| 3 | Evaluation report | Model comparison, metrics, limitations |
| 4 | Phase 2 recommendation | Architecture proposal for real-time streaming integration |
| 5 | Starter repo | Documented, reproducible setup on Apple Silicon |

---

## Grading Rubric

| Category | Weight | Criteria |
|---|---|---|
| **Functional prototype** | 30% | Pipeline runs end-to-end. Produces valid MP4 output. |
| **Identity preservation** | 20% | FaceNet similarity ≥ 0.6. Face is recognizably the same person. |
| **Lip sync quality** | 20% | Mouth movement matches audio. No obvious desync. |
| **Visual style** | 10% | At least one holographic effect applied. Style readable, not destructive. |
| **Performance** | 10% | Runs on Mac M-series within time target. Documented benchmark. |
| **Research depth** | 10% | Model comparison documented. Findings actionable. Recommendations specific. |

**Bonus (+10%):** Phase 2 prototype — avatar streams as a real-time video track in a WebRTC test room.

---

## Starter Repo Structure (Suggestion)

```
avatar-poc/
├── README.md               # Setup, usage, benchmark results
├── requirements.txt
├── run.py                  # CLI entry point
├── pipeline/
│   ├── tts.py              # TTS wrapper (Kokoro / XTTS)
│   ├── talking_head.py     # SadTalker wrapper
│   ├── background.py       # Background removal (rembg / RVM)
│   ├── lip_sync.py         # Wav2Lip wrapper
│   └── style.py            # Hologram effect (OpenCV / ComfyUI)
├── streaming/
│   └── avatar_publisher.py # Phase 2: WebRTC VideoSource track (bonus)
├── eval/
│   ├── identity_score.py   # FaceNet cosine similarity
│   ├── temporal_stability.py
│   └── benchmark.py        # Runtime measurement
├── samples/
│   ├── input/              # Test face images / clips
│   └── output/             # Generated videos
└── report/
    └── findings.md         # Model comparison, metrics, conclusions
```

---

## Key Insight for Mentee

Separate the problem into three independent layers. Each can be swapped without rebuilding the others:

| Layer | Responsibility | Primary model |
|---|---|---|
| **Motion** | Make the face move realistically | SadTalker |
| **Identity** | Keep it looking like the same person | Wav2Lip + FaceNet verification |
| **Isolation** | Separate face from background cleanly | RobustVideoMatting / rembg |
| **Style** | Apply the holographic aesthetic | OpenCV / ComfyUI |

Mastering the separation between these layers is the core skill of this project. A good pipeline makes each layer independently replaceable — so that swapping SadTalker for a newer model in the future doesn't require rewriting the lip sync or style code.

---

## Phase 2 Context (Future Work)

Phase 2 targets real-time delivery:
- Replace offline models with streaming-capable alternatives (e.g. Live-Avatar)
- Publish the avatar video as a WebRTC track in a live call room
- Support multi-user rooms
- Target: ≤ 500ms latency from TTS audio output to displayed video frame