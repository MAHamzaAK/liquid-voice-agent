# Behavioral Psychology Sales Coach - Voice Chatbot

## Project Overview

**Goal**: Build a voice chatbot that coaches salespeople using behavioral psychology principles. The system listens to customer conversations, detects the situation, selects appropriate psychological principles, and provides both audio responses and structured coaching explanations.

**Key Principle**: Keep it simple. Get end-to-end working first, optimize later.

---

## Three-Phase Approach

| Phase | Goal | Status |
|-------|------|--------|
| **Phase 1** | End-to-end pipeline working | ✅ Complete |
| **Phase 2** | Semantic detection, Pinecone, Streamlit UI | ✅ Complete |
| **Phase 3** | Real-time coaching (~1.3s), deep context tracking | 🔄 Current |

**Current Focus: Phase 3**

---

## Phase Summary

| Phase | Time to Coaching | Detection | UI |
|-------|------------------|-----------|-----|
| Phase 1 | ~6s (after voice) | Keyword matching | CLI |
| Phase 2 | ~6s (after voice) | Semantic (Pinecone) | Streamlit |
| **Phase 3** | **~1.3s** (before voice) | Context-aware semantic | Streamlit + SSE |

---

## Data Assets (Already Available)

### 1. principles.json
~80+ behavioral psychology principles from:
- Cialdini's "Influence"
- Voss's "Never Split the Difference"
- Kahneman's "Thinking, Fast and Slow"

### 2. situations.json
~50+ sales situations with:
- signals (what customer says)
- applicable_principles (which principles to use)
- typical_stage (discovery, demo, objection_handling, negotiation, closing)

### 3. Audio Format Requirements (LFM2.5-Audio)
- **Input**: Any sample rate (torchaudio handles conversion internally)
- **Output**: 24kHz (Mimi codec output)
- **Format**: WAV files via torchaudio
- **Language**: English only

---

# PHASE 1: End-to-End Pipeline

## Architecture (Simple)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1 ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   LOCAL (Your Machine)                                                       │
│   ┌──────────────┐                                                          │
│   │ Microphone   │──▶ Record audio until silence                            │
│   └──────────────┘                                                          │
│          │                                                                   │
│          ▼                                                                   │
│   ┌──────────────┐                                                          │
│   │ Upload to    │──▶ Modal Volume                                          │
│   │ Modal        │                                                          │
│   └──────────────┘                                                          │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   MODAL (Cloud GPU)                                                          │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                      SINGLE FUNCTION                                  │  │
│   │                                                                        │  │
│   │  1. TRANSCRIBE ──▶ LFM2.5-Audio ASR                                   │  │
│   │        │                                                               │  │
│   │        ▼                                                               │  │
│   │  2. DETECT ──▶ Simple keyword matching against situations.json        │  │
│   │        │                                                               │  │
│   │        ▼                                                               │  │
│   │  3. SELECT ──▶ Pick first applicable principle (no fancy scoring)     │  │
│   │        │                                                               │  │
│   │        ▼                                                               │  │
│   │  4. GENERATE ──▶ LFM2.5-Audio with principle in system prompt         │  │
│   │        │                                                               │  │
│   │        ▼                                                               │  │
│   │  5. FORMAT ──▶ Build coaching output YAML                             │  │
│   │                                                                        │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   LOCAL (Your Machine)                                                       │
│   ┌──────────────┐     ┌──────────────────────────────────────────────┐    │
│   │ Play Audio   │     │ Display Coaching Output                      │    │
│   │ Response     │     │ (principle, source, why_it_works)            │    │
│   └──────────────┘     └──────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1 Components

### Component 1: Situation Detector (Simple Keyword Matching)

```python
# Simple approach - no embeddings, no ML
def detect_situation(transcript: str, situations: dict) -> str:
    """
    Match transcript against situation signals using simple keyword matching.
    Returns first matching situation.
    """
    transcript_lower = transcript.lower()

    for situation_id, data in situations.items():
        for signal in data["signals"]:
            if signal.lower() in transcript_lower:
                return situation_id

    return "general_inquiry"  # Default fallback
```

### Component 2: Principle Selector (Simple Lookup)

```python
import json

def load_principles(path: str = "principles.json") -> dict:
    """
    Load principles from JSON file.
    Note: principles.json is an array, convert to dict keyed by principle_id.
    """
    with open(path, "r") as f:
        principles_list = json.load(f)
    return {p["principle_id"]: p for p in principles_list}

# Simple approach - just pick the first applicable principle
def select_principle(applicable_principles: list[str], principles: dict) -> dict:
    """
    Pick first applicable principle from the situation.
    No scoring, no optimization.
    """
    if not applicable_principles:
        return None
    principle_id = applicable_principles[0]
    return principles.get(principle_id)
```

### Component 3: Response Generator (LFM2.5-Audio)

```python
# Use LFM2.5-Audio with principle injected into system prompt
system_prompt = f"""
You are a helpful sales assistant. Respond to the customer using this approach:

PRINCIPLE: {principle["name"]}
APPROACH: {principle["intervention"]}
EXAMPLE: {principle["example_response"]}

Respond naturally and conversationally. Keep it brief (2-3 sentences).
Respond with interleaved text and audio.
"""
```

### Component 4: Coaching Output Formatter

```python
# Simple YAML output
def format_output(turn, transcript, situation, principle, response):
    return {
        "turn": turn,
        "timestamp": datetime.now().isoformat(),
        "customer_said": transcript,
        "detected_situation": situation["id"],
        "recommendation": {
            "principle": principle["name"],
            "source": f"{principle['source']['author']}, {principle['source']['book']}, Ch.{principle['source']['chapter']}",
            "response": response,
            "why_it_works": principle["mechanism"]
        }
    }
```

---

## Project Structure (Current)

```
liquid-audio-model/
├── PROJECT_PLAN.md              # This document (master plan)
├── PHASE1_IMPLEMENTATION.md     # Phase 1 details
├── PHASE2_IMPLEMENTATION.md     # Phase 2 details
├── PHASE3_IMPLEMENTATION.md     # Phase 3 details (current focus)
│
├── principles.json              # Psychology principles (~80)
├── situations.json              # Sales situations (~50)
├── pyproject.toml               # Dependencies
├── .env                         # Local environment variables (DO NOT COMMIT)
├── .env.example                 # Environment template
│
├── src/
│   ├── __init__.py
│   │
│   │── # Core Logic
│   ├── detector.py              # Situation detection (keyword + semantic)
│   ├── selector.py              # Principle selection (first-match + semantic)
│   ├── formatter.py             # Coaching output formatter
│   │
│   │── # Phase 2 Additions
│   ├── embeddings.py            # BGE-small-en-v1.5 embeddings
│   ├── pinecone_client.py       # Pinecone vector operations
│   ├── context.py               # Conversation context tracking
│   ├── principle_scorer.py      # Multi-factor principle scoring
│   │
│   │── # Audio
│   ├── audio_recorder.py        # Record from microphone
│   ├── audio_player.py          # Play response audio
│   ├── file_manager.py          # Upload/download Modal volumes
│   │
│   │── # Infrastructure
│   ├── modal_app.py             # Modal configuration
│   ├── server.py                # Modal server function (GPU)
│   └── client.py                # CLI client
│
├── streamlit_app/               # Phase 2 UI
│   ├── app.py                   # Main Streamlit app
│   └── components/
│       └── debug_panel.py       # Score visualization
│
├── scripts/
│   └── populate_pinecone.py     # Populate Pinecone index
│
└── assets/
    └── test_audio/              # Sample audio files
```

---

## Prerequisites & Setup

### 1. Required Accounts

| Service | Purpose | Sign Up |
|---------|---------|---------|
| **Modal** | Cloud GPU for LFM2.5-Audio | https://modal.com |
| **HuggingFace** | Download LFM2.5-Audio model | https://huggingface.co |

### 2. Required Secrets

#### HuggingFace Token
```bash
# Get token from: https://huggingface.co/settings/tokens
# Create a token with "Read" access
```

#### Modal Setup
```bash
# Install Modal CLI
pip install modal

# Authenticate (opens browser)
modal token new

# Create HuggingFace secret in Modal
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

### 3. Local Environment Setup

```bash
# Clone/navigate to project
cd /Users/kk/Documents/liquid-audio-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -e .

# Or with uv (faster)
uv sync
```

### 4. Environment Variables

Create `.env` file (DO NOT COMMIT):
```bash
# .env
HF_TOKEN=hf_your_huggingface_token_here
```

Add to `.gitignore`:
```
.env
venv/
__pycache__/
*.pyc
```

---

## Dependencies (pyproject.toml)

```toml
[project]
name = "behavioral-sales-coach"
version = "0.1.0"
description = "Voice chatbot with behavioral psychology coaching"
requires-python = ">=3.11"

dependencies = [
    # Core - liquid-audio is the official Liquid AI package
    "liquid-audio",
    "modal>=1.2.4",

    # Audio
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "pyaudio>=0.2.14",
    "pygame>=2.6.0",

    # Utilities
    "numpy>=2.0.0",
    "rich>=13.0.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

---

## Implementation Steps (Phase 1)

### Step 1: Project Setup
- [ ] Create pyproject.toml with dependencies
- [ ] Create project structure (folders, __init__.py files)
- [ ] Add .gitignore
- [ ] Create Modal account and authenticate
- [ ] Create HuggingFace token and add to Modal secrets
- [ ] Test Modal connection

### Step 2: Data Loaders
- [ ] Create function to load principles.json
- [ ] Create function to load situations.json
- [ ] Test loading both files

### Step 3: Core Logic (Simple)
- [ ] Implement simple keyword-based situation detector
- [ ] Implement simple principle selector (first match)
- [ ] Implement coaching output formatter
- [ ] Test with sample transcripts

### Step 4: Modal Server
- [ ] Create Modal app configuration
- [ ] Create Docker image with dependencies
- [ ] Create server function that:
  - Loads LFM2.5-Audio model
  - Transcribes audio (ASR)
  - Detects situation
  - Selects principle
  - Generates response
  - Returns audio + coaching output
- [ ] Deploy to Modal
- [ ] Test with sample audio file

### Step 5: Local Client
- [ ] Implement audio recorder (with silence detection)
- [ ] Implement audio player
- [ ] Implement file upload to Modal volume
- [ ] Implement file download from Modal volume
- [ ] Create main conversation loop
- [ ] Test end-to-end

### Step 6: Integration & Testing
- [ ] Run full conversation flow
- [ ] Test multiple turns
- [ ] Fix any issues
- [ ] Document usage

---

## How to Run (After Phase 1 Complete)

```bash
# 1. Deploy server to Modal (once)
modal deploy src/server.py

# 2. Run client
python -m src.client

# Or with modal run
modal run src/client.py
```

---

## Expected Output (Per Turn)

```yaml
turn: 1
timestamp: "2025-01-19T15:30:00Z"

customer_said: "That's too expensive, I saw it cheaper on Amazon"
detected_situation: "online_price_checking"

recommendation:
  principle: "Reference Point"
  source: "Kahneman, Thinking Fast and Slow, Ch.26"
  response: "I understand you're comparing prices. What you're getting here includes our 2-year warranty and free installation - that's about ₹5000 in value that online sellers don't include."
  why_it_works: "Customers evaluate based on reference points. By shifting the comparison to include total value, we change their reference frame."
```

**Audio**: Plays the response through speakers

---

## What Phase 1 Does NOT Include

These are intentionally left for Phase 2:

- ❌ Embedding-based situation detection
- ❌ Sophisticated principle scoring
- ❌ Persona detection
- ❌ Context tracking across turns
- ❌ Confidence scores
- ❌ Fallback principles
- ❌ Gradio UI
- ❌ Analytics/logging
- ❌ Custom hardware optimization

---

# PHASE 2: Semantic Detection & Streamlit UI ✅ COMPLETE

**Status**: Implemented. See `PHASE2_IMPLEMENTATION.md` for details.

### What Was Built

| Feature | Implementation |
|---------|----------------|
| Semantic detection | Pinecone vector search with BGE-small-en-v1.5 embeddings |
| Semantic selection | Multi-factor scoring (semantic + recency + stage + random) |
| Warm pool | Modal container stays warm (min_containers=1) |
| Streamlit UI | Voice chat interface with debug panel |
| Basic context | Turn tracking, recent principles |

### Files Created (Phase 2)

```
src/
├── embeddings.py          # BGE-small-en-v1.5 embedding model
├── pinecone_client.py     # Pinecone operations
├── context.py             # Basic conversation tracking
├── principle_scorer.py    # Multi-factor scoring

streamlit_app/
├── app.py                 # Main Streamlit app
└── components/
    └── debug_panel.py     # Score visualization

scripts/
└── populate_pinecone.py   # Index population
```

### Phase 2 Limitations (Solved in Phase 3)

- ❌ Coaching only shown after voice generation (~6s delay)
- ❌ No customer profile tracking
- ❌ No stage progression tracking
- ❌ Basic context (no signal extraction)

---

# PHASE 3: Real-time Coaching & Deep Context 🔄 CURRENT

**Status**: In progress. See `PHASE3_IMPLEMENTATION.md` for full details.

### Key Innovation

**Coaching tip in ~1.3s** (vs ~6s in Phase 2) by:
1. Skipping tip generation — use pre-written `quick_tip` from situations.json
2. SSE streaming — send coaching tip before voice generation completes

```
Phase 2: Audio → Transcribe → Detect → Select → Generate Voice → Show Tip (~6s)
Phase 3: Audio → Transcribe → Detect → Lookup Tip → SSE (~1.3s) → Voice (background)
```

### Phase 3a: Quick Win (~1.3s to tip)

| Task | File | Status |
|------|------|--------|
| Add `quick_tip` to all situations | `situations.json` | 🔲 Todo |
| Add `_get_quick_tip()` method | `src/server.py` | 🔲 Todo |
| Add `process_turn_streaming()` | `src/server.py` | 🔲 Todo |
| Create realtime panel component | `streamlit_app/components/` | 🔲 Todo |
| Update Streamlit for SSE | `streamlit_app/app.py` | 🔲 Todo |

### Phase 3b: Deep Context

| Task | File | Status |
|------|------|--------|
| Add `SalesStage` enum | `src/context.py` | 🔲 Todo |
| Add `CustomerProfile` dataclass | `src/context.py` | 🔲 Todo |
| Create signal extractor | `src/signal_extractor.py` | 🔲 Todo |
| Add `detect_situation_with_context()` | `src/detector.py` | 🔲 Todo |
| Add `select_principle_with_context()` | `src/selector.py` | 🔲 Todo |
| Add `best_stages`, `customer_fit` | `principles.json` | 🔲 Todo |

### Phase 3c: Local Whisper (~0.5s to tip)

| Task | File | Status |
|------|------|--------|
| Add Whisper dependency | `pyproject.toml` | 🔲 Todo |
| Create `LocalASR` class | `streamlit_app/local_asr.py` | 🔲 Todo |
| Add `process_transcript()` method | `src/server.py` | 🔲 Todo |

### Data Structure Changes (Phase 3)

**situations.json** — Add to each situation:
```json
{
  "price_shock_in_store": {
    "quick_tip": "Don't defend price. Acknowledge, then reframe total value.",
    "quick_tip_variants": {
      "high_price_sensitivity": "Lead with ROI and savings over time.",
      "repeat_objection": "Time for social proof — share a customer story."
    },
    "context_hints": {
      "increases_price_sensitivity": true,
      "indicates_objection": true
    }
  }
}
```

**principles.json** — Add to each principle:
```json
{
  "principle_id": "kahneman_anchors_01",
  "best_stages": ["objection_handling", "negotiation"],
  "customer_fit": {
    "high_price_sensitivity": 0.9,
    "high_urgency": 0.5
  }
}
```

### Success Criteria (Phase 3)

| Metric | Target |
|--------|--------|
| Time to coaching tip (lookup) | < 1.5 seconds |
| Time to coaching tip (local Whisper) | < 0.6 seconds |
| No principle repetition in 3 turns | 100% |
| Stage tracking accuracy | > 75% |

---

# PHASE 4: Future Considerations

- Voice tone analysis (detect frustration, excitement)
- Streaming audio playback (play while generating)
- A/B testing different principles
- Outcome tracking (did the tip help close?)
- Team analytics dashboard

---

## Files to Create (Phase 1 Checklist)

```
[ ] pyproject.toml
[ ] .gitignore
[ ] .env (local only)
[ ] src/__init__.py
[ ] src/detector.py
[ ] src/selector.py
[ ] src/formatter.py
[ ] src/audio_recorder.py
[ ] src/audio_player.py
[ ] src/file_manager.py
[ ] src/modal_app.py
[ ] src/server.py
[ ] src/client.py
```

---

## Quick Reference: Modal Commands

```bash
# Authenticate
modal token new

# Create secret
modal secret create huggingface-secret HF_TOKEN=hf_xxx

# Deploy app
modal deploy src/server.py

# Run function
modal run src/client.py

# View logs
modal app logs behavioral-sales-coach

# List apps
modal app list
```

---

## Quick Reference: Test Commands

```bash
# Test situation detection
python -c "from src.detector import detect_situation; print(detect_situation('too expensive'))"

# Test principle selection
python -c "from src.selector import select_principle; print(select_principle('price_shock_in_store'))"

# Run full pipeline locally (no audio, for testing)
python -m src.client --test
```

---

## Success Criteria (Phase 1)

| Criteria | Target |
|----------|--------|
| Audio recording works | Records until 2s silence |
| Audio uploads to Modal | File accessible on server |
| Situation detected | Returns valid situation ID |
| Principle selected | Returns principle from JSON |
| Response generated | Audio file created |
| Response plays | Audible on speakers |
| Coaching displayed | YAML shown in terminal |

---

## Troubleshooting

### "No microphone access"
```bash
# macOS: System Preferences > Security & Privacy > Privacy > Microphone
# Grant access to Terminal/VS Code
```

### "Modal authentication failed"
```bash
modal token new  # Re-authenticate
```

### "HuggingFace model access denied"
```bash
# 1. Accept model terms at: https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B
# 2. Check token has "Read" access
# 3. Recreate Modal secret
modal secret create huggingface-secret HF_TOKEN=hf_new_token
```

### "Out of GPU memory"
```bash
# Use smaller GPU or wait for availability
# Modal will auto-scale, just retry
```

---

## Quick Reference: Document Index

| Document | Purpose |
|----------|---------|
| `PROJECT_PLAN.md` | Master plan, phase overview, project structure |
| `PHASE1_IMPLEMENTATION.md` | Phase 1 implementation details (complete) |
| `PHASE2_IMPLEMENTATION.md` | Phase 2 implementation details (complete) |
| `PHASE3_IMPLEMENTATION.md` | Phase 3 implementation details (current) |

---

*Document created: 2025-01-19*
*Last updated: 2025-01-19*
*Current Phase: 3 (Real-time Coaching & Deep Context)*
