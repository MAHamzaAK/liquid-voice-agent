# Phase II Implementation Summary

## Overview

Phase II addresses four core problems identified in Phase I:
1. **Model Loading Time** - Cold start was 15-30 seconds
2. **Repetition Issues** - Same situation/principle detected repeatedly
3. **Testing Interface** - Needed Streamlit UI with debug visualization
4. **Semantic Selection** - Replace keyword matching with Pinecone vector search

## What Was Implemented

### 1. Model Loading Time - Warm Pool Configuration

**File:** `src/server.py`

Added Modal warm pool configuration:
```python
@app.cls(
    ...
    min_containers=1,      # Keep 1 container warm
    buffer_containers=1,   # Extra buffer when active
    scaledown_window=300,  # 5 min idle before scale down
)
```

**Cost:** ~$1.50-2.00/hour for warm L40S container

### 2. Semantic Detection & Selection

#### New Modules Created

| File | Purpose |
|------|---------|
| `src/embeddings.py` | BGE-small-en-v1.5 embedding model (384 dims) |
| `src/pinecone_client.py` | Pinecone operations for situations & principles |
| `src/context.py` | Conversation tracking across turns |
| `src/principle_scorer.py` | Multi-factor scoring (semantic + recency + stage + random) |

#### Updated Modules

| File | Changes |
|------|---------|
| `src/detector.py` | Added `detect_situation_semantic()` with Pinecone query |
| `src/selector.py` | Added `select_principle_semantic()` with scoring integration |

#### Scoring Formula

Principle selection now uses weighted scoring:
- **Semantic relevance** (40%): Cosine similarity to transcript
- **Recency penalty** (30%): Negative weight for recently used principles
- **Stage fit** (20%): Bonus for principles matching current sales stage
- **Random variation** (10%): Prevents deterministic selection

### 3. Pinecone Integration

**Index Structure:**
- Index name: `sales-coach-embeddings`
- Dimension: 384 (BGE-small-en-v1.5)
- Namespaces: `situations`, `principles`

**Population Script:** `scripts/populate_pinecone.py`

### 4. Streamlit Interface

**Files:**
- `streamlit_app/app.py` - Main app with conversation UI
- `streamlit_app/components/debug_panel.py` - Score visualization

**Features:**
- Browser microphone recording via `st.audio_input()`
- File upload for pre-recorded test audio
- Conversation history with playback
- Debug panel showing:
  - Situation detection scores
  - Principle selection scores with breakdown
  - Context state (stage, turn count, recent principles)

### 5. Dependencies & Configuration

**Added to `pyproject.toml`:**
```toml
"streamlit>=1.30.0",
"pinecone-client>=3.0.0",
"sentence-transformers>=2.2.0",
```

**Added to `.env.example`:**
```bash
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=sales-coach-embeddings
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -e .
```

### 2. Configure Pinecone
1. Create a Pinecone account at https://app.pinecone.io/
2. Create an API key
3. Copy `.env.example` to `.env` and add your key:
   ```bash
   PINECONE_API_KEY=your_key_here
   PINECONE_INDEX_NAME=sales-coach-embeddings
   ```

### 3. Populate Pinecone Index
```bash
python scripts/populate_pinecone.py
```

This embeds all situations and principles and uploads them to Pinecone.

### 4. Deploy to Modal
```bash
modal deploy src/server.py
```

### 5. Run Streamlit App
```bash
streamlit run streamlit_app/app.py
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT APP                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────┐  ┌──────────────────────────────┐ │
│  │ Conversation Panel          │  │ Debug Panel                  │ │
│  │ - Audio recording           │  │ - Situation scores           │ │
│  │ - File upload               │  │ - Principle scores + breakdown│
│  │ - History + playback        │  │ - Context state              │ │
│  └──────────────┬──────────────┘  └──────────────────────────────┘ │
└─────────────────┼───────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MODAL (GPU - L40S, Warm Pool)                    │
├─────────────────────────────────────────────────────────────────────┤
│  1. Transcribe audio (LFM2.5-Audio)                                 │
│  2. Embed transcript (BGE-small)                                    │
│  3. Query Pinecone for situations ──────────────┐                   │
│  4. Score principles (semantic + recency + stage)│                   │
│  5. Generate response (LFM2.5-Audio)            │                   │
│  6. Return coaching output                       │                   │
└──────────────────────────────────────────────────┼──────────────────┘
                                                   │
                                                   ▼
                                    ┌──────────────────────────────────┐
                                    │            PINECONE               │
                                    │  - situations namespace (~200)    │
                                    │  - principles namespace (~80)     │
                                    └──────────────────────────────────┘
```

## File Structure

```
liquid-audio-model/
├── src/
│   ├── server.py           # Modal server (updated with warm pool)
│   ├── detector.py         # Situation detection (semantic + keyword)
│   ├── selector.py         # Principle selection (semantic + first-match)
│   ├── embeddings.py       # NEW: BGE embedding model
│   ├── pinecone_client.py  # NEW: Pinecone operations
│   ├── context.py          # NEW: Conversation tracking
│   ├── principle_scorer.py # NEW: Multi-factor scoring
│   └── ...
├── streamlit_app/
│   ├── app.py              # NEW: Main Streamlit app
│   └── components/
│       └── debug_panel.py  # NEW: Score visualization
├── scripts/
│   └── populate_pinecone.py # NEW: Index population script
├── pyproject.toml          # Updated with new deps
└── .env.example            # Updated with Pinecone vars
```

## Next Steps

1. **Test the Streamlit app** with real audio
2. **Tune the scoring weights** based on observed behavior
3. **Adjust similarity thresholds** if detection is too loose/strict
4. **Consider adding:**
   - Streaming audio responses
   - Export conversation as JSON/YAML
   - Multi-user session persistence
