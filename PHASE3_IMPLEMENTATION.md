# Phase 3: Real-time Coaching & Deep Context

---

## Implementation Status

### Already Implemented (from Phase 2)

| Component | File | Status |
|-----------|------|--------|
| Basic context tracking | `src/context.py` | ✅ Exists (simpler structure) |
| Semantic detection | `src/detector.py` | ✅ Has semantic matching |
| Semantic selection | `src/selector.py` | ✅ Has semantic matching |
| Streamlit app | `streamlit_app/app.py` | ✅ Has voice chat interface |
| Server infrastructure | `src/server.py` | ✅ Has `process_turn()` method |
| Data files | `situations.json`, `principles.json` | ✅ Exist with basic structure |

### New Files to Create

| File | Purpose | Priority |
|------|---------|----------|
| `src/signal_extractor.py` | Extract customer signals (price sensitivity, urgency, etc.) | Phase 3b |
| `streamlit_app/components/realtime_panel.py` | Real-time coaching display component | Phase 3a |

### Files to Enhance

| File | Additions Needed | Priority |
|------|------------------|----------|
| `src/context.py` | `SalesStage` enum, `CustomerProfile` dataclass, recency tracking | Phase 3b |
| `src/detector.py` | `detect_situation_with_context()` with stage/repetition scoring | Phase 3b |
| `src/selector.py` | `select_principle_with_context()` with multi-factor scoring | Phase 3b |
| `src/server.py` | `process_turn_streaming()`, `_get_quick_tip()`, `_get_action_verb()` | Phase 3a |
| `streamlit_app/app.py` | SSE consumer, real-time UI updates, context panel | Phase 3a |
| `situations.json` | `quick_tip`, `quick_tip_variants`, `context_hints` per situation | Phase 3a |
| `principles.json` | `best_stages`, `customer_fit`, `sequence_hints` per principle | Phase 3b |

---

## Technical Notes & Caveats

### Modal Streaming Limitations

**Issue**: Modal doesn't have a `.remote_gen()` method for streaming generators.

**Solutions**:

1. **Web Endpoint with SSE** (Recommended):
```python
# src/server.py
@app.function(...)
@modal.web_endpoint(method="GET")
def process_turn_sse(session_id: str, context_json: str):
    """SSE endpoint for streaming events."""
    from starlette.responses import StreamingResponse

    def event_generator():
        # ... process and yield SSE events ...
        for event in process_turn_internal(session_id, context_json):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

2. **Polling Pattern** (Simpler):
```python
# Return all events at once, client displays progressively
@modal.method()
def process_turn(self, ...) -> list[dict]:
    events = []
    events.append({"event": "transcript", "data": {...}})
    # ... collect all events ...
    return events  # Client iterates and displays with delays
```

### Streamlit Execution Model

**Issue**: Streamlit reruns the entire script on each interaction. A blocking loop won't show progressive updates.

**Solution**: Use `st.empty()` placeholders with explicit updates:

```python
# Streamlit pattern for progressive display
def process_and_display(audio_bytes, session_id):
    # Create placeholders FIRST
    transcript_box = st.empty()
    situation_box = st.empty()
    coaching_box = st.empty()

    # Get all events at once from Modal
    SalesCoach = modal.Cls.from_name("behavioral-sales-coach", "SalesCoach")
    events = SalesCoach().process_turn.remote(
        session_id=session_id,
        context=st.session_state.context.to_dict()
    )

    # Display progressively with delays for effect
    for event in events:
        if event["event"] == "transcript":
            transcript_box.markdown(f"**You said:** {event['data']['text']}")
            time.sleep(0.3)  # Visual delay
        elif event["event"] == "situation":
            situation_box.info(f"Detected: {event['data']['name']}")
            time.sleep(0.3)
        elif event["event"] == "coaching_tip":
            coaching_box.success(f"💡 {event['data']['tip']}")
        # ... etc
```

### Volume Reload Requirement

**Issue**: Modal volumes need `volume.reload()` after external writes.

```python
# In server.py, before reading uploaded audio:
sessions_volume.reload()  # Ensure fresh data
wav, sr = torchaudio.load(audio_path)
```

### Type Consistency

**Pattern**: Use `dict` for JSON transfer over Modal, reconstruct `ConversationContext` on both ends:

```python
# Client → Server: serialize
context_dict = st.session_state.context.to_dict()
result = coach.process_turn.remote(context=context_dict)

# Server: reconstruct
ctx = ConversationContext.from_dict(context_dict)

# Server → Client: serialize back
return {"context": ctx.to_dict(), ...}

# Client: reconstruct
st.session_state.context = ConversationContext.from_dict(result["context"])
```

### Required Imports

Add these to the server code:

```python
import json
import time
from typing import Generator
import torch
import torchaudio
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

from .detector import detect_situation_semantic, DetectedSituation, SituationDetectionResult
from .selector import select_principle_semantic, SelectedPrinciple, PrincipleSelectionResult
from .context import ConversationContext
from .embeddings import embed_query  # Or however you expose this
```

---

## Existing Code Reference (Phase 2)

Understanding the current codebase before adding Phase 3 features:

### Return Types (Actual)

```python
# src/detector.py
class SituationDetectionResult:
    best_match: DetectedSituation
    all_candidates: list[DetectedSituation]
    method: str  # "semantic" or "keyword"

# src/selector.py
class PrincipleSelectionResult:
    selected: SelectedPrinciple
    all_candidates: list[dict]
    method: str  # "semantic" or "first_match"
```

### Function Signatures (Actual)

```python
# src/detector.py
def detect_situation_semantic(
    transcript: str,
    pinecone_client,
    embed_fn: callable,      # Required!
    situations: dict,
    top_k: int = 3,
    min_score: float = 0.5
) -> SituationDetectionResult

# src/selector.py
def select_principle_semantic(
    transcript: str,
    applicable_principles: list[str],
    principles: dict,
    pinecone_client,
    embed_fn: callable,      # Required!
    recent_principles: list[str],
    current_stage: str
) -> PrincipleSelectionResult
```

### ConversationContext (Actual)

```python
# src/context.py
class ConversationContext:
    session_id: str
    current_stage: str  # "discovery", "demo", "objection_handling", etc.
    turn_history: list[TurnRecord]

    # Properties
    turn_count: int
    used_principles: list[str]
    used_situations: list[str]
    recent_principles: list[str]  # Last 3

    # Methods
    def add_turn(transcript, detected_situation, situation_score, ...)
    def get_summary() -> str
    def to_dict() -> dict
    @classmethod from_dict(data: dict) -> ConversationContext
```

**Note**: Current context does NOT have `CustomerProfile`. That's a Phase 3b addition.

---

## Overview

Phase 3 introduces two major enhancements:

1. **Real-time Coaching Pipeline** — Deliver text-based coaching tips in ~2 seconds, before voice generation completes
2. **Deep Context Tracking** — Better situation detection through multi-turn conversation understanding

### Key Insight

The salesperson doesn't need to wait for voice generation. They need **immediate actionable guidance**:

```
Phase 2 Flow (5-6s to any output):
Audio → Transcribe → Detect → Select → Generate Voice+Text → Return All
                                              ↓
                                    [Wait 5-6 seconds]

Phase 3 Flow (2s to coaching, voice is bonus):
Audio → Transcribe ──┬──→ Detect ──→ Quick Tip (text-only) ──→ [SSE: 2s]
                     │
                     └──→ Voice Generation (background) ──→ [SSE: 5-6s]
```

---

## Architecture

### Server-Sent Events (SSE) Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STREAMLIT APP                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     REAL-TIME COACHING PANEL                         │   │
│   │                                                                      │   │
│   │   [~1.0s] 💬 "That's too expensive, I saw it cheaper online"        │   │
│   │                                                                      │   │
│   │   [~1.5s] 🎯 Situation: Price comparison with online competitor      │   │
│   │                                                                      │   │
│   │   [~2.0s] 💡 COACHING TIP                                           │   │
│   │           ┌─────────────────────────────────────────────────────┐   │   │
│   │           │ Use ANCHORING: Don't compete on price alone.        │   │   │
│   │           │                                                      │   │   │
│   │           │ Quick response: "I hear you. Online prices don't    │   │   │
│   │           │ include our 2-year warranty and free installation   │   │   │
│   │           │ — that's ₹5000 in value right there."               │   │   │
│   │           └─────────────────────────────────────────────────────┘   │   │
│   │                                                                      │   │
│   │   [~5.0s] 🔊 Voice Example Ready  [▶ Play]                          │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     CONTEXT PANEL                                    │   │
│   │                                                                      │   │
│   │   Turn: 3 of conversation                                           │   │
│   │   Stage: Objection Handling (was: Discovery → Demo)                 │   │
│   │   Customer Profile:                                                  │   │
│   │     - Price sensitive ████████░░ (high)                             │   │
│   │     - Decision maker: Uncertain                                      │   │
│   │     - Mentioned: "budget", "cheaper", "online"                      │   │
│   │   Recent Principles: Social Proof (T1), Scarcity (T2)               │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ SSE Connection
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MODAL SERVER (L40S GPU, Warm Pool)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   process_turn_streaming(session_id, context)                               │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STEP 1: Transcribe (ASR)                               [~1.0s]      │   │
│   │         LFM2.5 sequential generation, text-only                     │   │
│   │         ──→ yield SSE: {"event": "transcript", "data": "..."}       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                          │                                                   │
│                          ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STEP 2: Detect + Select (parallel)                     [~0.5s]      │   │
│   │         - Query Pinecone for situation (semantic)                   │   │
│   │         - Query Pinecone for principles (semantic)                  │   │
│   │         - Apply context-aware scoring                               │   │
│   │         ──→ yield SSE: {"event": "situation", "data": {...}}        │   │
│   │         ──→ yield SSE: {"event": "principle", "data": {...}}        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                          │                                                   │
│                          ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STEP 3: Quick Coaching (LOOKUP, INSTANT)               [~0.001s]    │   │
│   │         Lookup situation.quick_tip from situations.json             │   │
│   │         No model call needed!                                        │   │
│   │         ──→ yield SSE: {"event": "coaching_tip", "data": "..."}     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                          │                                                   │
│                          ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STEP 4: Voice Generation (INTERLEAVED)                 [~3-4s]      │   │
│   │         LFM2.5 interleaved generation                               │   │
│   │         Full response with audio                                     │   │
│   │         ──→ yield SSE: {"event": "voice_ready", "data": path}       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                          │                                                   │
│                          ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ STEP 5: Update Context                                              │   │
│   │         - Track principle usage                                      │   │
│   │         - Update customer profile signals                           │   │
│   │         - Detect stage transitions                                   │   │
│   │         ──→ yield SSE: {"event": "context_update", "data": {...}}   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Timeline Comparison

| Event | Phase 2 | Phase 3 (Generated) | Phase 3 (Lookup) |
|-------|---------|---------------------|------------------|
| Audio uploaded | 0.0s | 0.0s | 0.0s |
| Transcript available | — | **1.0s** ✓ | **1.0s** ✓ |
| Situation + Tip shown | — | **2.0s** | **1.3s** ✓ |
| Full response ready | 5-6s | 5-6s | 5-6s |

**Key Optimization**: Skip tip generation, use pre-written `quick_tip` from `situations.json`.

| Approach | Time to Tip | Model Calls | Notes |
|----------|-------------|-------------|-------|
| Generate tip with LFM2.5 | ~2.0s | 2 (ASR + tip) | Flexible but slow |
| Lookup `situation.quick_tip` | ~1.3s | 1 (ASR only) | **Recommended** |
| Local Whisper + Lookup | ~0.5s | 0 on server | Future optimization |

**Result**: Salesperson gets actionable guidance **4+ seconds earlier** than Phase 2.

---

## Part 1: Real-time Coaching Pipeline

### 1.1 Server Function (Polling Pattern)

**File**: `src/server.py`

Since Modal doesn't support generator streaming via `.remote_gen()`, we use a **polling pattern**:
the server returns all events as a list, and the client displays them progressively.

```python
import json
import time
import modal
import torch
import torchaudio
from typing import Optional
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

from .detector import detect_situation_semantic, SituationDetectionResult
from .selector import select_principle_semantic, PrincipleSelectionResult
from .context import ConversationContext

# ... Modal app, image, volume definitions ...

@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/sessions": sessions_volume, "/model_cache": models_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=1,
    timeout=60 * 10,
)
class SalesCoach:

    @modal.enter()
    def load_model(self):
        """Load model once when container starts."""
        # ... existing model loading code ...
        # Also load embed_query function from embeddings.py

    @modal.method()
    def process_turn_with_events(
        self,
        session_id: str,
        turn_number: int,
        context_dict: dict
    ) -> list[dict]:
        """
        Process turn and return all events as a list.

        Client displays these progressively for "streaming" effect.

        Returns list of events:
        - transcript: Customer's words
        - situation: Detected situation
        - principle: Selected principle
        - coaching_tip: Quick actionable tip (LOOKUP, fast)
        - voice_ready: Path to audio file
        - context_update: Updated conversation context
        """
        events = []

        # Reconstruct ConversationContext from dict
        context = ConversationContext.from_dict(context_dict)

        # Reload volume to get fresh uploaded audio
        sessions_volume.reload()

        # === STEP 1: Transcribe ===
        audio_path = f"/sessions/{session_id}/question.wav"
        wav, sr = torchaudio.load(audio_path)

        # ASR with sequential generation (text-only)
        chat = ChatState(self.processor)
        chat.new_turn("system")
        chat.add_text("Perform ASR.")
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(wav, sr)
        chat.end_turn()
        chat.new_turn("assistant")

        transcript_tokens = []
        for t in self.model.generate_sequential(**chat, max_new_tokens=256):
            if t.numel() == 1:
                transcript_tokens.append(t)

        transcript = self.processor.text.decode(torch.cat(transcript_tokens))

        events.append({
            "event": "transcript",
            "data": {"text": transcript, "timestamp": time.time()}
        })

        # === STEP 2: Detect Situation ===
        detection_result = detect_situation_semantic(
            transcript=transcript,
            pinecone_client=self.pinecone,
            embed_fn=self.embed_query,
            situations=self.situations,
            top_k=3,
            min_score=0.5
        )
        situation = detection_result.best_match

        events.append({
            "event": "situation",
            "data": {
                "id": situation.situation_id,
                "name": situation.situation_id.replace("_", " ").title(),
                "confidence": situation.confidence_score,
                "stage": situation.typical_stage,
                "matched_signal": situation.matched_signal
            }
        })

        # === STEP 3: Select Principle ===
        selection_result = select_principle_semantic(
            transcript=transcript,
            applicable_principles=situation.applicable_principles,
            principles=self.principles,
            pinecone_client=self.pinecone,
            embed_fn=self.embed_query,
            recent_principles=context.recent_principles,
            current_stage=context.current_stage
        )
        principle = selection_result.selected

        events.append({
            "event": "principle",
            "data": {
                "id": principle.principle_id,
                "name": principle.name,
                "source": f"{principle.author}, {principle.book}",
                "intervention": principle.intervention,
                "score": principle.selection_score
            }
        })

        # === STEP 4: Quick Coaching Tip (LOOKUP, INSTANT) ===
        quick_tip = self._get_quick_tip(situation, principle, context)

        events.append({
            "event": "coaching_tip",
            "data": {
                "tip": quick_tip,
                "principle_name": principle.name,
                "action_verb": self._get_action_verb(principle)
            }
        })

        # === STEP 5: Full Voice Response (SLOW) ===
        voice_path, voice_text = self._generate_voice_response(
            wav=wav,
            sr=sr,
            session_id=session_id,
            principle=principle
        )

        events.append({
            "event": "voice_ready",
            "data": {
                "path": voice_path,
                "text": voice_text
            }
        })

        # === STEP 6: Update Context ===
        context.add_turn(
            transcript=transcript,
            detected_situation=situation.situation_id,
            situation_score=situation.confidence_score,
            selected_principle=principle.principle_id,
            principle_score=principle.selection_score,
            response_text=voice_text
        )

        events.append({
            "event": "context_update",
            "data": context.to_dict()
        })

        return events

    def _get_quick_tip(
        self,
        situation: DetectedSituation,
        principle: SelectedPrinciple,
        context: ConversationContext  # Use actual class, not dict
    ) -> str:
        """
        Get quick coaching tip via LOOKUP (no generation).

        This is instant (~0ms) because:
        - No model call
        - Pre-written tips in situations.json
        - Falls back to principle.intervention if no quick_tip

        Priority:
        1. situation.quick_tip_variants[context_key] (context-specific)
        2. situation.quick_tip (default for situation)
        3. principle.intervention (fallback)

        Note: Phase 3a uses basic context. Phase 3b adds CustomerProfile for
        price_sensitivity and urgency_level checks.
        """
        situation_data = self.situations.get(situation.situation_id, {})

        # Check for context-specific variant
        variants = situation_data.get("quick_tip_variants", {})

        # Phase 3b: CustomerProfile checks (when implemented)
        # if hasattr(context, 'customer') and context.customer.price_sensitivity > 0.7:
        #     if "high_price_sensitivity" in variants:
        #         return variants["high_price_sensitivity"]

        # Check if this situation was detected recently (repeat objection)
        # Uses existing ConversationContext.used_situations property
        recent_situations = context.used_situations[-2:] if context.used_situations else []
        if situation.situation_id in recent_situations:
            if "repeat_objection" in variants:
                return variants["repeat_objection"]

        # Default quick_tip for situation
        if "quick_tip" in situation_data:
            return situation_data["quick_tip"]

        # Fallback to principle intervention
        return principle.intervention

    def _get_action_verb(self, principle: SelectedPrinciple) -> str:
        """Extract action verb for UI display."""
        action_verbs = {
            "anchoring": "Reframe",
            "loss_aversion": "Highlight risk",
            "social_proof": "Show evidence",
            "scarcity": "Create urgency",
            "reciprocity": "Offer value",
            "liking": "Build rapport",
            "authority": "Establish expertise",
            "labeling": "Acknowledge",
            "mirroring": "Reflect back",
        }

        for key, verb in action_verbs.items():
            if key in principle.principle_id.lower():
                return verb

        return "Apply"

    def _generate_voice_response(
        self,
        wav: torch.Tensor,
        sr: int,
        session_id: str,
        principle: SelectedPrinciple
    ) -> tuple[Optional[str], str]:
        """
        Generate voice response using LFM2.5 interleaved generation.

        This is the slow part (~3-4s). The coaching tip is already
        displayed by the time this completes.

        Returns:
            (audio_path, text_response)
        """
        system_prompt = f"""You are a helpful sales assistant. Respond using:

PRINCIPLE: {principle.name}
APPROACH: {principle.intervention}

Respond naturally in 2-3 sentences. Use interleaved text and audio."""

        chat = ChatState(self.processor)
        chat.new_turn("system")
        chat.add_text(system_prompt)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(wav, sr)
        chat.end_turn()
        chat.new_turn("assistant")

        text_tokens = []
        audio_tokens = []

        for t in self.model.generate_interleaved(
            **chat,
            max_new_tokens=512,
            audio_temperature=1.0,
            audio_top_k=4
        ):
            if t.numel() == 1:
                text_tokens.append(t)
            else:
                audio_tokens.append(t)

        response_text = self.processor.text.decode(torch.cat(text_tokens)) if text_tokens else ""

        audio_path = None
        if audio_tokens:
            audio_codes = torch.stack(audio_tokens[:-1], 1).unsqueeze(0)
            with torch.no_grad():
                waveform = self.processor.decode(audio_codes)

            audio_path = f"/sessions/{session_id}/answer.wav"
            torchaudio.save(audio_path, waveform.cpu(), 24_000)
            sessions_volume.commit()

        return audio_path, response_text
```

### 1.2 Streamlit Consumer (Polling Pattern)

**File**: `streamlit_app/app.py`

Uses polling pattern since Modal doesn't support generator streaming.

```python
import streamlit as st
import modal
import time
import tempfile
from datetime import datetime
from pathlib import Path

# Import from project
from src.context import ConversationContext
from src.file_manager import FileManager

# Constants
APP_NAME = "behavioral-sales-coach"
SESSIONS_VOLUME = "sales-coach-sessions"

# Page config
st.set_page_config(
    page_title="Sales Coach - Real-time",
    layout="wide"
)

# Initialize session state
if "context" not in st.session_state:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.context = ConversationContext(session_id=session_id)
    st.session_state.session_id = session_id

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Layout
col_main, col_context = st.columns([2, 1])

with col_main:
    st.title("🎯 Real-time Sales Coach")

    # Audio input
    audio_input = st.audio_input("Speak or upload audio")

    # Create placeholders BEFORE processing
    transcript_box = st.empty()
    situation_box = st.empty()
    coaching_box = st.empty()
    voice_box = st.empty()

with col_context:
    st.subheader("📊 Conversation Context")
    context_box = st.empty()


def upload_audio(audio_bytes: bytes, session_id: str) -> str:
    """Upload audio to Modal volume."""
    # Save to temp file first
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    # Upload to Modal volume
    file_manager = FileManager(
        volume_name=SESSIONS_VOLUME,
        session_id=session_id
    )
    file_manager.upload(temp_path, "question.wav")

    # Cleanup temp file
    Path(temp_path).unlink()

    return f"/{session_id}/question.wav"


def process_and_display(audio_bytes: bytes):
    """Process audio and display events progressively."""
    session_id = st.session_state.session_id
    ctx = st.session_state.context

    # Show processing indicator
    with st.spinner("Processing audio..."):
        # 1. Upload audio
        upload_audio(audio_bytes, session_id)

        # 2. Call Modal (returns list of events)
        SalesCoach = modal.Cls.from_name(APP_NAME, "SalesCoach")
        events = SalesCoach().process_turn_with_events.remote(
            session_id=session_id,
            turn_number=ctx.turn_count + 1,
            context_dict=ctx.to_dict()
        )

    # 3. Display events progressively (simulated streaming)
    for event in events:
        event_type = event["event"]
        data = event["data"]

        if event_type == "transcript":
            transcript_box.markdown(f"""
            ### 💬 Customer Said
            > "{data['text']}"
            """)
            time.sleep(0.2)  # Visual delay

        elif event_type == "situation":
            situation_box.info(
                f"🎯 **{data['name']}** | "
                f"Confidence: {data['confidence']:.0%} | "
                f"Stage: {data['stage']}"
            )
            time.sleep(0.2)

        elif event_type == "coaching_tip":
            coaching_box.success(f"""
            ### 💡 {data['action_verb'].upper()} — {data['principle_name']}

            {data['tip']}
            """)
            # This is the key moment - tip displayed ~1.3s after audio

        elif event_type == "voice_ready":
            if data.get("path"):
                voice_box.markdown("### 🔊 Voice Example Ready")
                # Download and play audio
                # file_manager.download("answer.wav", local_path)
                # voice_box.audio(local_path)

        elif event_type == "context_update":
            st.session_state.context = ConversationContext.from_dict(data)
            display_context(st.session_state.context)


def display_context(context: ConversationContext):
    """Display conversation context panel."""
    context_box.markdown(f"""
    **Turn:** {context.turn_count}

    **Stage:** {context.current_stage}

    **Recent Principles:**
    {', '.join(context.recent_principles) or 'None yet'}

    **Situations Detected:**
    {', '.join(context.used_situations[-3:]) or 'None yet'}
    """)


# Main flow
if audio_input:
    process_and_display(audio_input.getvalue())
```

---

## Part 2: Deep Context Tracking

### 2.1 Enhanced Context Model

**File**: `src/context.py`

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from enum import Enum


class SalesStage(Enum):
    DISCOVERY = "discovery"
    QUALIFICATION = "qualification"
    DEMO = "demo"
    OBJECTION_HANDLING = "objection_handling"
    NEGOTIATION = "negotiation"
    CLOSING = "closing"
    POST_SALE = "post_sale"


@dataclass
class CustomerProfile:
    """Tracks customer signals across conversation."""

    # Sensitivity indicators (0.0 to 1.0)
    price_sensitivity: float = 0.5
    urgency_level: float = 0.5
    decision_authority: float = 0.5  # Are they the decision maker?

    # Extracted information
    mentioned_budget: Optional[str] = None
    mentioned_timeline: Optional[str] = None
    mentioned_competitors: list[str] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)

    # Behavioral signals
    objection_count: int = 0
    positive_signals: int = 0
    questions_asked: int = 0


@dataclass
class ConversationContext:
    """Full conversation context for multi-turn awareness."""

    session_id: str
    turn_count: int = 0

    # Stage tracking
    current_stage: SalesStage = SalesStage.DISCOVERY
    stage_history: list[SalesStage] = field(default_factory=list)

    # Principle tracking
    principles_used: list[str] = field(default_factory=list)
    principle_timestamps: dict[str, datetime] = field(default_factory=dict)

    # Situation tracking
    situations_detected: list[str] = field(default_factory=list)

    # Customer profile
    customer: CustomerProfile = field(default_factory=CustomerProfile)

    # Transcript history (last N turns)
    transcript_history: list[dict] = field(default_factory=list)

    def add_turn(
        self,
        transcript: str,
        situation_id: str,
        principle_id: str,
        detected_stage: str
    ):
        """Record a conversation turn."""
        self.turn_count += 1

        # Update principle tracking
        self.principles_used.append(principle_id)
        self.principle_timestamps[principle_id] = datetime.now()

        # Update situation tracking
        self.situations_detected.append(situation_id)

        # Update stage
        new_stage = SalesStage(detected_stage)
        if new_stage != self.current_stage:
            self.stage_history.append(self.current_stage)
            self.current_stage = new_stage

        # Store transcript
        self.transcript_history.append({
            "turn": self.turn_count,
            "transcript": transcript,
            "situation": situation_id,
            "principle": principle_id,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only last 10 turns
        if len(self.transcript_history) > 10:
            self.transcript_history = self.transcript_history[-10:]

    def get_principle_recency(self, principle_id: str) -> float:
        """
        Get recency score for a principle (0.0 = just used, 1.0 = never used).
        Used for avoiding repetition.
        """
        if principle_id not in self.principle_timestamps:
            return 1.0

        last_used = self.principle_timestamps[principle_id]
        seconds_ago = (datetime.now() - last_used).total_seconds()

        # Decay over 5 minutes (300 seconds)
        return min(1.0, seconds_ago / 300)

    def get_recent_principles(self, n: int = 3) -> list[str]:
        """Get the N most recently used principles."""
        return self.principles_used[-n:] if self.principles_used else []

    def to_dict(self) -> dict:
        """Serialize for SSE transmission."""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "stage": self.current_stage.value,
            "stage_history": [s.value for s in self.stage_history],
            "recent_principles": self.get_recent_principles(),
            "customer_signals": self._format_customer_signals(),
            "price_sensitivity": self.customer.price_sensitivity,
            "urgency_level": self.customer.urgency_level,
        }

    def _format_customer_signals(self) -> list[str]:
        """Format customer signals for display."""
        signals = []

        if self.customer.price_sensitivity > 0.7:
            signals.append("💰 High price sensitivity")
        if self.customer.urgency_level > 0.7:
            signals.append("⏰ High urgency")
        if self.customer.mentioned_competitors:
            signals.append(f"🏢 Comparing: {', '.join(self.customer.mentioned_competitors)}")
        if self.customer.pain_points:
            signals.append(f"😓 Pain points: {', '.join(self.customer.pain_points[:2])}")
        if self.customer.objection_count >= 3:
            signals.append(f"⚠️ Multiple objections ({self.customer.objection_count})")

        return signals
```

### 2.2 Context-Aware Signal Extraction

**File**: `src/signal_extractor.py`

```python
"""
Signal Extractor - Extract customer signals from transcript.

Updates customer profile based on what they say.
"""

import re
from typing import Optional
from .context import CustomerProfile


# Signal patterns
PRICE_SIGNALS = [
    r"too expensive",
    r"can't afford",
    r"over.*budget",
    r"cheaper",
    r"discount",
    r"price.*high",
    r"cost.*much",
]

URGENCY_SIGNALS = [
    r"need.*today",
    r"right now",
    r"urgent",
    r"asap",
    r"immediately",
    r"can't wait",
    r"deadline",
]

COMPETITOR_PATTERNS = [
    r"amazon",
    r"flipkart",
    r"online",
    r"other store",
    r"competitor",
    r"elsewhere",
]

DECISION_AUTHORITY_SIGNALS = [
    (r"i.*decide", 0.8),
    (r"my decision", 0.8),
    (r"check with.*spouse|wife|husband|partner|family", 0.3),
    (r"ask.*boss|manager", 0.2),
    (r"need.*approval", 0.3),
]

PAIN_POINT_PATTERNS = [
    r"problem with",
    r"frustrated",
    r"doesn't work",
    r"broken",
    r"need.*solution",
    r"struggling with",
]

POSITIVE_SIGNALS = [
    r"sounds good",
    r"i like",
    r"interesting",
    r"tell me more",
    r"how do i",
    r"ready to",
]


def extract_signals(transcript: str, profile: CustomerProfile) -> CustomerProfile:
    """
    Extract signals from transcript and update customer profile.

    Args:
        transcript: Current turn's transcript
        profile: Existing customer profile

    Returns:
        Updated customer profile
    """
    text = transcript.lower()

    # Update price sensitivity
    price_matches = sum(1 for p in PRICE_SIGNALS if re.search(p, text))
    if price_matches > 0:
        profile.price_sensitivity = min(1.0, profile.price_sensitivity + 0.15 * price_matches)

    # Update urgency
    urgency_matches = sum(1 for p in URGENCY_SIGNALS if re.search(p, text))
    if urgency_matches > 0:
        profile.urgency_level = min(1.0, profile.urgency_level + 0.2 * urgency_matches)

    # Extract competitors
    for pattern in COMPETITOR_PATTERNS:
        if re.search(pattern, text):
            # Try to extract the competitor name
            match = re.search(rf"({pattern})", text)
            if match and match.group(1) not in profile.mentioned_competitors:
                profile.mentioned_competitors.append(match.group(1))

    # Update decision authority
    for pattern, authority_score in DECISION_AUTHORITY_SIGNALS:
        if re.search(pattern, text):
            profile.decision_authority = authority_score
            break

    # Track objections (price, competitor, delay signals)
    if any(re.search(p, text) for p in PRICE_SIGNALS + COMPETITOR_PATTERNS):
        profile.objection_count += 1

    # Track positive signals
    if any(re.search(p, text) for p in POSITIVE_SIGNALS):
        profile.positive_signals += 1

    # Track questions
    if "?" in transcript:
        profile.questions_asked += 1

    # Extract budget mentions
    budget_match = re.search(r"budget.*?(\d+[k,\d]*|\$[\d,]+)", text)
    if budget_match:
        profile.mentioned_budget = budget_match.group(1)

    # Extract timeline mentions
    timeline_patterns = [
        r"need.*by\s+(\w+)",
        r"deadline.*?(\w+\s+\d+|\d+\s+\w+)",
        r"within\s+(\d+\s+(?:days?|weeks?|months?))",
    ]
    for pattern in timeline_patterns:
        match = re.search(pattern, text)
        if match:
            profile.mentioned_timeline = match.group(1)
            break

    return profile
```

### 2.3 Context-Aware Situation Detection

**File**: `src/detector.py` (updated)

```python
def detect_situation_with_context(
    transcript: str,
    context: ConversationContext,
    pinecone_client: PineconeClient,
    embed_fn: callable,
    situations: dict
) -> DetectedSituation:
    """
    Detect situation using semantic search + conversation context.

    Context awareness:
    - Boosts situations matching current stage
    - Considers conversation trajectory
    - Detects repeated objections
    """
    from .context import SalesStage

    # Get semantic matches from Pinecone
    query_embedding = embed_fn(transcript)
    semantic_matches = pinecone_client.query_situations(
        query_embedding=query_embedding,
        top_k=5
    )

    # Score adjustments based on context
    scored_matches = []

    for match in semantic_matches:
        situation_id = match["id"]
        base_score = match["score"]

        situation_data = situations.get(situation_id, {})
        situation_stage = situation_data.get("typical_stage", "unknown")

        # Stage alignment bonus (+20% if matches current stage)
        stage_bonus = 0.2 if situation_stage == context.current_stage.value else 0.0

        # Repeated situation penalty (-10% if detected in last 2 turns)
        recent_situations = context.situations_detected[-2:]
        repetition_penalty = -0.1 if situation_id in recent_situations else 0.0

        # Trajectory bonus: boost closing signals if we're in late stage
        trajectory_bonus = 0.0
        if context.current_stage in [SalesStage.NEGOTIATION, SalesStage.CLOSING]:
            if "closing" in situation_id or "commitment" in situation_id:
                trajectory_bonus = 0.15

        final_score = base_score + stage_bonus + repetition_penalty + trajectory_bonus

        scored_matches.append({
            "id": situation_id,
            "score": final_score,
            "base_score": base_score,
            "adjustments": {
                "stage_bonus": stage_bonus,
                "repetition_penalty": repetition_penalty,
                "trajectory_bonus": trajectory_bonus
            },
            "data": situation_data
        })

    # Sort by final score
    scored_matches.sort(key=lambda x: x["score"], reverse=True)

    best_match = scored_matches[0]

    return DetectedSituation(
        situation_id=best_match["id"],
        confidence=best_match["score"],
        matched_signals=[],  # Could extract from semantic search
        applicable_principles=best_match["data"].get("applicable_principles", []),
        typical_stage=best_match["data"].get("typical_stage", "unknown"),
        score_breakdown=best_match["adjustments"]
    )
```

### 2.4 Context-Aware Principle Selection

**File**: `src/selector.py` (updated)

```python
def select_principle_with_context(
    situation: DetectedSituation,
    transcript: str,
    context: ConversationContext,
    pinecone_client: PineconeClient,
    embed_fn: callable,
    principles: dict
) -> SelectedPrinciple:
    """
    Select principle using semantic relevance + context factors.

    Scoring factors:
    - Semantic relevance to transcript (40%)
    - Recency penalty for recently used principles (30%)
    - Stage fit bonus (20%)
    - Customer profile fit (10%)
    """

    # Get candidate principles
    candidates = situation.applicable_principles

    # Get query embedding once for reuse
    query_embedding = embed_fn(transcript)

    if not candidates:
        # Fallback to semantic search
        semantic_results = pinecone_client.query_principles(
            query_embedding=query_embedding,
            top_k=5
        )
        candidates = [r["id"] for r in semantic_results]

    # Score each candidate
    scored_candidates = []

    for principle_id in candidates:
        if principle_id not in principles:
            continue

        principle_data = principles[principle_id]

        # Semantic relevance (query Pinecone for this principle's similarity)
        principle_results = pinecone_client.query_principles(
            query_embedding=query_embedding,
            filter={"principle_id": principle_id},
            top_k=1
        )
        semantic_score = principle_results[0]["score"] if principle_results else 0.5

        # Recency penalty
        recency = context.get_principle_recency(principle_id)
        recency_score = recency  # 0 = just used, 1 = not used recently

        # Stage fit
        principle_stages = principle_data.get("best_stages", [])
        stage_fit = 1.0 if context.current_stage.value in principle_stages else 0.5

        # Customer profile fit
        profile_fit = _compute_profile_fit(principle_data, context.customer)

        # Weighted combination
        final_score = (
            0.40 * semantic_score +
            0.30 * recency_score +
            0.20 * stage_fit +
            0.10 * profile_fit
        )

        scored_candidates.append({
            "id": principle_id,
            "score": final_score,
            "breakdown": {
                "semantic": semantic_score,
                "recency": recency_score,
                "stage_fit": stage_fit,
                "profile_fit": profile_fit
            },
            "data": principle_data
        })

    # Sort and select best
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    best = scored_candidates[0]
    p = best["data"]
    source = p.get("source", {})

    return SelectedPrinciple(
        principle_id=best["id"],
        name=p.get("name", ""),
        author=source.get("author", ""),
        book=source.get("book", ""),
        chapter=source.get("chapter", 0),
        page=source.get("page", ""),
        definition=p.get("definition", ""),
        intervention=p.get("intervention", ""),
        example_response=p.get("example_response", ""),
        mechanism=p.get("mechanism", ""),
        score=best["score"],
        score_breakdown=best["breakdown"]
    )


def _compute_profile_fit(principle_data: dict, profile: CustomerProfile) -> float:
    """
    Compute how well a principle fits the customer profile.
    """
    fit_score = 0.5  # Default neutral

    triggers = principle_data.get("triggers", [])

    # Price-related principles for price-sensitive customers
    if profile.price_sensitivity > 0.7:
        if any(t in ["price_resistant", "budget_conscious"] for t in triggers):
            fit_score += 0.3

    # Urgency principles for urgent customers
    if profile.urgency_level > 0.7:
        if any(t in ["time_pressure", "urgent_need"] for t in triggers):
            fit_score += 0.3

    # Authority principles when dealing with decision makers
    if profile.decision_authority > 0.7:
        if "authority" in principle_data.get("principle_id", "").lower():
            fit_score += 0.2

    return min(1.0, fit_score)
```

---

## Part 3: Updated Data Structures

### 3.1 Enhanced situations.json with Quick Tips

**This is the key optimization** — pre-written tips eliminate the need for model generation.

Add `quick_tip` and `quick_tip_variants` to each situation:

```json
{
  "price_shock_in_store": {
    "signals": ["that is expensive", "too costly", "over budget"],
    "contra_signals": ["worth it", "reasonable price"],
    "applicable_principles": ["kahneman_anchors_01", "kahneman_loss_aversion_01"],
    "typical_stage": "objection_handling",
    "priority": 5,

    "quick_tip": "Don't defend price. Acknowledge, then reframe total value.",

    "quick_tip_variants": {
      "high_price_sensitivity": "They're very price-focused. Lead with ROI and savings over time.",
      "high_urgency": "Urgency detected. Emphasize what they'll miss by waiting.",
      "repeat_objection": "Third time on price. Time for social proof — share a customer story."
    },

    "context_hints": {
      "increases_price_sensitivity": true,
      "indicates_objection": true,
      "common_next_situations": ["online_price_checking", "discount_request"]
    }
  },

  "online_price_checking": {
    "signals": ["amazon", "online", "cheaper elsewhere", "saw it for less"],
    "applicable_principles": ["kahneman_anchors_01", "cialdini_authority_01"],
    "typical_stage": "objection_handling",
    "priority": 4,

    "quick_tip": "Don't compete on price. Highlight what online doesn't include: warranty, support, instant availability.",

    "quick_tip_variants": {
      "repeat_objection": "They keep comparing online. Ask: 'What's most important — lowest price or peace of mind?'"
    }
  },

  "just_browsing": {
    "signals": ["just looking", "browsing", "not ready to buy"],
    "applicable_principles": ["cialdini_liking_01", "voss_mirroring_01"],
    "typical_stage": "discovery",
    "priority": 2,

    "quick_tip": "Build rapport, don't push. Ask open questions about what caught their eye.",

    "quick_tip_variants": {
      "high_urgency": "They say browsing but seem rushed. Gently probe: 'Is there a timeline I should know about?'"
    }
  },

  "discount_expectation": {
    "signals": ["any discount", "best price", "deal", "offer"],
    "applicable_principles": ["cialdini_scarcity_01", "cialdini_reciprocity_01"],
    "typical_stage": "negotiation",
    "priority": 4,

    "quick_tip": "Don't give discount first. Offer value-add instead: free accessory, extended warranty, priority support.",

    "quick_tip_variants": {
      "repeat_objection": "They're anchored on discount. Try: 'If I could include X for free, would that work?'"
    }
  }
}
```

### Quick Tip Guidelines

When writing quick tips:

1. **Actionable** — Start with a verb: "Don't...", "Ask...", "Highlight...", "Shift..."
2. **Brief** — 1-2 sentences max
3. **Contextual** — Variants for repeat objections, high urgency, etc.
4. **Principle-aligned** — Tips should reflect the underlying psychology
```

### 3.2 Enhanced principles.json

Add stage and profile fit indicators:

```json
{
  "principle_id": "kahneman_anchors_01",
  "name": "Anchoring",
  "source": { ... },
  "definition": "...",
  "intervention": "...",

  "best_stages": ["objection_handling", "negotiation"],
  "customer_fit": {
    "high_price_sensitivity": 0.9,
    "high_urgency": 0.5,
    "low_decision_authority": 0.3
  },
  "sequence_hints": {
    "works_well_after": ["social_proof", "liking"],
    "good_followed_by": ["loss_aversion", "scarcity"]
  }
}
```

---

## Part 4: Implementation Phases

### Phase 3a: Quick Win (~1.3s to tip)

**Goal**: Lookup-based tips, no extra model calls

| # | Task | File | Effort | Depends On |
|---|------|------|--------|------------|
| 1 | Add `quick_tip` to all situations | `situations.json` | Medium | — |
| 2 | Add `_get_quick_tip()` method | `src/server.py` | Low | #1 |
| 3 | Add `_get_action_verb()` method | `src/server.py` | Low | — |
| 4 | Add `process_turn_with_events()` method | `src/server.py` | Medium | #2, #3 |
| 5 | Create `realtime_panel.py` component | `streamlit_app/components/` | Medium | — |
| 6 | Update Streamlit for SSE consumption | `streamlit_app/app.py` | Medium | #4, #5 |

**Deliverable**: Coaching tip displayed in ~1.3s after audio upload.

### Phase 3b: Deep Context

**Goal**: Better detection via conversation tracking

| # | Task | File | Effort | Depends On |
|---|------|------|--------|------------|
| 1 | Add `SalesStage` enum | `src/context.py` | Low | — |
| 2 | Add `CustomerProfile` dataclass | `src/context.py` | Low | — |
| 3 | Enhance `ConversationContext` class | `src/context.py` | Medium | #1, #2 |
| 4 | Create signal extractor | `src/signal_extractor.py` | Medium | #2 |
| 5 | Add `detect_situation_with_context()` | `src/detector.py` | Medium | #3 |
| 6 | Add `select_principle_with_context()` | `src/selector.py` | Medium | #3, #5 |
| 7 | Add `best_stages`, `customer_fit` to principles | `principles.json` | Medium | — |
| 8 | Add context panel to Streamlit | `streamlit_app/app.py` | Low | #3 |
| 9 | Wire signal extraction into server | `src/server.py` | Low | #4, #6 |

**Deliverable**: Context-aware detection that avoids repetition and tracks customer profile.

### Phase 3c: Local Whisper (~0.5s to tip)

**Goal**: Sub-second tips via local ASR

| # | Task | File | Effort | Depends On |
|---|------|------|--------|------------|
| 1 | Add Whisper dependency | `pyproject.toml` | Low | — |
| 2 | Create `LocalASR` class | `streamlit_app/local_asr.py` | Low | #1 |
| 3 | Add `process_transcript()` method | `src/server.py` | Low | Phase 3a |
| 4 | Update Streamlit to use local ASR | `streamlit_app/app.py` | Medium | #2, #3 |

**Deliverable**: Coaching tip displayed in ~0.5s (local transcription + remote detection).

---

## Guide 2: Critical Fixes & Clarifications

This section addresses implementation issues and inconsistencies in the code examples above.

### 2.1 Data Structure Access (Critical)

**Issue**: Code examples use dictionary access `[]` on dataclass objects.

**Wrong** (lines 1172-1173, 1255):
```python
situation_id = match["id"]      # ❌ Wrong - match is a dataclass
base_score = match["score"]     # ❌ Wrong
candidates = [r["id"] for r in semantic_results]  # ❌ Wrong
```

**Correct**:
```python
situation_id = match.situation_id   # ✅ Dataclass attribute access
base_score = match.score            # ✅
candidates = [r.principle_id for r in semantic_results]  # ✅
```

**Pattern**: Pinecone query results should be converted to dataclasses first:

```python
@dataclass
class SituationMatch:
    situation_id: str
    score: float
    metadata: dict

def query_situations(query_embedding: list[float], top_k: int = 5) -> list[SituationMatch]:
    """Query Pinecone and return typed results."""
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return [
        SituationMatch(
            situation_id=m.id,
            score=m.score,
            metadata=m.metadata or {}
        )
        for m in results.matches
    ]
```

---

### 2.2 Streamlit Blocking Issue (Critical)

**Issue**: `time.sleep()` blocks Streamlit execution. Streamlit reruns the entire script on each interaction, so sleep-based "progressive display" won't work as intended.

**Wrong**:
```python
for event in events:
    if event_type == "transcript":
        transcript_box.markdown(...)
        time.sleep(0.2)  # ❌ Blocks entire script, no visual update
```

**Correct Approaches**:

**Option A: Display all at once (simplest)**
```python
# Just display everything immediately - no sleep needed
for event in events:
    if event["event"] == "transcript":
        transcript_box.markdown(f"> {event['data']['text']}")
    elif event["event"] == "coaching_tip":
        coaching_box.success(event['data']['tip'])
    # ... etc
```

**Option B: Use session state for staged display**
```python
# Track display stage in session state
if "display_stage" not in st.session_state:
    st.session_state.display_stage = 0
    st.session_state.events = []

# On new audio, fetch events and start display
if audio_input and not st.session_state.events:
    st.session_state.events = SalesCoach().process_turn_with_events.remote(...)
    st.session_state.display_stage = 0

# Display based on current stage
events = st.session_state.events
stage = st.session_state.display_stage

if stage >= 1 and len(events) > 0:
    transcript_box.markdown(events[0]["data"]["text"])
if stage >= 2 and len(events) > 1:
    situation_box.info(events[1]["data"]["name"])
# ... etc

# Button to advance (or auto-advance with st.rerun)
if st.button("Next") and stage < len(events):
    st.session_state.display_stage += 1
    st.rerun()
```

**Option C: Use streamlit-autorefresh for timed updates**
```python
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 500ms while displaying
if st.session_state.display_stage < len(st.session_state.events):
    st_autorefresh(interval=500, key="display_refresh")
```

---

### 2.3 Pinecone Client Pattern (Critical)

**Issue**: Document mixes `self.pinecone`, `self.pinecone_index`, and `PineconeClient` wrapper. Need consistency.

**Recommendation**: Use a `PineconeClient` wrapper class for clean abstraction.

**File**: `src/pinecone_client.py`

```python
from dataclasses import dataclass
from pinecone import Pinecone

@dataclass
class SituationMatch:
    situation_id: str
    score: float
    metadata: dict

@dataclass
class PrincipleMatch:
    principle_id: str
    score: float
    metadata: dict


class PineconeClient:
    """Wrapper for Pinecone operations with typed returns."""

    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def query_situations(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> list[SituationMatch]:
        """Query situations index and return typed results."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"type": "situation"}
        )
        return [
            SituationMatch(
                situation_id=m.id,
                score=m.score,
                metadata=m.metadata or {}
            )
            for m in results.matches
            if m.score >= min_score
        ]

    def query_principles(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict = None
    ) -> list[PrincipleMatch]:
        """Query principles index and return typed results."""
        query_filter = {"type": "principle"}
        if filter:
            query_filter.update(filter)

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=query_filter
        )
        return [
            PrincipleMatch(
                principle_id=m.id,
                score=m.score,
                metadata=m.metadata or {}
            )
            for m in results.matches
        ]
```

**Usage in server.py**:
```python
from .pinecone_client import PineconeClient

@modal.enter()
def load_model(self):
    # ... model loading ...
    self.pinecone = PineconeClient(
        api_key=os.environ["PINECONE_API_KEY"],
        index_name="sales-coach"
    )
```

---

### 2.4 Error Handling & Fallbacks (Critical)

**Issue**: No error handling for Pinecone failures, empty results, or transcription errors.

**Add to `process_turn_with_events()`**:

```python
@modal.method()
def process_turn_with_events(self, session_id: str, turn_number: int, context_dict: dict) -> list[dict]:
    events = []
    context = ConversationContext.from_dict(context_dict)

    # === STEP 1: Transcribe with error handling ===
    try:
        sessions_volume.reload()
        audio_path = f"/sessions/{session_id}/question.wav"
        wav, sr = torchaudio.load(audio_path)
        transcript = self._transcribe(wav, sr)
    except FileNotFoundError:
        events.append({"event": "error", "data": {"message": "Audio file not found"}})
        return events
    except Exception as e:
        events.append({"event": "error", "data": {"message": f"Transcription failed: {str(e)}"}})
        return events

    events.append({"event": "transcript", "data": {"text": transcript}})

    # === STEP 2: Detect situation with fallback ===
    try:
        detection_result = detect_situation_semantic(
            transcript=transcript,
            pinecone_client=self.pinecone,
            embed_fn=self.embed_query,
            situations=self.situations
        )
        situation = detection_result.best_match
    except Exception as e:
        # Fallback to keyword-based detection
        situation = self._fallback_keyword_detection(transcript)
        events.append({
            "event": "warning",
            "data": {"message": f"Using fallback detection: {str(e)}"}
        })

    # Handle empty results
    if situation is None:
        situation = DetectedSituation(
            situation_id="general_inquiry",
            confidence_score=0.3,
            matched_signal="[no match]",
            applicable_principles=["cialdini_liking_01"],  # Safe default
            typical_stage="discovery"
        )
        events.append({
            "event": "warning",
            "data": {"message": "No situation matched, using default"}
        })

    events.append({"event": "situation", "data": {...}})

    # ... rest of processing ...

    return events


def _fallback_keyword_detection(self, transcript: str) -> DetectedSituation:
    """Simple keyword-based fallback when Pinecone is unavailable."""
    text_lower = transcript.lower()

    # Simple keyword matching
    if any(kw in text_lower for kw in ["expensive", "price", "cost", "budget"]):
        return DetectedSituation(
            situation_id="price_shock_in_store",
            confidence_score=0.6,
            matched_signal="price keyword",
            applicable_principles=["kahneman_anchors_01"],
            typical_stage="objection_handling"
        )
    elif any(kw in text_lower for kw in ["just looking", "browsing"]):
        return DetectedSituation(
            situation_id="just_browsing",
            confidence_score=0.6,
            matched_signal="browsing keyword",
            applicable_principles=["cialdini_liking_01"],
            typical_stage="discovery"
        )

    # Default fallback
    return DetectedSituation(
        situation_id="general_inquiry",
        confidence_score=0.3,
        matched_signal="[default]",
        applicable_principles=["cialdini_liking_01"],
        typical_stage="discovery"
    )
```

---

### 2.5 Signal Extraction Integration (Medium)

**Issue**: `signal_extractor.py` is defined but not wired into the server.

**Add to `process_turn_with_events()`** after transcription:

```python
from .signal_extractor import extract_signals

@modal.method()
def process_turn_with_events(self, session_id: str, turn_number: int, context_dict: dict) -> list[dict]:
    events = []
    context = ConversationContext.from_dict(context_dict)

    # ... transcription step ...

    # === Extract customer signals from transcript ===
    # Update customer profile based on what they said
    context.customer = extract_signals(transcript, context.customer)

    # Emit signal update event for UI
    if context.customer.price_sensitivity > 0.7 or context.customer.urgency_level > 0.7:
        events.append({
            "event": "customer_signal",
            "data": {
                "price_sensitivity": context.customer.price_sensitivity,
                "urgency_level": context.customer.urgency_level,
                "objection_count": context.customer.objection_count
            }
        })

    # ... continue with detection ...
```

---

### 2.6 Context Method Signature Mismatch (Critical)

**Issue**: `context.add_turn()` is called with 6 parameters, but Phase 3b defines it with 4.

**Solution**: Use a unified signature that supports both phases.

**Updated `add_turn()` in `src/context.py`**:

```python
def add_turn(
    self,
    transcript: str,
    detected_situation: str,
    selected_principle: str,
    # Optional parameters for backward compatibility
    situation_score: float = 0.0,
    principle_score: float = 0.0,
    response_text: str = "",
    detected_stage: str = None
):
    """Record a conversation turn.

    Args:
        transcript: What the customer said
        detected_situation: Situation ID that was detected
        selected_principle: Principle ID that was selected
        situation_score: Confidence score for situation (optional)
        principle_score: Score for principle selection (optional)
        response_text: Generated response text (optional)
        detected_stage: Explicit stage override (optional, auto-detected if None)
    """
    self.turn_count += 1

    # Update principle tracking
    self.principles_used.append(selected_principle)
    self.principle_timestamps[selected_principle] = datetime.now()

    # Update situation tracking
    self.situations_detected.append(detected_situation)

    # Update stage (auto-detect from situation if not provided)
    if detected_stage:
        new_stage = SalesStage(detected_stage)
    else:
        new_stage = self._infer_stage_from_situation(detected_situation)

    if new_stage != self.current_stage:
        self.stage_history.append(self.current_stage)
        self.current_stage = new_stage

    # Store transcript with all metadata
    self.transcript_history.append({
        "turn": self.turn_count,
        "transcript": transcript,
        "situation": detected_situation,
        "situation_score": situation_score,
        "principle": selected_principle,
        "principle_score": principle_score,
        "response_text": response_text,
        "timestamp": datetime.now().isoformat()
    })

    # Keep only last 10 turns
    if len(self.transcript_history) > 10:
        self.transcript_history = self.transcript_history[-10:]


def _infer_stage_from_situation(self, situation_id: str) -> SalesStage:
    """Infer sales stage from situation ID."""
    stage_mapping = {
        "just_browsing": SalesStage.DISCOVERY,
        "product_inquiry": SalesStage.DISCOVERY,
        "price_shock": SalesStage.OBJECTION_HANDLING,
        "online_price_checking": SalesStage.OBJECTION_HANDLING,
        "discount_expectation": SalesStage.NEGOTIATION,
        "closing_signal": SalesStage.CLOSING,
    }
    for key, stage in stage_mapping.items():
        if key in situation_id:
            return stage
    return self.current_stage  # Keep current if unknown
```

---

### 2.7 DetectedSituation Field Names (Medium)

**Issue**: Code uses `confidence` but dataclass has `confidence_score`. Uses `matched_signals` (plural) but should be `matched_signal`.

**Correct DetectedSituation dataclass** (`src/detector.py`):

```python
@dataclass
class DetectedSituation:
    situation_id: str
    confidence_score: float          # ✅ Not "confidence"
    matched_signal: str              # ✅ Singular, not "matched_signals"
    applicable_principles: list[str]
    typical_stage: str
    score_breakdown: dict = field(default_factory=dict)  # Optional for debugging
```

**Correct usage**:
```python
return DetectedSituation(
    situation_id=best_match.situation_id,
    confidence_score=best_match.score,         # ✅ confidence_score
    matched_signal=best_match.metadata.get("matched_signal", ""),  # ✅ singular
    applicable_principles=situation_data.get("applicable_principles", []),
    typical_stage=situation_data.get("typical_stage", "unknown"),
    score_breakdown=adjustments
)
```

---

### 2.8 Principle Scoring Integration (Medium)

**Issue**: `select_principle_with_context()` doesn't use existing `principle_scorer.py`.

**Integration approach**: Use existing scorer as a component.

```python
# In src/selector.py
from .principle_scorer import score_principle  # If exists

def select_principle_with_context(
    situation: DetectedSituation,
    transcript: str,
    context: ConversationContext,
    pinecone_client: PineconeClient,
    embed_fn: callable,
    principles: dict
) -> SelectedPrinciple:
    """Select principle using semantic + context + existing scorer."""

    candidates = situation.applicable_principles
    query_embedding = embed_fn(transcript)

    if not candidates:
        # Fallback to semantic search
        results = pinecone_client.query_principles(query_embedding, top_k=5)
        candidates = [r.principle_id for r in results]

    scored_candidates = []

    for principle_id in candidates:
        if principle_id not in principles:
            continue

        principle_data = principles[principle_id]

        # Use existing scorer if available
        try:
            base_score = score_principle(
                principle_id=principle_id,
                transcript=transcript,
                situation_id=situation.situation_id
            )
        except NameError:
            # Fallback if scorer not available
            base_score = 0.5

        # Add context adjustments
        recency = context.get_principle_recency(principle_id)
        stage_fit = 1.0 if context.current_stage.value in principle_data.get("best_stages", []) else 0.5

        final_score = 0.5 * base_score + 0.3 * recency + 0.2 * stage_fit

        scored_candidates.append({
            "id": principle_id,
            "score": final_score,
            "data": principle_data
        })

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    # ... rest of function
```

---

### 2.9 Summary: Required Changes by File

| File | Changes Required | Priority |
|------|------------------|----------|
| `src/pinecone_client.py` | **Create new file** with typed wrapper | High |
| `src/detector.py` | Fix dataclass access, field names | High |
| `src/selector.py` | Fix dataclass access, integrate scorer | High |
| `src/context.py` | Unify `add_turn()` signature | High |
| `src/server.py` | Add error handling, signal extraction | High |
| `streamlit_app/app.py` | Remove `time.sleep()`, use state-based display | High |

---

## Recommended Implementation Order

```
Phase 3a (Quick Win)
├── 1. situations.json — Add quick_tips to all ~50 situations
├── 2. src/server.py — Add _get_quick_tip(), _get_action_verb()
├── 3. src/server.py — Add process_turn_with_events() method
├── 4. streamlit_app/components/realtime_panel.py — Create component
└── 5. streamlit_app/app.py — Wire up SSE consumer

    ↓ Test: Verify ~1.3s to coaching tip

Phase 3b (Deep Context)
├── 1. src/context.py — Enhance with SalesStage, CustomerProfile
├── 2. src/signal_extractor.py — Create signal extraction
├── 3. src/detector.py — Add context-aware detection
├── 4. src/selector.py — Add context-aware selection
├── 5. principles.json — Add best_stages, customer_fit
└── 6. src/server.py — Wire context updates

    ↓ Test: Verify no principle repetition, stage tracking works

Phase 3c (Local Whisper)
├── 1. pyproject.toml — Add faster-whisper
├── 2. streamlit_app/local_asr.py — Create LocalASR class
├── 3. src/server.py — Add process_transcript() for text input
└── 4. streamlit_app/app.py — Switch to local ASR flow

    ↓ Test: Verify ~0.5s to coaching tip
```

---

## Part 5: Files Summary

### New Files

| File | Purpose | Phase |
|------|---------|-------|
| `src/context.py` | ConversationContext and CustomerProfile classes | 3b |
| `src/signal_extractor.py` | Extract customer signals from transcript | 3b |
| `streamlit_app/local_asr.py` | Local Whisper transcription | 3c |

### Modified Files

| File | Changes | Phase |
|------|---------|-------|
| `situations.json` | Add `quick_tip` and `quick_tip_variants` | 3a |
| `src/server.py` | Add streaming methods, `_get_quick_tip()` | 3a |
| `streamlit_app/app.py` | SSE consumer, real-time UI | 3a |
| `src/detector.py` | Add `detect_situation_with_context()` | 3b |
| `src/selector.py` | Add `select_principle_with_context()` | 3b |
| `principles.json` | Add `best_stages` and `customer_fit` | 3b |
| `pyproject.toml` | Add `openai-whisper` or `faster-whisper` | 3c |

---

## Part 5: Setup & Running

### Deploy Updated Server

```bash
modal deploy src/server.py
```

### Run Streamlit App

```bash
streamlit run streamlit_app/app.py
```

### Expected Timeline

| Event | Time | What User Sees |
|-------|------|----------------|
| Audio uploaded | 0.0s | "Processing..." |
| Transcript | ~1.0s | Customer's words displayed |
| Situation | ~1.3s | "Price objection detected" |
| Coaching tip | ~2.0s | **Highlighted tip box with action** |
| Voice ready | ~5.0s | Play button appears |

---

## Success Criteria

| Metric | Target (Lookup) | Target (Local Whisper) |
|--------|-----------------|------------------------|
| Time to coaching tip | < 1.5 seconds | < 0.6 seconds |
| Situation detection accuracy | > 80% | > 80% |
| Principle relevance (user rating) | > 4/5 | > 4/5 |
| No principle repetition in 3 turns | 100% | 100% |
| Stage tracking accuracy | > 75% | > 75% |

---

## What Phase 3 Does NOT Include

- ❌ CRM integration
- ❌ External LLM calls (Haiku, etc.) — uses same LFM2.5 model
- ❌ Persistent storage across sessions
- ❌ Multi-user authentication
- ❌ Mobile app

---

## Future Optimization: Local Whisper ASR (~0.5s total)

For sub-second coaching tips, run ASR locally instead of on Modal:

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LOCAL (Streamlit)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Audio ──→ Whisper Tiny/Base (local) ──→ Transcript                        │
│                    ~0.3s                       │                             │
│                                                │                             │
│                                                ▼                             │
│                                    Send transcript (not audio)              │
│                                                │                             │
└────────────────────────────────────────────────┼────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODAL SERVER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Transcript ──→ Detect (Pinecone) ──→ Lookup quick_tip ──→ SSE            │
│                      ~0.2s                   ~0ms                            │
│                                                                              │
│   (Voice generation still uses audio, runs in background)                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Timeline with Local Whisper

| Event | Time |
|-------|------|
| Audio recorded | 0.0s |
| Local Whisper transcription | ~0.3s |
| Send transcript to Modal | ~0.1s |
| Detect + Lookup tip | ~0.1s |
| **Coaching tip displayed** | **~0.5s** ✓ |

### Implementation Sketch

```python
# streamlit_app/local_asr.py
import whisper

class LocalASR:
    def __init__(self, model_size: str = "tiny"):
        """
        Load Whisper locally for fast transcription.

        Model sizes:
        - tiny: 39M params, ~0.3s, good enough for most cases
        - base: 74M params, ~0.5s, better accuracy
        """
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file locally."""
        result = self.model.transcribe(audio_path, language="en")
        return result["text"]


# In Streamlit app
local_asr = LocalASR(model_size="tiny")

if audio_input:
    # Save audio to temp file
    temp_path = save_temp_audio(audio_input)

    # Transcribe locally (fast!)
    transcript = local_asr.transcribe(temp_path)
    st.write(f"You said: {transcript}")

    # Send transcript (not audio) to Modal for detection
    result = SalesCoach().process_transcript.remote(
        transcript=transcript,
        session_id=session_id,
        context=context
    )
```

### New Server Method for Transcript Input

```python
@modal.method()
def process_transcript(
    self,
    transcript: str,
    session_id: str,
    context: dict
) -> Generator[dict, None, None]:
    """
    Process pre-transcribed text (no ASR needed).
    Used when client does local Whisper transcription.
    """
    yield {"event": "transcript", "data": {"text": transcript}}

    # Detect situation
    situation = detect_situation_with_context(transcript, context, ...)
    yield {"event": "situation", "data": {...}}

    # Select principle
    principle = select_principle_with_context(situation, transcript, context, ...)
    yield {"event": "principle", "data": {...}}

    # Lookup quick tip (instant)
    quick_tip = self._get_quick_tip(situation, principle, context)
    yield {"event": "coaching_tip", "data": {"tip": quick_tip}}

    # Voice generation still needs audio - client uploads separately
    # Or skip voice entirely in "speed mode"
```

### Dependencies for Local Whisper

Add to `pyproject.toml`:

```toml
# Optional - for local ASR
"openai-whisper>=20231117",
```

Or use `faster-whisper` for even better performance:

```toml
"faster-whisper>=0.10.0",
```

---

## Future Considerations (Phase 4+)

- Voice tone analysis (detect frustration, excitement)
- Streaming audio playback (play while generating)
- A/B testing different principles
- Outcome tracking (did the tip help close?)
- Team analytics dashboard

---

*Document created: 2025-01-19*
*For: Phase 3 Implementation - Real-time Coaching & Deep Context*
