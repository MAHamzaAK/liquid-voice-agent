"""
Streamlit App - Sales Coach Web Interface

Phase II: Voice Chat Interface with auto-upload, auto-play, and continuous conversation.
"""

import os
import sys
import tempfile
import time
import base64
from pathlib import Path
from datetime import datetime

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()


# Page config
st.set_page_config(
    page_title="Behavioral Sales Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "context" not in st.session_state:
        from context import ConversationContext
        st.session_state.context = ConversationContext(
            session_id=st.session_state.session_id
        )
    if "debug_info" not in st.session_state:
        st.session_state.debug_info = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "waiting_for_response" not in st.session_state:
        st.session_state.waiting_for_response = False
    if "last_processed_audio" not in st.session_state:
        st.session_state.last_processed_audio = None
    if "auto_play_enabled" not in st.session_state:
        st.session_state.auto_play_enabled = True


def new_session():
    """Start a new conversation session."""
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.conversation_history = []
    from context import ConversationContext
    st.session_state.context = ConversationContext(
        session_id=st.session_state.session_id
    )
    st.session_state.debug_info = None


def process_audio(audio_bytes: bytes) -> dict:
    """
    Process audio through the Modal backend.

    Args:
        audio_bytes: Raw audio bytes from recording or file upload

    Returns:
        Dictionary with transcript, response, coaching info, and debug data
    """
    import modal

    # Save audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        # Import constants from src (modal_app.py defines these)
        from modal_app import APP_NAME, SESSIONS_VOLUME

        session_id = st.session_state.session_id
        turn_number = len(st.session_state.conversation_history) + 1

        # Get the Modal volume
        volume = modal.Volume.from_name(SESSIONS_VOLUME, create_if_missing=True)

        # Upload audio using batch_upload API
        # Path is relative to volume root (maps to /sessions/{session_id} in container)
        remote_path = f"{session_id}/question.wav"
        with volume.batch_upload(force=True) as batch:
            batch.put_file(temp_path, remote_path)

        # Prepare context data for semantic selection
        context_data = {
            "recent_principles": st.session_state.context.recent_principles,
            "current_stage": st.session_state.context.current_stage,
        }

        # Process on Modal with context
        SalesCoach = modal.Cls.from_name(APP_NAME, "SalesCoach")
        result = SalesCoach().process_turn.remote(
            session_id=session_id,
            turn_number=turn_number,
            context_data=context_data
        )

        # Download response audio if available
        response_audio = None
        if result.get("audio_path"):
            try:
                response_path = tempfile.mktemp(suffix=".wav")
                with open(response_path, "wb") as f:
                    for chunk in volume.read_file(f"{session_id}/answer.wav"):
                        f.write(chunk)
                with open(response_path, "rb") as f:
                    response_audio = f.read()
                os.unlink(response_path)
            except Exception as e:
                st.warning(f"Could not download response audio: {e}")

        result["response_audio"] = response_audio
        return result

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def render_sidebar():
    """Render the sidebar with session info and controls."""
    with st.sidebar:
        st.title("🎯 Sales Coach")
        st.caption("Behavioral Psychology Coaching")

        st.divider()

        # Session info
        st.subheader("Session")
        st.text(f"ID: {st.session_state.session_id}")
        st.text(f"Turns: {len(st.session_state.conversation_history)}")
        if st.session_state.context:
            st.text(f"Stage: {st.session_state.context.current_stage}")

        if st.button("🔄 New Session", use_container_width=True):
            new_session()
            st.rerun()

        st.divider()

        # Quick tips
        st.subheader("How it works")
        st.markdown("""
        1. **Click the mic** and speak as a customer
        2. Recording **stops automatically** after 2s of silence
        3. AI responds with coaching (audio auto-plays)
        4. **Grounding shown automatically** - principle, situation, and reasoning
        """)
        
        st.divider()
        
        # Optional: Advanced debug toggle (collapsed by default)
        with st.expander("⚙️ Advanced Settings", expanded=False):
            show_debug = st.toggle("Show Technical Debug Panel", value=False)
            st.session_state.show_debug = show_debug


def render_debug_panel(debug_info: dict):
    """Render the debug panel with detection scores and methods."""
    if not debug_info:
        st.info("Process audio to see detection details")
        return

    # Methods used
    detection_method = debug_info.get("detection_method", "unknown")
    selection_method = debug_info.get("selection_method", "unknown")

    col1, col2 = st.columns(2)
    with col1:
        method_color = "🟢" if detection_method == "semantic" else "🟡"
        st.markdown(f"**Detection:** {method_color} {detection_method}")
    with col2:
        method_color = "🟢" if selection_method == "semantic" else "🟡"
        st.markdown(f"**Selection:** {method_color} {selection_method}")
    
    # Performance timing
    timing = debug_info.get("timing", {})
    if timing:
        st.divider()
        st.markdown("##### ⏱️ Performance Timing")
        
        total_time = timing.get("total_time", 0)
        if total_time:
            st.metric("Total Time", f"{total_time:.2f}s")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if timing.get("transcription_time"):
                st.metric("Transcription", f"{timing['transcription_time']:.2f}s")
            if timing.get("generation_time"):
                st.metric("Generation", f"{timing['generation_time']:.2f}s", delta=f"{timing['generation_time']/total_time*100:.0f}% of total" if total_time else None)
        
        with col2:
            if timing.get("detection_time"):
                st.metric("Detection", f"{timing['detection_time']:.2f}s")
            if timing.get("decode_time"):
                st.metric("Decode/Save", f"{timing['decode_time']:.2f}s")
        
        with col3:
            if timing.get("selection_time"):
                st.metric("Selection", f"{timing['selection_time']:.2f}s")
        
        # Highlight bottleneck
        if timing.get("generation_time") and total_time:
            gen_pct = (timing['generation_time'] / total_time) * 100
            if gen_pct > 50:
                st.warning(f"⚠️ Generation is {gen_pct:.0f}% of total time - this is the bottleneck!")

    st.divider()

    # Situation detection
    st.markdown("##### Situation Detection")
    situation_candidates = debug_info.get("situation_candidates", [])
    if situation_candidates:
        for sit in situation_candidates[:5]:
            score = sit.get("score", 0)
            situation_id = sit.get("situation_id", "unknown")
            signal = sit.get("signal", "")
            st.progress(
                min(max(score, 0), 1.0),
                text=f"{situation_id}: {score:.2f}"
            )
            if signal:
                st.caption(f"  Signal: \"{signal[:50]}...\"" if len(signal) > 50 else f"  Signal: \"{signal}\"")
    else:
        st.text(f"Detected: {debug_info.get('detected_situation', 'N/A')}")
        st.text(f"Signal: {debug_info.get('matched_signal', 'N/A')}")
        st.text(f"Score: {debug_info.get('situation_score', 'N/A')}")

    st.divider()

    # Principle selection
    st.markdown("##### Principle Selection")
    principle_candidates = debug_info.get("principle_candidates", [])
    if principle_candidates:
        for prin in principle_candidates[:5]:
            score = prin.get("final_score", 0)
            name = prin.get("name", "Unknown")
            st.progress(
                min(max(score, 0), 1.0),
                text=f"{name}: {score:.3f}"
            )

            # Show breakdown in expander
            breakdown = prin.get("breakdown", {})
            if breakdown:
                with st.expander(f"Score breakdown for {name}"):
                    for factor, data in breakdown.items():
                        raw = data.get("raw", 0)
                        weight = data.get("weight", 0)
                        weighted = data.get("weighted", 0)
                        st.text(f"  {factor}: {raw:.3f} × {weight} = {weighted:.3f}")
    else:
        st.text(f"Selected: {debug_info.get('selected_principle', 'N/A')}")
        st.text(f"Score: {debug_info.get('selection_score', 'N/A')}")

    st.divider()

    # Context info
    st.markdown("##### Context Used")
    context_used = debug_info.get("context_used", {})
    if context_used:
        st.text(f"Stage: {context_used.get('current_stage', 'N/A')}")
        recent = context_used.get("recent_principles", [])
        st.text(f"Recent principles: {', '.join(recent) if recent else 'None'}")
    else:
        ctx = st.session_state.context
        st.text(f"Stage: {ctx.current_stage}")
        st.text(f"Recent principles: {', '.join(ctx.recent_principles) or 'None'}")


def render_conversation():
    """Render the conversation history."""
    for i, turn in enumerate(st.session_state.conversation_history):
        # Customer message (you)
        with st.chat_message("user"):
            st.markdown(f"**You (Customer):** {turn.get('transcript', 'N/A')}")

        # Salesperson response (AI)
        with st.chat_message("assistant"):
            # Get coaching info
            coaching = turn.get("coaching", {})
            rec = coaching.get("recommendation", {}) if coaching else {}
            
            # Show grounding inline - immediately visible at top
            if rec.get("principle"):
                principle_name = rec.get("principle", "Unknown")
                principle_source = rec.get("source", "")
                situation = coaching.get("detected_situation", "general_inquiry")
                
                # Compact grounding header
                st.markdown(f"**🧠 Principle:** {principle_name} | **📋 Situation:** {situation.replace('_', ' ').title()}")
                st.caption(f"📚 {principle_source}")
                st.divider()
            
            # Salesperson response text
            st.markdown(turn.get('text_response', 'N/A'))

            # Audio playback with auto-play (single player)
            if turn.get("response_audio"):
                audio_bytes = turn["response_audio"]
                if st.session_state.auto_play_enabled:
                    # Use Streamlit's audio player with autoplay
                    st.audio(audio_bytes, format="audio/wav", autoplay=True)
                else:
                    # Show without autoplay
                    st.audio(audio_bytes, format="audio/wav", autoplay=False)

            # Why it works - shown inline below response
            if rec.get("why_it_works"):
                st.divider()
                st.markdown(f"**💡 Why this works:** {rec.get('why_it_works', '')}")
            
            # Optional: Detailed breakdown in expander for those who want more
            if coaching:
                with st.expander("📊 Detailed breakdown", expanded=False):
                    st.markdown(f"**Full Situation:** {coaching.get('detected_situation', 'N/A')}")
                    st.markdown(f"**Matched Signal:** {coaching.get('matched_signal', 'N/A')}")
                    st.markdown(f"**Stage:** {coaching.get('stage', 'N/A')}")
                    if rec.get("source"):
                        st.markdown(f"**Full Source:** {rec.get('source', 'N/A')}")


def render_input_area():
    """Render the continuous voice chat input area."""
    
    st.markdown("### 🎤 Voice Chat")
    
    # Show status
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.session_state.waiting_for_response:
            st.info("🔄 Processing your message... Please wait for the response.")
        elif st.session_state.processing:
            st.info("🎙️ Recording... (stops after 2s of silence)")
        else:
            st.caption("💬 Click mic → Speak → Click stop → Auto-uploads immediately")
    
    with col2:
        # Auto-play toggle
        auto_play = st.toggle("🔊 Auto-play", value=st.session_state.auto_play_enabled, key="auto_play_toggle")
        st.session_state.auto_play_enabled = auto_play
    
    # Audio input - auto-processes when recording stops
    # Note: Streamlit's audio_input requires clicking stop, but we auto-upload immediately when available
    turn_count = len(st.session_state.conversation_history)
    
    # Reset waiting flag if we have a new turn (response completed)
    if "last_turn_count" not in st.session_state:
        st.session_state.last_turn_count = turn_count
    if st.session_state.last_turn_count < turn_count:
        st.session_state.waiting_for_response = False
        st.session_state.last_turn_count = turn_count
        # Reset the audio hash to allow new recording
        st.session_state.last_processed_audio = None
    
    # Use a stable key that resets after each turn to allow continuous recording
    recording_key = f"audio_input_{turn_count}"
    
    # Only show audio input when not waiting for response
    if not st.session_state.waiting_for_response:
        audio_value = st.audio_input(
            "🎤 Click mic, speak, then click stop - auto-uploads immediately",
            key=recording_key,
        )

        # Auto-upload immediately when audio becomes available (after user clicks stop)
        if audio_value is not None:
            audio_bytes = audio_value.getvalue()
            audio_hash = hash(audio_bytes)

            # Only process if this is new audio (not already processed)
            if st.session_state.last_processed_audio != audio_hash and len(audio_bytes) > 0:
                st.session_state.last_processed_audio = audio_hash
                st.session_state.waiting_for_response = True
                # Process immediately
                process_and_display(audio_bytes)
    else:
        # Show placeholder while processing
        st.caption("🔄 Processing your message... Please wait")
        
        # Add helper message
        st.info("✨ When response is ready, you can record again immediately")


def process_and_display(audio_bytes: bytes):
    """Process audio and update display - auto-triggers and auto-plays response."""
    # Use a placeholder to show processing status
    status_placeholder = st.empty()
    
    with status_placeholder.container():
        with st.spinner("🎤 Processing your message..."):
            try:
                result = process_audio(audio_bytes)

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                    st.session_state.waiting_for_response = False
                    status_placeholder.empty()
                    return

                # Add to conversation history
                st.session_state.conversation_history.append(result)

                # Update context with actual scores from Modal
                coaching = result.get("coaching", {})
                rec = coaching.get("recommendation", {})
                debug = result.get("debug", {})

                st.session_state.context.add_turn(
                    transcript=result.get("transcript", ""),
                    detected_situation=coaching.get("detected_situation", "general_inquiry"),
                    situation_score=coaching.get("situation_score", 1.0),
                    selected_principle=rec.get("principle_id", rec.get("principle", "")),
                    principle_score=rec.get("selection_score", 1.0),
                    response_text=result.get("text_response", "")
                )

                # Update debug info with full data from Modal
                st.session_state.debug_info = {
                    "detected_situation": coaching.get("detected_situation"),
                    "matched_signal": coaching.get("matched_signal"),
                    "selected_principle": rec.get("principle"),
                    "situation_score": coaching.get("situation_score", 0),
                    "selection_score": rec.get("selection_score", 0),
                    # Phase II: Full debug data
                    "detection_method": debug.get("detection_method", "unknown"),
                    "selection_method": debug.get("selection_method", "unknown"),
                    "situation_candidates": debug.get("situation_candidates", []),
                    "principle_candidates": debug.get("principle_candidates", []),
                    "context_used": debug.get("context_used", {}),
                    # Performance timing
                    "timing": debug.get("timing", {}),
                }
                
                # Clear waiting flag and status
                st.session_state.waiting_for_response = False
                status_placeholder.empty()
                
                # Auto-scroll to bottom and trigger rerun
                st.rerun()

            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state.waiting_for_response = False
                status_placeholder.empty()


def main():
    """Main app entry point."""
    init_session_state()
    render_sidebar()

    # Main content area - full width by default, grounding is inline
    st.title("Voice Chat")
    render_conversation()
    st.divider()
    render_input_area()
    
    # Optional debug panel - only shown if explicitly enabled
    if st.session_state.get("show_debug", False):
        st.divider()
        st.title("Technical Debug")
        render_debug_panel(st.session_state.debug_info)
    
    # Auto-scroll to bottom when new message arrives
    if st.session_state.conversation_history:
        scroll_script = """
        <script>
            // Scroll to bottom of chat when page loads
            window.addEventListener('load', function() {
                window.scrollTo(0, document.body.scrollHeight);
            });
        </script>
        """
        st.markdown(scroll_script, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
