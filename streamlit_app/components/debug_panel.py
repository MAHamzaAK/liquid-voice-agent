"""
Debug Panel Component - Visualization of detection scores

Shows situation detection scores, principle selection scores, and context state.
"""

import streamlit as st
from typing import Optional


def render_situation_scores(candidates: list[dict]):
    """
    Render situation detection scores as progress bars.

    Args:
        candidates: List of situation candidates with scores
    """
    st.markdown("##### Situation Candidates")

    if not candidates:
        st.info("No situation candidates yet")
        return

    for sit in candidates[:5]:
        situation_id = sit.get("situation_id", "Unknown")
        score = sit.get("score", sit.get("confidence_score", 0))
        signal = sit.get("signal_text", sit.get("matched_signal", ""))

        # Color based on score
        if score >= 0.8:
            color = "green"
        elif score >= 0.6:
            color = "orange"
        else:
            color = "red"

        st.progress(
            min(float(score), 1.0),
            text=f"**{situation_id}** ({score:.2f})"
        )

        if signal:
            st.caption(f'Signal: "{signal[:50]}..."' if len(signal) > 50 else f'Signal: "{signal}"')


def render_principle_scores(candidates: list[dict]):
    """
    Render principle selection scores with breakdown.

    Args:
        candidates: List of principle candidates with score breakdowns
    """
    st.markdown("##### Principle Candidates")

    if not candidates:
        st.info("No principle candidates yet")
        return

    for i, prin in enumerate(candidates[:5]):
        name = prin.get("name", "Unknown")
        final_score = prin.get("final_score", 0)
        breakdown = prin.get("breakdown", {})

        # Main score bar
        st.progress(
            min(float(final_score), 1.0),
            text=f"**{name}** ({final_score:.3f})"
        )

        # Breakdown in expander
        with st.expander(f"Score breakdown for {name}", expanded=(i == 0)):
            cols = st.columns(4)

            with cols[0]:
                semantic = breakdown.get("semantic", {})
                st.metric(
                    "Semantic",
                    f"{semantic.get('weighted', 0):.3f}",
                    help="Similarity to transcript (weight: 40%)"
                )

            with cols[1]:
                recency = breakdown.get("recency", {})
                st.metric(
                    "Recency",
                    f"{recency.get('weighted', 0):.3f}",
                    help="Penalty for recent use (weight: 30%)"
                )

            with cols[2]:
                stage = breakdown.get("stage_fit", {})
                st.metric(
                    "Stage Fit",
                    f"{stage.get('weighted', 0):.3f}",
                    help="Fit with current stage (weight: 20%)"
                )

            with cols[3]:
                random = breakdown.get("random", {})
                st.metric(
                    "Random",
                    f"{random.get('weighted', 0):.3f}",
                    help="Variation factor (weight: 10%)"
                )


def render_context_state(
    current_stage: str,
    turn_count: int,
    recent_principles: list[str],
    recent_situations: list[str]
):
    """
    Render conversation context state.

    Args:
        current_stage: Current sales stage
        turn_count: Number of turns
        recent_principles: Recently used principle IDs
        recent_situations: Recently detected situation IDs
    """
    st.markdown("##### Conversation Context")

    # Stage indicator
    stages = ["discovery", "demo", "objection_handling", "negotiation", "closing"]
    stage_idx = stages.index(current_stage) if current_stage in stages else 0
    st.progress(
        (stage_idx + 1) / len(stages),
        text=f"Stage: **{current_stage.replace('_', ' ').title()}**"
    )

    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Turns", turn_count)
    with col2:
        st.metric("Unique Principles", len(set(recent_principles)))

    # Recent history
    if recent_principles:
        st.markdown("**Recent Principles:**")
        for p in recent_principles[-3:]:
            st.text(f"  • {p}")

    if recent_situations:
        st.markdown("**Recent Situations:**")
        for s in recent_situations[-3:]:
            st.text(f"  • {s}")


def render_full_debug_panel(
    situation_candidates: Optional[list[dict]] = None,
    principle_candidates: Optional[list[dict]] = None,
    context = None
):
    """
    Render the complete debug panel.

    Args:
        situation_candidates: List of situation matches with scores
        principle_candidates: List of principle matches with scores
        context: ConversationContext object
    """
    st.markdown("## Debug Panel")

    # Situation detection
    render_situation_scores(situation_candidates or [])

    st.divider()

    # Principle selection
    render_principle_scores(principle_candidates or [])

    st.divider()

    # Context state
    if context:
        render_context_state(
            current_stage=context.current_stage,
            turn_count=context.turn_count,
            recent_principles=context.used_principles,
            recent_situations=context.used_situations
        )
    else:
        st.info("No context available yet")
