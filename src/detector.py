"""
Situation Detector - Semantic and keyword matching

Phase 1: Keyword substring matching (still available as fallback)
Phase 2: Semantic similarity via Pinecone embeddings
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class DetectedSituation:
    situation_id: str
    matched_signal: str
    applicable_principles: list[str]
    typical_stage: str
    priority: int
    confidence_score: float = 1.0  # Phase 2: Semantic similarity score


@dataclass
class SituationDetectionResult:
    """Full detection result with all candidates for debugging."""
    best_match: DetectedSituation
    all_candidates: list[DetectedSituation]
    method: str  # "semantic" or "keyword"


def load_situations(path: str = "situations.json") -> dict:
    """Load situations from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def detect_situation_semantic(
    transcript: str,
    pinecone_client,
    embed_fn: callable,
    situations: dict,
    top_k: int = 3,
    min_score: float = 0.5
) -> SituationDetectionResult:
    """
    Detect situation using semantic similarity via Pinecone.

    Args:
        transcript: The transcribed customer speech
        pinecone_client: PineconeClient instance
        embed_fn: Function to generate embeddings (embed_query)
        situations: Dictionary of situations (for fallback and full data)
        top_k: Number of top matches to return
        min_score: Minimum similarity score threshold

    Returns:
        SituationDetectionResult with best match and all candidates
    """
    # Generate embedding for transcript
    query_embedding = embed_fn(transcript)

    # Query Pinecone for similar situations
    matches = pinecone_client.query_situations(
        query_embedding=query_embedding,
        top_k=top_k,
        min_score=min_score
    )

    if not matches:
        # Fall back to keyword matching if no semantic matches
        keyword_result = detect_situation_keyword(transcript, situations)
        return SituationDetectionResult(
            best_match=keyword_result,
            all_candidates=[keyword_result],
            method="keyword_fallback"
        )

    # Convert Pinecone matches to DetectedSituation objects
    candidates = []
    for match in matches:
        candidates.append(DetectedSituation(
            situation_id=match.situation_id,
            matched_signal=match.signal_text,
            applicable_principles=match.applicable_principles,
            typical_stage=match.typical_stage,
            priority=match.priority,
            confidence_score=match.score
        ))

    # Best match is the highest scoring
    best_match = candidates[0]

    return SituationDetectionResult(
        best_match=best_match,
        all_candidates=candidates,
        method="semantic"
    )


def detect_situation_keyword(
    transcript: str,
    situations: dict
) -> DetectedSituation:
    """
    Detect situation from transcript using keyword matching (Phase 1 logic).

    Args:
        transcript: The transcribed customer speech
        situations: Dictionary of situations from situations.json

    Returns:
        DetectedSituation (always returns something, falls back to general_inquiry)
    """
    transcript_lower = transcript.lower()

    # Sort by priority (higher priority first)
    sorted_situations = sorted(
        situations.items(),
        key=lambda x: x[1].get("priority", 0),
        reverse=True
    )

    for situation_id, data in sorted_situations:
        for signal in data.get("signals", []):
            signal_lower = signal.lower()
            # First, try exact phrase match (original behavior)
            if signal_lower in transcript_lower:
                return DetectedSituation(
                    situation_id=situation_id,
                    matched_signal=signal,
                    applicable_principles=data.get("applicable_principles", []),
                    typical_stage=data.get("typical_stage", "unknown"),
                    priority=data.get("priority", 0),
                    confidence_score=1.0  # Exact match = high confidence
                )
            # Fallback: match on key words (words longer than 3 chars or common keywords)
            signal_words = [w for w in signal_lower.split() if len(w) > 3]
            important_short_words = {"too", "not", "no", "why", "how", "what", "when", "where"}
            signal_words.extend([w for w in signal_lower.split() if w in important_short_words])

            # Check if enough signal words appear in transcript
            matches = sum(1 for word in signal_words if word in transcript_lower)
            if matches >= min(2, len(signal_words)):
                return DetectedSituation(
                    situation_id=situation_id,
                    matched_signal=signal,
                    applicable_principles=data.get("applicable_principles", []),
                    typical_stage=data.get("typical_stage", "unknown"),
                    priority=data.get("priority", 0),
                    confidence_score=0.7  # Partial match = lower confidence
                )

    # No match found - return a default
    return DetectedSituation(
        situation_id="general_inquiry",
        matched_signal="",
        applicable_principles=["cialdini_liking_01"],
        typical_stage="discovery",
        priority=0,
        confidence_score=0.0
    )


# Backwards compatibility: alias for Phase 1 code
def detect_situation(transcript: str, situations: dict) -> DetectedSituation:
    """
    Detect situation (backwards compatible wrapper).

    Use detect_situation_semantic for Phase 2 semantic matching.
    """
    return detect_situation_keyword(transcript, situations)
