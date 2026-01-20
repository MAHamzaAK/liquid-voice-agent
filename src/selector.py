"""
Principle Selector - First-match and semantic selection

Phase 1: Simple first-match lookup (still available)
Phase 2: Semantic selection with multi-factor scoring
"""

import json
from typing import Optional
from dataclasses import dataclass


@dataclass
class SelectedPrinciple:
    principle_id: str
    name: str
    author: str
    book: str
    chapter: int
    page: str
    definition: str
    intervention: str
    example_response: str
    mechanism: str
    selection_score: float = 1.0  # Phase 2: Combined selection score


@dataclass
class PrincipleSelectionResult:
    """Full selection result with all candidates for debugging."""
    selected: SelectedPrinciple
    all_candidates: list[dict]  # Score breakdowns for each candidate
    method: str  # "semantic" or "first_match"


def load_principles(path: str = "principles.json") -> dict:
    """Load principles from JSON file."""
    with open(path, "r") as f:
        principles_list = json.load(f)

    # Convert list to dict keyed by principle_id
    return {p["principle_id"]: p for p in principles_list}


def select_principle_semantic(
    transcript: str,
    applicable_principles: list[str],
    principles: dict,
    pinecone_client,
    embed_fn: callable,
    recent_principles: list[str],
    current_stage: str
) -> PrincipleSelectionResult:
    """
    Select principle using semantic similarity with multi-factor scoring.

    Args:
        transcript: The customer transcript
        applicable_principles: List of principle IDs from detected situation
        principles: Dictionary of all principles
        pinecone_client: PineconeClient instance
        embed_fn: Function to generate embeddings
        recent_principles: List of recently used principle IDs
        current_stage: Current conversation stage

    Returns:
        PrincipleSelectionResult with selected principle and all candidates
    """
    from .principle_scorer import score_principles, get_score_breakdown

    if not applicable_principles:
        # Fall back to first-match if no applicable principles
        return _fallback_selection(principles)

    # Generate embedding for transcript
    query_embedding = embed_fn(transcript)

    # Query Pinecone for semantically similar principles
    principle_matches = pinecone_client.query_principles(
        query_embedding=query_embedding,
        top_k=10,
        filter_ids=applicable_principles,
        min_score=0.3
    )

    if not principle_matches:
        # Fall back to first-match if no semantic matches
        result = select_principle_first_match(applicable_principles, principles)
        if result:
            return PrincipleSelectionResult(
                selected=result,
                all_candidates=[],
                method="first_match_fallback"
            )
        return _fallback_selection(principles)

    # Score principles using multi-factor scoring
    scored = score_principles(
        principle_matches=principle_matches,
        principles_data=principles,
        recent_principles=recent_principles,
        current_stage=current_stage,
        applicable_principles=applicable_principles
    )

    if not scored:
        return _fallback_selection(principles)

    # Get the best principle
    best = scored[0]
    best_data = principles.get(best.principle_id, {})
    source = best_data.get("source", {})

    selected = SelectedPrinciple(
        principle_id=best.principle_id,
        name=best.name,
        author=best.author,
        book=best.book,
        chapter=source.get("chapter", 0),
        page=source.get("page", ""),
        definition=best_data.get("definition", ""),
        intervention=best_data.get("intervention", ""),
        example_response=best_data.get("example_response", ""),
        mechanism=best_data.get("mechanism", ""),
        selection_score=best.final_score
    )

    # Get score breakdowns for all candidates
    all_candidates = [get_score_breakdown(s) for s in scored[:5]]

    return PrincipleSelectionResult(
        selected=selected,
        all_candidates=all_candidates,
        method="semantic"
    )


def select_principle_first_match(
    applicable_principles: list[str],
    principles: dict
) -> Optional[SelectedPrinciple]:
    """
    Select the first applicable principle (Phase 1 logic).

    Args:
        applicable_principles: List of principle IDs from detected situation
        principles: Dictionary of all principles

    Returns:
        SelectedPrinciple if found, None otherwise
    """
    if not applicable_principles:
        return None

    principle_id = applicable_principles[0]

    if principle_id not in principles:
        return None

    p = principles[principle_id]
    source = p.get("source", {})

    return SelectedPrinciple(
        principle_id=principle_id,
        name=p.get("name", ""),
        author=source.get("author", ""),
        book=source.get("book", ""),
        chapter=source.get("chapter", 0),
        page=source.get("page", ""),
        definition=p.get("definition", ""),
        intervention=p.get("intervention", ""),
        example_response=p.get("example_response", ""),
        mechanism=p.get("mechanism", ""),
        selection_score=1.0
    )


def _fallback_selection(principles: dict) -> PrincipleSelectionResult:
    """Return a fallback selection when nothing else matches."""
    fallback_id = "cialdini_liking_01"
    if fallback_id in principles:
        p = principles[fallback_id]
        source = p.get("source", {})
        selected = SelectedPrinciple(
            principle_id=fallback_id,
            name=p.get("name", "Liking"),
            author=source.get("author", "Robert Cialdini"),
            book=source.get("book", "Influence"),
            chapter=source.get("chapter", 0),
            page=source.get("page", ""),
            definition=p.get("definition", ""),
            intervention=p.get("intervention", ""),
            example_response=p.get("example_response", ""),
            mechanism=p.get("mechanism", ""),
            selection_score=0.0
        )
    else:
        selected = SelectedPrinciple(
            principle_id="fallback",
            name="Active Listening",
            author="General",
            book="Sales Best Practices",
            chapter=1,
            page="1",
            definition="Demonstrate understanding by reflecting what the customer says.",
            intervention="Mirror the customer's words and ask clarifying questions.",
            example_response="I hear that you're looking for... Can you tell me more?",
            mechanism="Active listening builds rapport.",
            selection_score=0.0
        )

    return PrincipleSelectionResult(
        selected=selected,
        all_candidates=[],
        method="fallback"
    )


# Backwards compatibility
def select_principle(
    applicable_principles: list[str],
    principles: dict
) -> Optional[SelectedPrinciple]:
    """
    Select principle (backwards compatible wrapper).

    Use select_principle_semantic for Phase 2 semantic selection.
    """
    return select_principle_first_match(applicable_principles, principles)
