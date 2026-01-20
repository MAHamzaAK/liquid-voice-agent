"""
Principle Scorer - Multi-factor scoring for principle selection

Phase II: Replaces simple first-match with intelligent scoring that considers:
- Semantic relevance to transcript (40%)
- Recency penalty for recently used principles (30%)
- Stage fit (20%)
- Random variation (10%)
"""

import random
from dataclasses import dataclass
from typing import Optional

from .pinecone_client import PrincipleMatch


@dataclass
class ScoredPrinciple:
    """A principle with its computed score breakdown."""
    principle_id: str
    name: str
    author: str
    book: str

    # Score components
    semantic_score: float      # From Pinecone query (0-1)
    recency_penalty: float     # 0 if not recent, negative otherwise
    stage_bonus: float         # Bonus for stage fit (0 or positive)
    random_factor: float       # Small random variation

    @property
    def final_score(self) -> float:
        """Calculate final weighted score."""
        return (
            self.semantic_score * 0.4 +
            self.recency_penalty * 0.3 +
            self.stage_bonus * 0.2 +
            self.random_factor * 0.1
        )


def score_principles(
    principle_matches: list[PrincipleMatch],
    principles_data: dict,
    recent_principles: list[str],
    current_stage: str,
    applicable_principles: Optional[list[str]] = None
) -> list[ScoredPrinciple]:
    """
    Score principles using multiple factors.

    Args:
        principle_matches: Results from Pinecone query
        principles_data: Full principles dictionary
        recent_principles: List of recently used principle IDs
        current_stage: Current conversation stage
        applicable_principles: Optional list to filter by

    Returns:
        List of ScoredPrinciple sorted by final score (highest first)
    """
    scored = []

    for match in principle_matches:
        principle_id = match.principle_id

        # Skip if not in applicable list (when provided)
        if applicable_principles and principle_id not in applicable_principles:
            continue

        # Get full principle data
        p_data = principles_data.get(principle_id, {})
        source = p_data.get("source", {})

        # Calculate recency penalty
        recency_penalty = 0.0
        if principle_id in recent_principles:
            # More recent = stronger penalty
            try:
                idx = recent_principles.index(principle_id)
                recency_penalty = -0.5 * (len(recent_principles) - idx) / len(recent_principles)
            except ValueError:
                pass

        # Calculate stage bonus
        stage_bonus = _calculate_stage_bonus(p_data, current_stage)

        # Add small random factor for variety
        random_factor = random.uniform(0, 1)

        scored.append(ScoredPrinciple(
            principle_id=principle_id,
            name=match.name,
            author=match.author,
            book=match.book,
            semantic_score=match.score,
            recency_penalty=recency_penalty,
            stage_bonus=stage_bonus,
            random_factor=random_factor,
        ))

    # Sort by final score (highest first)
    scored.sort(key=lambda x: x.final_score, reverse=True)
    return scored


def _calculate_stage_bonus(principle_data: dict, current_stage: str) -> float:
    """
    Calculate bonus for principles that fit the current stage.

    Args:
        principle_data: Full principle data dictionary
        current_stage: Current conversation stage

    Returns:
        Bonus score (0.0 to 0.5)
    """
    # Map principles to stages they're best suited for
    stage_principle_fit = {
        "discovery": ["liking", "social_proof", "active_listening"],
        "demo": ["authority", "contrast", "anchoring"],
        "objection_handling": ["loss_aversion", "scarcity", "reframing", "empathy"],
        "negotiation": ["reciprocity", "commitment", "anchoring"],
        "closing": ["scarcity", "urgency", "commitment", "social_proof"],
    }

    principle_name = principle_data.get("name", "").lower()
    triggers = [t.lower() for t in principle_data.get("triggers", [])]

    # Check if principle fits current stage
    stage_fits = stage_principle_fit.get(current_stage, [])
    for fit in stage_fits:
        if fit in principle_name or fit in " ".join(triggers):
            return 0.5

    return 0.0


def select_best_principle(
    scored_principles: list[ScoredPrinciple],
    min_score: float = 0.3
) -> Optional[ScoredPrinciple]:
    """
    Select the best principle from scored list.

    Args:
        scored_principles: List of scored principles (already sorted)
        min_score: Minimum final score threshold

    Returns:
        Best ScoredPrinciple or None if none meet threshold
    """
    if not scored_principles:
        return None

    best = scored_principles[0]
    if best.final_score >= min_score:
        return best

    return None


def get_score_breakdown(principle: ScoredPrinciple) -> dict:
    """
    Get detailed score breakdown for debugging/visualization.

    Args:
        principle: A scored principle

    Returns:
        Dictionary with score components and weights
    """
    return {
        "principle_id": principle.principle_id,
        "name": principle.name,
        "final_score": round(principle.final_score, 3),
        "breakdown": {
            "semantic": {
                "raw": round(principle.semantic_score, 3),
                "weight": 0.4,
                "weighted": round(principle.semantic_score * 0.4, 3),
            },
            "recency": {
                "raw": round(principle.recency_penalty, 3),
                "weight": 0.3,
                "weighted": round(principle.recency_penalty * 0.3, 3),
            },
            "stage_fit": {
                "raw": round(principle.stage_bonus, 3),
                "weight": 0.2,
                "weighted": round(principle.stage_bonus * 0.2, 3),
            },
            "random": {
                "raw": round(principle.random_factor, 3),
                "weight": 0.1,
                "weighted": round(principle.random_factor * 0.1, 3),
            },
        }
    }
