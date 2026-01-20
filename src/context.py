"""
Conversation Context - Multi-turn state tracking

Phase II: Track conversation history to avoid repetition and improve coherence.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


@dataclass
class TurnRecord:
    """Record of a single conversation turn."""
    turn_number: int
    timestamp: str
    transcript: str
    detected_situation: str
    situation_score: float
    selected_principle: str
    principle_score: float
    response_text: str


@dataclass
class ConversationContext:
    """
    Tracks conversation state across multiple turns.

    Stores history of situations and principles used to:
    - Avoid selecting the same principle repeatedly
    - Track conversation progression through sales stages
    - Build richer prompts with context
    """
    session_id: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    current_stage: str = "discovery"
    turn_history: list[TurnRecord] = field(default_factory=list)

    @property
    def turn_count(self) -> int:
        """Number of turns in this conversation."""
        return len(self.turn_history)

    @property
    def used_principles(self) -> list[str]:
        """List of principle IDs used in this conversation."""
        return [t.selected_principle for t in self.turn_history]

    @property
    def used_situations(self) -> list[str]:
        """List of situation IDs detected in this conversation."""
        return [t.detected_situation for t in self.turn_history]

    @property
    def recent_principles(self) -> list[str]:
        """Last 3 principles used (for recency penalty)."""
        return self.used_principles[-3:]

    @property
    def last_transcript(self) -> Optional[str]:
        """The most recent customer transcript."""
        if self.turn_history:
            return self.turn_history[-1].transcript
        return None

    def add_turn(
        self,
        transcript: str,
        detected_situation: str,
        situation_score: float,
        selected_principle: str,
        principle_score: float,
        response_text: str
    ) -> TurnRecord:
        """
        Add a new turn to the conversation history.

        Args:
            transcript: Customer's transcribed speech
            detected_situation: ID of detected situation
            situation_score: Confidence score for situation detection
            selected_principle: ID of selected principle
            principle_score: Final score for principle selection
            response_text: Generated response text

        Returns:
            The created TurnRecord
        """
        record = TurnRecord(
            turn_number=self.turn_count + 1,
            timestamp=datetime.now().isoformat(),
            transcript=transcript,
            detected_situation=detected_situation,
            situation_score=situation_score,
            selected_principle=selected_principle,
            principle_score=principle_score,
            response_text=response_text
        )
        self.turn_history.append(record)
        self._update_stage(detected_situation)
        return record

    def _update_stage(self, detected_situation: str):
        """
        Update conversation stage based on detected situation.

        Simple heuristic: Move through stages as certain situations are detected.
        """
        stage_indicators = {
            "discovery": ["initial_inquiry", "information_gathering", "general_inquiry"],
            "demo": ["product_interest", "feature_question", "comparison_request"],
            "objection_handling": ["price_shock", "competitor_mention", "hesitation", "skepticism"],
            "negotiation": ["discount_request", "bundle_interest", "value_question"],
            "closing": ["ready_to_buy", "payment_question", "timeline_question"],
        }

        # Check if situation suggests a later stage
        stage_order = ["discovery", "demo", "objection_handling", "negotiation", "closing"]
        current_idx = stage_order.index(self.current_stage)

        for stage, indicators in stage_indicators.items():
            if any(ind in detected_situation for ind in indicators):
                stage_idx = stage_order.index(stage)
                # Only advance, never go back
                if stage_idx > current_idx:
                    self.current_stage = stage
                    break

    def get_summary(self) -> str:
        """
        Get a text summary of the conversation for prompt injection.

        Returns:
            A concise summary of conversation context
        """
        if not self.turn_history:
            return "This is the first turn of the conversation."

        lines = [
            f"Conversation has {self.turn_count} turn(s) so far.",
            f"Current stage: {self.current_stage}",
        ]

        if self.turn_count >= 2:
            recent = self.turn_history[-2:]
            lines.append("Recent history:")
            for turn in recent:
                lines.append(f"- Turn {turn.turn_number}: Customer mentioned '{turn.transcript[:50]}...' (situation: {turn.detected_situation})")

        # Note principles to avoid
        if self.recent_principles:
            lines.append(f"Recently used principles (avoid if possible): {', '.join(self.recent_principles)}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize context to dictionary for passing to Modal."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "current_stage": self.current_stage,
            "turn_history": [
                {
                    "turn_number": t.turn_number,
                    "timestamp": t.timestamp,
                    "transcript": t.transcript,
                    "detected_situation": t.detected_situation,
                    "situation_score": t.situation_score,
                    "selected_principle": t.selected_principle,
                    "principle_score": t.principle_score,
                    "response_text": t.response_text,
                }
                for t in self.turn_history
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationContext":
        """Deserialize context from dictionary."""
        ctx = cls(
            session_id=data["session_id"],
            started_at=data.get("started_at", datetime.now().isoformat()),
            current_stage=data.get("current_stage", "discovery"),
        )
        for t in data.get("turn_history", []):
            ctx.turn_history.append(TurnRecord(
                turn_number=t["turn_number"],
                timestamp=t["timestamp"],
                transcript=t["transcript"],
                detected_situation=t["detected_situation"],
                situation_score=t["situation_score"],
                selected_principle=t["selected_principle"],
                principle_score=t["principle_score"],
                response_text=t["response_text"],
            ))
        return ctx

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "ConversationContext":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
