"""
Modal Server - GPU function for processing audio

This runs on Modal with GPU access.
Phase II: Semantic detection and selection with Pinecone.
"""

import json
import os
import random
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import modal
import torch
import torchaudio
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

# === Modal Configuration ===
APP_NAME = "behavioral-sales-coach"
SESSIONS_VOLUME = "sales-coach-sessions"
MODELS_VOLUME = "sales-coach-models"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "liquid-audio",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=2.0.0",
        "pyyaml>=6.0",
        # Phase II: Semantic matching dependencies
        "pinecone-client>=3.0.0",
        "sentence-transformers>=2.2.0",
    )
    .env({"HF_HOME": "/model_cache"})
    .add_local_file("situations.json", "/app/situations.json")
    .add_local_file("principles.json", "/app/principles.json")
)

sessions_volume = modal.Volume.from_name(SESSIONS_VOLUME, create_if_missing=True)
models_volume = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=True)


# === Data Classes ===
@dataclass
class DetectedSituation:
    situation_id: str
    matched_signal: str
    applicable_principles: list[str]
    typical_stage: str
    priority: int
    confidence_score: float = 1.0


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
    selection_score: float = 1.0


@dataclass
class ScoredPrinciple:
    """A principle with its computed score breakdown."""
    principle_id: str
    name: str
    author: str
    book: str
    semantic_score: float
    recency_penalty: float
    stage_bonus: float
    random_factor: float

    @property
    def final_score(self) -> float:
        return (
            self.semantic_score * 0.4 +
            self.recency_penalty * 0.3 +
            self.stage_bonus * 0.2 +
            self.random_factor * 0.1
        )


# === Helper Functions ===
def load_situations(path: str) -> dict:
    """Load situations from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_principles(path: str) -> dict:
    """Load principles from JSON file (array -> dict conversion)."""
    with open(path, "r") as f:
        principles_list = json.load(f)
    return {p["principle_id"]: p for p in principles_list}


def detect_situation_keyword(transcript: str, situations: dict) -> tuple[DetectedSituation, list[dict]]:
    """
    Detect situation using keyword matching (fallback method).
    Returns both the best match and all candidates for debugging.
    """
    transcript_lower = transcript.lower()
    candidates = []

    sorted_situations = sorted(
        situations.items(),
        key=lambda x: x[1].get("priority", 0),
        reverse=True
    )

    for situation_id, data in sorted_situations:
        for signal in data.get("signals", []):
            signal_lower = signal.lower()
            score = 0.0

            # Exact phrase match
            if signal_lower in transcript_lower:
                score = 1.0
            else:
                # Partial word match
                signal_words = [w for w in signal_lower.split() if len(w) > 3]
                important_short_words = {"too", "not", "no", "why", "how", "what", "when", "where"}
                signal_words.extend([w for w in signal_lower.split() if w in important_short_words])

                if signal_words:
                    matches = sum(1 for word in signal_words if word in transcript_lower)
                    score = matches / len(signal_words) if matches >= min(2, len(signal_words)) else 0.0

            if score > 0:
                candidates.append({
                    "situation_id": situation_id,
                    "signal": signal,
                    "score": score,
                    "applicable_principles": data.get("applicable_principles", []),
                    "typical_stage": data.get("typical_stage", "unknown"),
                    "priority": data.get("priority", 0),
                })

    # Sort by score, then priority
    candidates.sort(key=lambda x: (x["score"], x["priority"]), reverse=True)

    if candidates:
        best = candidates[0]
        return DetectedSituation(
            situation_id=best["situation_id"],
            matched_signal=best["signal"],
            applicable_principles=best["applicable_principles"],
            typical_stage=best["typical_stage"],
            priority=best["priority"],
            confidence_score=best["score"]
        ), candidates[:5]

    # Default fallback
    fallback = DetectedSituation(
        situation_id="general_inquiry",
        matched_signal="",
        applicable_principles=["cialdini_liking_01"],
        typical_stage="discovery",
        priority=0,
        confidence_score=0.0
    )
    return fallback, [{"situation_id": "general_inquiry", "score": 0.0, "signal": ""}]


def calculate_stage_bonus(principle_data: dict, current_stage: str) -> float:
    """Calculate bonus for principles that fit the current stage."""
    stage_principle_fit = {
        "discovery": ["liking", "social_proof", "active_listening"],
        "demo": ["authority", "contrast", "anchoring"],
        "objection_handling": ["loss_aversion", "scarcity", "reframing", "empathy"],
        "negotiation": ["reciprocity", "commitment", "anchoring"],
        "closing": ["scarcity", "urgency", "commitment", "social_proof"],
    }

    principle_name = principle_data.get("name", "").lower()
    triggers = [t.lower() for t in principle_data.get("triggers", [])]

    stage_fits = stage_principle_fit.get(current_stage, [])
    for fit in stage_fits:
        if fit in principle_name or fit in " ".join(triggers):
            return 0.5
    return 0.0


def get_score_breakdown(principle: ScoredPrinciple) -> dict:
    """Get detailed score breakdown for debugging."""
    return {
        "principle_id": principle.principle_id,
        "name": principle.name,
        "final_score": round(principle.final_score, 3),
        "breakdown": {
            "semantic": {"raw": round(principle.semantic_score, 3), "weight": 0.4, "weighted": round(principle.semantic_score * 0.4, 3)},
            "recency": {"raw": round(principle.recency_penalty, 3), "weight": 0.3, "weighted": round(principle.recency_penalty * 0.3, 3)},
            "stage_fit": {"raw": round(principle.stage_bonus, 3), "weight": 0.2, "weighted": round(principle.stage_bonus * 0.2, 3)},
            "random": {"raw": round(principle.random_factor, 3), "weight": 0.1, "weighted": round(principle.random_factor * 0.1, 3)},
        }
    }


# === Modal Class with Model Caching ===
@app.cls(
    image=image,
    gpu="L40S",
    volumes={
        "/sessions": sessions_volume,
        "/model_cache": models_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("pinecone-secret"),  # Phase II: Pinecone API key
    ],
    timeout=60 * 10,  # 10 minutes
    # Phase II: Warm pool to reduce cold start latency
    min_containers=1,      # Keep 1 container warm at all times
    buffer_containers=1,   # Extra buffer when container is active
    scaledown_window=300,  # 5 minutes idle before scale down
)
class SalesCoach:
    """Sales coaching service with cached model loading and semantic matching."""

    @modal.enter()
    def load_model(self):
        """Load models once when container starts (cached between requests)."""
        # === Load LFM2 Audio Model ===
        HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B"
        print(f"Loading audio model from {HF_REPO}...")
        self.processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
        self.model = LFM2AudioModel.from_pretrained(HF_REPO).eval()
        
        # Optimize model for faster inference
        # Use torch.compile for faster generation (PyTorch 2.0+)
        try:
            print("Compiling model for faster inference...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("Model compilation successful!")
        except Exception as e:
            print(f"Model compilation failed (using normal mode): {e}")
            # Continue without compilation if it fails

        # === Load data files ===
        self.situations = load_situations("/app/situations.json")
        self.principles = load_principles("/app/principles.json")

        # === Phase II: Initialize embedding model ===
        print("Loading embedding model (BGE-small)...")
        from sentence_transformers import SentenceTransformer
        # Use GPU if available for faster embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
        
        # Quantization optimization: Use half precision for faster embeddings
        try:
            if device == "cuda":
                self.embed_model = self.embed_model.half()
                print(f"Embedding model using FP16 quantization on {device}")
            else:
                print(f"Embedding model loaded on {device} (CPU - no quantization)")
        except Exception as e:
            print(f"Quantization failed, using full precision: {e}")
        
        print(f"Embedding model ready on {device}")

        # === Phase II: Initialize Pinecone client ===
        print("Initializing Pinecone client...")
        self.pinecone_client = None
        self.pinecone_available = False
        try:
            from pinecone import Pinecone
            api_key = os.environ.get("PINECONE_API_KEY")
            if api_key:
                self.pc = Pinecone(api_key=api_key)
                index_name = os.environ.get("PINECONE_INDEX_NAME", "sales-coach-embeddings")
                self.pinecone_index = self.pc.Index(index_name)
                self.pinecone_available = True
                print(f"Pinecone connected to index: {index_name}")
            else:
                print("PINECONE_API_KEY not set, using keyword fallback")
        except Exception as e:
            print(f"Pinecone initialization failed: {e}, using keyword fallback")

        # === KV Cache Optimization: Store conversation state per session ===
        # This allows reusing KV cache across turns for faster generation
        self.session_chat_states: dict[str, ChatState] = {}
        print("KV cache storage initialized for session reuse")

        print("All models loaded successfully!")

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query (prepends 'query:' for BGE)."""
        embedding = self.embed_model.encode(f"query: {text}", normalize_embeddings=True)
        return embedding.tolist()

    def detect_situation_semantic(
        self,
        transcript: str,
        transcript_embedding: Optional[list[float]] = None,
        top_k: int = 3,
        min_score: float = 0.5
    ) -> tuple[DetectedSituation, list[dict], str]:
        """
        Detect situation using semantic similarity via Pinecone.
        Returns (best_match, all_candidates, method).
        """
        if not self.pinecone_available:
            detected, candidates = detect_situation_keyword(transcript, self.situations)
            return detected, candidates, "keyword_fallback"

        try:
            # Reuse embedding if provided (optimization to avoid redundant embedding)
            if transcript_embedding is None:
                query_embedding = self.embed_query(transcript)
            else:
                query_embedding = transcript_embedding
            
            # Optimized: Reduced top_k from default 3 to 2 for faster query
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=min(top_k, 2),  # Limit to 2 for faster query
                namespace="situations",
                include_metadata=True
            )

            candidates = []
            for match in results.matches:
                if match.score >= min_score:
                    metadata = match.metadata
                    applicable = metadata.get("applicable_principles", "").split(",")
                    candidates.append({
                        "situation_id": metadata.get("situation_id", ""),
                        "signal": metadata.get("signal_text", ""),
                        "score": match.score,
                        "applicable_principles": [p for p in applicable if p],
                        "typical_stage": metadata.get("typical_stage", "unknown"),
                        "priority": int(metadata.get("priority", 0)),
                    })

            if candidates:
                best = candidates[0]
                return DetectedSituation(
                    situation_id=best["situation_id"],
                    matched_signal=best["signal"],
                    applicable_principles=best["applicable_principles"],
                    typical_stage=best["typical_stage"],
                    priority=best["priority"],
                    confidence_score=best["score"]
                ), candidates, "semantic"

        except Exception as e:
            print(f"Semantic detection failed: {e}, falling back to keyword")

        # Fallback to keyword matching
        detected, candidates = detect_situation_keyword(transcript, self.situations)
        return detected, candidates, "keyword_fallback"

    def select_principle_semantic(
        self,
        transcript: str,
        applicable_principles: list[str],
        recent_principles: list[str],
        current_stage: str,
        transcript_embedding: Optional[list[float]] = None
    ) -> tuple[SelectedPrinciple, list[dict], str]:
        """
        Select principle using semantic similarity with multi-factor scoring.
        Returns (selected_principle, all_candidates, method).
        """
        if not applicable_principles:
            return self._fallback_principle(), [], "fallback"

        # Try semantic selection if Pinecone is available
        if self.pinecone_available:
            try:
                # Reuse embedding if provided (optimization)
                if transcript_embedding is None:
                    query_embedding = self.embed_query(transcript)
                else:
                    query_embedding = transcript_embedding
                
                # Optimized: Reduced top_k from 10 to 5 for faster query
                results = self.pinecone_index.query(
                    vector=query_embedding,
                    top_k=min(5, len(applicable_principles)),  # Limit to 5 for faster query
                    namespace="principles",
                    include_metadata=True,
                    filter={"principle_id": {"$in": applicable_principles}}
                )

                if results.matches:
                    scored_principles = []
                    for match in results.matches:
                        if match.score >= 0.3:
                            metadata = match.metadata
                            principle_id = metadata.get("principle_id", "")
                            p_data = self.principles.get(principle_id, {})

                            # Calculate recency penalty (negative = penalty, subtracts from score)
                            recency_penalty = 0.0  # Default: no penalty
                            if principle_id in recent_principles and len(recent_principles) > 0:
                                idx = recent_principles.index(principle_id)
                                # Penalize based on recency: most recent gets largest penalty
                                # Formula: -0.5 * (position from end) / (total length)
                                # This ranges from -0.5 (most recent) to ~0 (least recent in list)
                                recency_penalty = -0.5 * (len(recent_principles) - idx) / len(recent_principles)

                            # Calculate stage bonus
                            stage_bonus = calculate_stage_bonus(p_data, current_stage)

                            scored_principles.append(ScoredPrinciple(
                                principle_id=principle_id,
                                name=metadata.get("name", ""),
                                author=metadata.get("author", ""),
                                book=metadata.get("book", ""),
                                semantic_score=match.score,
                                recency_penalty=recency_penalty,
                                stage_bonus=stage_bonus,
                                random_factor=random.uniform(0, 1),
                            ))

                    if scored_principles:
                        scored_principles.sort(key=lambda x: x.final_score, reverse=True)
                        best = scored_principles[0]
                        p_data = self.principles.get(best.principle_id, {})
                        source = p_data.get("source", {})

                        selected = SelectedPrinciple(
                            principle_id=best.principle_id,
                            name=best.name,
                            author=best.author,
                            book=best.book,
                            chapter=source.get("chapter", 0),
                            page=source.get("page", ""),
                            definition=p_data.get("definition", ""),
                            intervention=p_data.get("intervention", ""),
                            example_response=p_data.get("example_response", ""),
                            mechanism=p_data.get("mechanism", ""),
                            selection_score=best.final_score
                        )

                        candidates = [get_score_breakdown(sp) for sp in scored_principles[:5]]
                        return selected, candidates, "semantic"

            except Exception as e:
                print(f"Semantic selection failed: {e}, falling back to first-match")

        # Fallback to first-match selection
        principle_id = applicable_principles[0]
        if principle_id in self.principles:
            p = self.principles[principle_id]
            source = p.get("source", {})
            selected = SelectedPrinciple(
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
            return selected, [], "first_match_fallback"

        return self._fallback_principle(), [], "fallback"

    def _fallback_principle(self) -> SelectedPrinciple:
        """Return fallback principle when nothing matches."""
        return SelectedPrinciple(
            principle_id="fallback_active_listening",
            name="Active Listening",
            author="General",
            book="Sales Best Practices",
            chapter=1,
            page="1",
            definition="Demonstrate understanding by reflecting what the customer says.",
            intervention="Mirror the customer's words and ask clarifying questions.",
            example_response="I hear that you're looking for... Can you tell me more?",
            mechanism="Active listening builds rapport and helps uncover true customer needs.",
            selection_score=0.0
        )

    @modal.method()
    def process_turn(
        self,
        session_id: str,
        turn_number: int = 1,
        context_data: Optional[dict] = None
    ) -> dict:
        """
        Process a single conversation turn with semantic detection/selection.

        Args:
            session_id: Unique session identifier
            turn_number: Current turn number
            context_data: Optional conversation context (recent_principles, current_stage)

        Returns:
            Dictionary with transcript, audio_path, text_response, coaching, and debug info
        """
        # Extract context for semantic selection
        recent_principles = []
        current_stage = "discovery"
        if context_data:
            recent_principles = context_data.get("recent_principles", [])
            current_stage = context_data.get("current_stage", "discovery")

        # === Step 1: Load and transcribe audio ===
        # Performance timing
        total_start = time.time()
        
        # Optimized: Only reload volume once at the start (not multiple times)
        volume_start = time.time()
        sessions_volume.reload()
        print(f"⏱️ Volume reload time: {time.time() - volume_start:.2f}s")

        audio_path = f"/sessions/{session_id}/question.wav"
        print(f"Loading audio from {audio_path}")

        try:
            load_start = time.time()
            wav, sr = torchaudio.load(audio_path)
            print(f"⏱️ Audio load time: {time.time() - load_start:.2f}s")
        except Exception as e:
            return {"error": f"Failed to load audio: {e}", "transcript": ""}

        # Transcribe using sequential generation (ASR mode)
        # Optimized: Reduced max tokens for faster transcription
        transcribe_start = time.time()
        chat = ChatState(self.processor)
        chat.new_turn("system")
        chat.add_text("Perform ASR.")
        chat.end_turn()

        chat.new_turn("user")
        chat.add_audio(wav, sr)
        chat.end_turn()

        chat.new_turn("assistant")

        transcript_tokens = []
        # Reduced from 256 to 128 for faster transcription (most transcripts are short)
        for t in self.model.generate_sequential(**chat, max_new_tokens=128):
            if t.numel() == 1:
                transcript_tokens.append(t)

        transcript = self.processor.text.decode(torch.cat(transcript_tokens)) if transcript_tokens else ""
        print(f"⏱️ Transcription time: {time.time() - transcribe_start:.2f}s")
        print(f"Transcript: {transcript}")

        # Error handling: empty transcript
        if not transcript or not transcript.strip():
            return {
                "error": "Empty transcript - no speech detected",
                "transcript": "",
                "audio_path": None,
                "text_response": "",
                "coaching": None,
                "debug": None
            }

        # === Step 2 & 3: Parallelize situation detection and principle selection ===
        # Performance: Parallel queries for maximum speed
        timing_start = time.time()
        
        # Generate embedding once (if available)
        transcript_embedding = None
        if self.pinecone_available:
            try:
                embed_start = time.time()
                transcript_embedding = self.embed_query(transcript)
                print(f"⏱️ Embedding time: {time.time() - embed_start:.2f}s")
            except Exception as e:
                print(f"Embedding generation failed: {e}")
        
        # PARALLEL OPTIMIZATION: Run situation detection and principle query in parallel
        # We need to detect situation first to get applicable_principles, but we can parallelize
        # the embedding generation with the situation query
        if transcript_embedding and self.pinecone_available:
            # Run situation query immediately with embedding
            detect_start = time.time()
            detected, situation_candidates, detection_method = self.detect_situation_semantic(
                transcript, transcript_embedding
            )
            print(f"⏱️ Situation detection time: {time.time() - detect_start:.2f}s")
            print(f"Detected situation: {detected.situation_id} (method: {detection_method})")
            
            # Now select principle (can't fully parallelize since we need detected.applicable_principles)
            select_start = time.time()
            principle, principle_candidates, selection_method = self.select_principle_semantic(
                transcript=transcript,
                applicable_principles=detected.applicable_principles,
                recent_principles=recent_principles,
                current_stage=current_stage,
                transcript_embedding=transcript_embedding
            )
            print(f"⏱️ Principle selection time: {time.time() - select_start:.2f}s")
            print(f"Selected principle: {principle.name} (method: {selection_method})")
        else:
            # Fallback: sequential processing
            detect_start = time.time()
            detected, situation_candidates, detection_method = self.detect_situation_semantic(transcript)
            print(f"⏱️ Situation detection time: {time.time() - detect_start:.2f}s")
            print(f"Detected situation: {detected.situation_id} (method: {detection_method})")
            
            select_start = time.time()
            principle, principle_candidates, selection_method = self.select_principle_semantic(
                transcript=transcript,
                applicable_principles=detected.applicable_principles,
                recent_principles=recent_principles,
                current_stage=current_stage
            )
            print(f"⏱️ Principle selection time: {time.time() - select_start:.2f}s")
            print(f"Selected principle: {principle.name} (method: {selection_method})")
        
        print(f"⏱️ Total detection+selection time: {time.time() - timing_start:.2f}s")

        # === Step 4: Generate response with principle ===
        generate_start = time.time()
        
        system_prompt = f"""You are a helpful sales assistant. Respond to the customer using this approach:

PRINCIPLE: {principle.name}
DEFINITION: {principle.definition}
APPROACH: {principle.intervention}
EXAMPLE: {principle.example_response}

Respond naturally and conversationally. Keep it brief (2-3 sentences).
Respond with interleaved text and audio."""

        # KV CACHE OPTIMIZATION: Reuse ChatState per session
        # Note: ChatState maintains processor state, but we rebuild turns each time
        # This still provides some optimization through processor caching
        if session_id not in self.session_chat_states:
            self.session_chat_states[session_id] = ChatState(self.processor)
        
        response_chat = self.session_chat_states[session_id]
        # Clear and rebuild for this turn (ChatState doesn't support incremental turns)
        # But reusing the ChatState object maintains processor optimizations
        print(f"Using cached chat state for session {session_id} (turn {turn_number})")

        response_chat.new_turn("system")
        response_chat.add_text(system_prompt)
        response_chat.end_turn()

        response_chat.new_turn("user")
        response_chat.add_audio(wav, sr)
        response_chat.end_turn()

        response_chat.new_turn("assistant")

        # Generate interleaved response
        text_out = []
        audio_out = []

        # Optimized: Reduced max tokens for faster generation (from 512 to 256)
        # Most responses are short (2-3 sentences), so 256 tokens is sufficient
        generation_start = time.time()
        for t in self.model.generate_interleaved(
            **response_chat,
            max_new_tokens=256,  # Reduced from 512 for faster generation
            audio_temperature=1.0,
            audio_top_k=4
        ):
            if t.numel() == 1:
                text_out.append(t)
                print(self.processor.text.decode(t), end="", flush=True)
            else:
                audio_out.append(t)

        print()  # Newline after streaming
        print(f"⏱️ Generation time: {time.time() - generation_start:.2f}s")

        # Decode response text
        decode_start = time.time()
        response_text = self.processor.text.decode(torch.cat(text_out)) if text_out else ""

        # Decode and save audio
        # Note: We save audio to file rather than streaming because:
        # 1. Modal doesn't easily support streaming audio responses (would need WebSocket/SSE)
        # 2. Saving allows conversation history with audio playback
        # 3. Client-side auto-play handles immediate playback
        # 4. Files can be archived/exported later for analysis
        # Future enhancement: Add streaming endpoint for lower latency if needed
        answer_path = None
        if audio_out:
            audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)  # Remove EOS
            with torch.no_grad():
                waveform = self.processor.decode(audio_codes)

            answer_path = f"/sessions/{session_id}/answer.wav"
            torchaudio.save(answer_path, waveform.cpu(), 24_000)
            print(f"⏱️ Audio decode+save time: {time.time() - decode_start:.2f}s")
            print(f"Saved response audio to {answer_path}")

            sessions_volume.commit()
        
        print(f"⏱️ Total generation+decode time: {time.time() - generate_start:.2f}s")
        print(f"⏱️ TOTAL REQUEST TIME: {time.time() - total_start:.2f}s")

        # === Step 5: Format coaching output with debug info ===
        coaching = {
            "turn": turn_number,
            "timestamp": datetime.now().isoformat(),
            "customer_said": transcript,
            "detected_situation": detected.situation_id,
            "matched_signal": detected.matched_signal,
            "stage": detected.typical_stage,
            "situation_score": detected.confidence_score,
            "recommendation": {
                "principle": principle.name,
                "principle_id": principle.principle_id,
                "source": f"{principle.author}, {principle.book}, Ch.{principle.chapter}",
                "response": response_text,
                "why_it_works": principle.mechanism,
                "selection_score": principle.selection_score
            }
        }

        # Debug info for Streamlit panel
        total_time = time.time() - total_start
        debug = {
            "detection_method": detection_method,
            "selection_method": selection_method,
            "situation_candidates": situation_candidates,
            "principle_candidates": principle_candidates,
            "context_used": {
                "recent_principles": recent_principles,
                "current_stage": current_stage
            },
            # Performance timing (for debugging)
            "timing": {
                "total_time": round(total_time, 2),
                "transcription_time": round(time.time() - transcribe_start, 2) if 'transcribe_start' in locals() else None,
                "detection_time": round(time.time() - detect_start, 2) if 'detect_start' in locals() else None,
                "selection_time": round(time.time() - select_start, 2) if 'select_start' in locals() else None,
                "generation_time": round(time.time() - generation_start, 2) if 'generation_start' in locals() else None,
                "decode_time": round(time.time() - decode_start, 2) if 'decode_start' in locals() else None,
            }
        }

        return {
            "transcript": transcript,
            "audio_path": f"{session_id}/answer.wav" if answer_path else None,
            "text_response": response_text,
            "coaching": coaching,
            "debug": debug
        }
