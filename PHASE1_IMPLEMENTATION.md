# Phase 1: End-to-End Implementation Guide

> **Purpose**: This document contains everything needed for an agent to implement Phase 1 of the Behavioral Psychology Sales Coach voice chatbot. Read this document completely before starting implementation.

---

## Project Context

### What We're Building
A voice chatbot that:
1. Listens to customer audio (sales conversation)
2. Detects the sales situation (e.g., price objection, browsing, etc.)
3. Selects a behavioral psychology principle to handle the situation
4. Generates an audio response using LFM2.5-Audio
5. Displays structured coaching output explaining WHY this principle was used

### Technology Stack
- **Model**: LiquidAI/LFM2.5-Audio-1.5B (HuggingFace)
- **Compute**: Modal (serverless GPU)
- **Audio**: PyAudio (recording), Pygame (playback)
- **Language**: Python 3.11+

### Key Design Principle
**Keep it simple. Get it working first.** No fancy ML, no embeddings, no optimization. Just keyword matching and first-match selection.

---

## Data Assets (Already Exist)

### 1. principles.json
Location: `/Users/kk/Documents/liquid-audio-model/principles.json`

Contains ~80+ behavioral psychology principles from:
- Cialdini's "Influence: The Psychology of Persuasion"
- Voss's "Never Split the Difference"
- Kahneman's "Thinking, Fast and Slow"

**Structure of each principle**:
```json
{
  "principle_id": "kahneman_loss_aversion_01",
  "name": "Loss Aversion",
  "source": {
    "book": "Thinking, Fast and Slow",
    "author": "Daniel Kahneman",
    "chapter": 34,
    "page": "307-316",
    "chapter_name": "Frames and Reality",
    "path": "data/markdown/Thinking_Fast_and_Slow/chapter_34.md"
  },
  "definition": "Customers are more motivated to avoid losses than to achieve equivalent gains...",
  "triggers": ["price_resistant", "comparing_competitors"],
  "signals": ["What if I lose money on this?", "I'm worried about the risks involved."],
  "contrary_signals": ["I'm ready to invest!", "This seems like a great opportunity!"],
  "intervention": "Emphasize the potential losses of not taking action rather than just the benefits.",
  "example_response": "By not choosing this option, you could miss out on significant savings...",
  "mechanism": "Loss aversion suggests that the pain of losing is psychologically more impactful than the pleasure of gaining."
}
```

### 2. situations.json
Location: `/Users/kk/Documents/liquid-audio-model/situations.json`

Contains ~50+ sales situations mapped to applicable principles.

**Structure of each situation**:
```json
{
  "price_shock_in_store": {
    "signals": ["that is expensive", "i did not expect this price", "why is it so costly", "too much for me"],
    "contra_signals": ["quality looks worth it", "i was expecting this range"],
    "applicable_principles": ["kahneman_anchors_01", "kahneman_loss_aversion_01", "voss_labeling_01"],
    "typical_stage": "objection_handling",
    "priority": 5
  }
}
```

**Key situations include**:
- `price_shock_in_store` - Customer surprised by price
- `online_price_checking` - Comparing with Amazon/online
- `just_browsing` - Not ready to buy
- `need_to_check_with_family` - Needs approval
- `discount_expectation` - Asking for discounts
- `fear_of_wrong_choice` - Uncertain about decision
- `walking_away_pause` - About to leave

### Audio Format Requirements (LFM2.5-Audio)
- **Input**: Any sample rate (torchaudio handles conversion internally)
- **Output**: 24kHz (Mimi codec output)
- **Format**: WAV files via torchaudio
- **Language**: English only

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LOCAL MACHINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────┐                                                        │
│   │ audio_recorder │ ─── Records from microphone until 2s silence           │
│   └───────┬────────┘                                                        │
│           │ question.wav                                                    │
│           ▼                                                                  │
│   ┌────────────────┐                                                        │
│   │ file_manager   │ ─── Uploads audio to Modal volume                      │
│   └───────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│   ┌────────────────┐                                                        │
│   │ Modal Remote   │ ─── Calls server.process_turn()                        │
│   │ Function Call  │                                                        │
│   └───────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│   ┌────────────────┐     ┌─────────────────────────────────────────────┐   │
│   │ file_manager   │     │ formatter                                    │   │
│   │ (download)     │     │ (display coaching YAML in terminal)          │   │
│   └───────┬────────┘     └─────────────────────────────────────────────┘   │
│           │ answer.wav                                                      │
│           ▼                                                                  │
│   ┌────────────────┐                                                        │
│   │ audio_player   │ ─── Plays response through speakers                    │
│   └────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Modal Remote Call
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODAL (Cloud GPU - L40S)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   server.py :: process_turn(session_id)                                     │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ 1. Load audio from /sessions/{session_id}/question.wav              │   │
│   │                           │                                          │   │
│   │                           ▼                                          │   │
│   │ 2. TRANSCRIBE ─── LFM2.5-Audio sequential generation (ASR)          │   │
│   │                           │                                          │   │
│   │                           ▼ transcript: "that's too expensive"       │   │
│   │                                                                      │   │
│   │ 3. DETECT ─── detector.detect_situation(transcript)                 │   │
│   │               Simple keyword matching against situations.json        │   │
│   │                           │                                          │   │
│   │                           ▼ situation_id: "price_shock_in_store"     │   │
│   │                                                                      │   │
│   │ 4. SELECT ─── selector.select_principle(situation)                  │   │
│   │               Pick first applicable_principle from situation         │   │
│   │                           │                                          │   │
│   │                           ▼ principle: "kahneman_anchors_01"         │   │
│   │                                                                      │   │
│   │ 5. GENERATE ─── LFM2.5-Audio interleaved generation                 │   │
│   │                 System prompt includes principle details             │   │
│   │                           │                                          │   │
│   │                           ▼ audio tokens + text tokens               │   │
│   │                                                                      │   │
│   │ 6. SAVE ─── Save audio to /sessions/{session_id}/answer.wav         │   │
│   │                                                                      │   │
│   │ 7. FORMAT ─── Build coaching output dictionary                      │   │
│   │                                                                      │   │
│   │ 8. RETURN ─── {audio_path, transcript, text_response, coaching}     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Data files bundled with Modal image:                                       │
│   ├── principles.json                                                        │
│   └── situations.json                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Files to Create

### Project Structure
```
liquid-audio-model/
├── PROJECT_PLAN.md                 # Overall project plan (exists)
├── PHASE1_IMPLEMENTATION.md        # This document (exists)
├── principles.json                 # Psychology principles (exists)
├── situations.json                 # Sales situations (exists)
│
├── pyproject.toml                  # Dependencies - CREATE
├── .gitignore                      # Git ignore - CREATE
│
├── src/
│   ├── __init__.py                 # Package init - CREATE
│   │
│   ├── detector.py                 # Situation detection - CREATE
│   ├── selector.py                 # Principle selection - CREATE
│   ├── formatter.py                # Output formatting - CREATE
│   │
│   ├── audio_recorder.py           # Microphone recording - CREATE
│   ├── audio_player.py             # Audio playback - CREATE
│   ├── file_manager.py             # Modal volume ops - CREATE
│   │
│   ├── modal_app.py                # Modal configuration - CREATE
│   ├── server.py                   # Modal server function - CREATE
│   └── client.py                   # Main entrypoint - CREATE
│
└── assets/
    └── test_audio/                 # Test files - CREATE (empty folder)
```

---

## Implementation Details

### File 1: pyproject.toml

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

### File 2: .gitignore

```
# Environment
.env
venv/
.venv/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# Audio files (generated)
*.wav
assets/test_audio/*.wav

# IDE
.vscode/
.idea/

# OS
.DS_Store
```

### File 2.5: .env.example

```bash
# Environment variables template
# Copy this file to .env and fill in your values
# DO NOT commit .env to version control

# HuggingFace token (required)
# Get from: https://huggingface.co/settings/tokens
# Create a token with "Read" access
HF_TOKEN=hf_your_huggingface_token_here
```

### File 3: src/__init__.py

```python
"""Behavioral Sales Coach - Voice Chatbot with Psychology Principles"""
```

### File 4: src/detector.py

```python
"""
Situation Detector - Simple keyword matching

Phase 1: No embeddings, no ML. Just substring matching.
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


def load_situations(path: str = "situations.json") -> dict:
    """Load situations from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def detect_situation(
    transcript: str,
    situations: dict
) -> Optional[DetectedSituation]:
    """
    Detect situation from transcript using simple keyword matching.

    Args:
        transcript: The transcribed customer speech
        situations: Dictionary of situations from situations.json

    Returns:
        DetectedSituation if match found, None otherwise
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
            if signal.lower() in transcript_lower:
                return DetectedSituation(
                    situation_id=situation_id,
                    matched_signal=signal,
                    applicable_principles=data.get("applicable_principles", []),
                    typical_stage=data.get("typical_stage", "unknown"),
                    priority=data.get("priority", 0)
                )

    # No match found - return a default
    return DetectedSituation(
        situation_id="general_inquiry",
        matched_signal="",
        applicable_principles=["cialdini_liking_01"],  # Default to rapport building
        typical_stage="discovery",
        priority=0
    )
```

### File 5: src/selector.py

```python
"""
Principle Selector - Simple first-match lookup

Phase 1: No scoring, no optimization. Just pick the first applicable principle.
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


def load_principles(path: str = "principles.json") -> dict:
    """Load principles from JSON file."""
    with open(path, "r") as f:
        principles_list = json.load(f)

    # Convert list to dict keyed by principle_id
    return {p["principle_id"]: p for p in principles_list}


def select_principle(
    applicable_principles: list[str],
    principles: dict
) -> Optional[SelectedPrinciple]:
    """
    Select the first applicable principle.

    Args:
        applicable_principles: List of principle IDs from detected situation
        principles: Dictionary of all principles

    Returns:
        SelectedPrinciple if found, None otherwise
    """
    if not applicable_principles:
        return None

    # Just pick the first one (Phase 1 - keep it simple)
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
        mechanism=p.get("mechanism", "")
    )
```

### File 6: src/formatter.py

```python
"""
Output Formatter - Structured coaching output

Formats the coaching output as YAML for display.
"""

from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any
import yaml

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax


@dataclass
class CoachingOutput:
    turn: int
    timestamp: str
    customer_said: str
    detected_situation: str
    matched_signal: str
    stage: str
    recommendation: dict


def format_coaching_output(
    turn: int,
    transcript: str,
    situation_id: str,
    matched_signal: str,
    stage: str,
    principle_name: str,
    principle_source: str,
    response_text: str,
    why_it_works: str
) -> CoachingOutput:
    """
    Format the coaching output.

    Returns:
        CoachingOutput dataclass
    """
    return CoachingOutput(
        turn=turn,
        timestamp=datetime.now().isoformat(),
        customer_said=transcript,
        detected_situation=situation_id,
        matched_signal=matched_signal,
        stage=stage,
        recommendation={
            "principle": principle_name,
            "source": principle_source,
            "response": response_text,
            "why_it_works": why_it_works
        }
    )


def display_coaching_output(output: CoachingOutput) -> None:
    """Display coaching output in terminal with rich formatting."""
    console = Console()

    # Convert to YAML
    yaml_str = yaml.dump(asdict(output), default_flow_style=False, sort_keys=False)

    # Display with syntax highlighting
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)

    console.print(Panel(
        syntax,
        title=f"[bold green]Turn {output.turn} - Coaching Output[/bold green]",
        border_style="green"
    ))
```

### File 7: src/audio_recorder.py

```python
"""
Audio Recorder - Record from microphone with silence detection

Based on the Liquid AI cookbook example.
"""

import wave
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio


class AudioRecorder:
    """Records audio from microphone with automatic silence detection."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16

        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.frames: list[bytes] = []
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None

        # Silence detection
        self.silence_threshold = 0.02
        self.silence_duration = 0.0
        self.max_silence_duration: Optional[float] = None
        self.last_chunk_time: Optional[float] = None
        self.has_detected_sound = False

    def start_recording(
        self,
        silence_duration: float = 2.0,
        silence_threshold: float = 0.02,
    ) -> None:
        """Start recording with automatic silence detection."""
        if self.is_recording:
            print("Already recording!")
            return

        self.frames = []
        self.is_recording = True
        self.silence_duration = 0.0
        self.max_silence_duration = silence_duration
        self.silence_threshold = silence_threshold
        self.last_chunk_time = time.time()
        self.has_detected_sound = False

        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.is_recording = False
            return

        self.recording_thread = threading.Thread(target=self._record_loop)
        self.recording_thread.start()

        print(f"Recording... (will stop after {silence_duration}s of silence)")

    def _is_silent(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk is silent."""
        if audio_chunk.dtype == np.int16:
            normalized = audio_chunk.astype(np.float32) / 32768.0
        else:
            normalized = audio_chunk

        energy = np.sqrt(np.mean(normalized**2))
        return energy < self.silence_threshold

    def _record_loop(self) -> None:
        """Internal recording loop."""
        while self.is_recording:
            try:
                current_time = time.time()
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)

                audio_chunk = np.frombuffer(data, dtype=np.int16)

                if self.max_silence_duration is not None:
                    is_silent = self._is_silent(audio_chunk)

                    if not is_silent:
                        self.has_detected_sound = True
                        self.silence_duration = 0.0
                    elif self.has_detected_sound:
                        elapsed = current_time - self.last_chunk_time
                        self.silence_duration += elapsed

                        if self.silence_duration >= self.max_silence_duration:
                            print(f"\nSilence detected. Stopping...")
                            self.is_recording = False
                            break

                self.last_chunk_time = current_time

            except Exception as e:
                print(f"Error during recording: {e}")
                break

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return audio data."""
        self.is_recording = False

        if self.recording_thread:
            self.recording_thread.join()

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        audio_data = b"".join(self.frames)
        return np.frombuffer(audio_data, dtype=np.int16)

    def save_to_file(self, filename: str, audio_data: np.ndarray) -> Path:
        """Save audio to WAV file."""
        filepath = Path(filename)

        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

        print(f"Saved to {filepath}")
        return filepath

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.is_recording:
            self.stop_recording()
        self.audio.terminate()
```

### File 8: src/audio_player.py

```python
"""
Audio Player - Play audio files using pygame
"""

import time
from pathlib import Path


class AudioPlayer:
    """Plays audio files using pygame."""

    def __init__(self):
        self.mixer = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize pygame mixer."""
        try:
            import pygame
            pygame.mixer.init()
            self.mixer = pygame.mixer
            print("Audio player initialized")
        except ImportError:
            print("Warning: pygame not installed")
            self.mixer = None
        except Exception as e:
            print(f"Warning: Could not initialize audio player: {e}")
            self.mixer = None

    def play(self, audio_path: str, wait: bool = True) -> bool:
        """
        Play an audio file.

        Args:
            audio_path: Path to audio file
            wait: If True, wait for playback to complete

        Returns:
            True if playback started successfully
        """
        if self.mixer is None:
            print("Audio player not available")
            return False

        audio_file = Path(audio_path)
        if not audio_file.exists():
            print(f"Audio file not found: {audio_path}")
            return False

        try:
            print(f"Playing: {audio_path}")
            self.mixer.music.load(str(audio_file))
            self.mixer.music.play()

            if wait:
                while self.mixer.music.get_busy():
                    time.sleep(0.1)
                print("Playback complete")

            return True

        except Exception as e:
            print(f"Error playing audio: {e}")
            return False

    def stop(self) -> None:
        """Stop playback."""
        if self.mixer:
            self.mixer.music.stop()

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.mixer:
            self.mixer.quit()
```

### File 9: src/file_manager.py

```python
"""
File Manager - Upload/download files to Modal volumes

Uses Modal's Python Volume API instead of subprocess calls.
"""

from pathlib import Path
import time
import modal


class FileManager:
    """Manages file uploads/downloads with Modal volumes."""

    def __init__(self, volume_name: str, session_id: str):
        self.volume_name = volume_name
        self.session_id = session_id
        self.volume = modal.Volume.from_name(volume_name, create_if_missing=True)
        self.session_dir = f"/{session_id}"

    def upload(self, local_path: str, remote_filename: str = "question.wav") -> str:
        """
        Upload a local file to Modal volume.

        Returns:
            Remote path in the volume
        """
        local_file = Path(local_path)
        if not local_file.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        remote_path = f"{self.session_dir}/{remote_filename}"

        # Upload using batch_upload API
        import io
        with self.volume.batch_upload() as batch:
            # Create session directory marker
            batch.put_file(io.BytesIO(b""), f"{self.session_dir}/.marker")
            # Upload the actual file
            batch.put_file(str(local_file), remote_path)

        print(f"Uploaded to {remote_path}")
        return remote_path

    def download(
        self,
        remote_filename: str,
        local_path: str,
        max_retries: int = 5,
        retry_delay: float = 1.0
    ) -> Path:
        """
        Download a file from Modal volume using Volume API.

        Returns:
            Path to local file
        """
        local_file = Path(local_path)
        local_file.parent.mkdir(parents=True, exist_ok=True)

        remote_path = f"{self.session_dir}/{remote_filename}"

        for attempt in range(max_retries):
            try:
                print(f"Downloading {remote_filename} (attempt {attempt + 1})...")

                # Use Modal's Volume read_file API
                with open(local_file, "wb") as f:
                    for chunk in self.volume.read_file(remote_path):
                        f.write(chunk)

                print(f"Downloaded to {local_path}")
                return local_file

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Download failed: {e}. Retry in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"Download failed after {max_retries} attempts: {e}")

        raise RuntimeError("Download failed after all retries")
```

### File 10: src/modal_app.py

```python
"""
Modal App Configuration

Defines the Modal app, Docker image, volumes, and secrets.
"""

import modal


# App name - used for deployment
APP_NAME = "behavioral-sales-coach"

# Volume names
SESSIONS_VOLUME = "sales-coach-sessions"
MODELS_VOLUME = "sales-coach-models"


def get_app() -> modal.App:
    """Get the Modal app."""
    return modal.App(APP_NAME)


def get_image() -> modal.Image:
    """
    Get the Docker image with all dependencies.

    Includes:
    - Python 3.12
    - liquid-audio
    - PyTorch and torchaudio
    - FFmpeg for audio processing
    - Data files (situations.json, principles.json)
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install(
            "ffmpeg",
            "libsndfile1",
        )
        .pip_install(
            "liquid-audio",
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "numpy>=2.0.0",
            "pyyaml>=6.0",
        )
        .env({"HF_HOME": "/model_cache"})
        .copy_local_file("situations.json", "/app/situations.json")
        .copy_local_file("principles.json", "/app/principles.json")
    )


def get_volume(name: str) -> modal.Volume:
    """Get or create a Modal volume."""
    return modal.Volume.from_name(name, create_if_missing=True)


def get_secrets() -> list[modal.Secret]:
    """Get required secrets (HuggingFace token)."""
    return [modal.Secret.from_name("huggingface-secret")]
```

### File 11: src/server.py

```python
"""
Modal Server - GPU function for processing audio

This runs on Modal with GPU access.
Note: All dependencies are inlined to avoid import issues in Modal context.
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import modal
import torch
import torchaudio
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

# === Modal Configuration (inlined to avoid import issues) ===
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
    )
    .env({"HF_HOME": "/model_cache"})
    .copy_local_file("situations.json", "/app/situations.json")
    .copy_local_file("principles.json", "/app/principles.json")
)

sessions_volume = modal.Volume.from_name(SESSIONS_VOLUME, create_if_missing=True)
models_volume = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=True)


# === Inlined Data Classes ===
@dataclass
class DetectedSituation:
    situation_id: str
    matched_signal: str
    applicable_principles: list[str]
    typical_stage: str
    priority: int


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


# === Inlined Helper Functions ===
def load_situations(path: str) -> dict:
    """Load situations from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_principles(path: str) -> dict:
    """Load principles from JSON file (array -> dict conversion)."""
    with open(path, "r") as f:
        principles_list = json.load(f)
    return {p["principle_id"]: p for p in principles_list}


def detect_situation(transcript: str, situations: dict) -> DetectedSituation:
    """Detect situation from transcript using keyword matching."""
    transcript_lower = transcript.lower()

    sorted_situations = sorted(
        situations.items(),
        key=lambda x: x[1].get("priority", 0),
        reverse=True
    )

    for situation_id, data in sorted_situations:
        for signal in data.get("signals", []):
            if signal.lower() in transcript_lower:
                return DetectedSituation(
                    situation_id=situation_id,
                    matched_signal=signal,
                    applicable_principles=data.get("applicable_principles", []),
                    typical_stage=data.get("typical_stage", "unknown"),
                    priority=data.get("priority", 0)
                )

    # Default fallback
    return DetectedSituation(
        situation_id="general_inquiry",
        matched_signal="",
        applicable_principles=["cialdini_liking_01"],
        typical_stage="discovery",
        priority=0
    )


def select_principle(applicable_principles: list[str], principles: dict) -> Optional[SelectedPrinciple]:
    """Select the first applicable principle."""
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
        mechanism=p.get("mechanism", "")
    )


# === Default Fallback Principle ===
FALLBACK_PRINCIPLE = SelectedPrinciple(
    principle_id="fallback_active_listening",
    name="Active Listening",
    author="General",
    book="Sales Best Practices",
    chapter=1,
    page="1",
    definition="Demonstrate understanding by reflecting what the customer says.",
    intervention="Mirror the customer's words and ask clarifying questions.",
    example_response="I hear that you're looking for... Can you tell me more about what's important to you?",
    mechanism="Active listening builds rapport and helps uncover true customer needs."
)


# === Modal Class with Model Caching ===
@app.cls(
    image=image,
    gpu="L40S",
    volumes={
        "/sessions": sessions_volume,
        "/model_cache": models_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 10,  # 10 minutes
)
class SalesCoach:
    """Sales coaching service with cached model loading."""

    @modal.enter()
    def load_model(self):
        """Load model once when container starts (cached between requests)."""
        HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B"
        print(f"Loading model from {HF_REPO}...")

        self.processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
        self.model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

        # Load data files
        self.situations = load_situations("/app/situations.json")
        self.principles = load_principles("/app/principles.json")

        print("Model and data loaded successfully!")

    @modal.method()
    def process_turn(self, session_id: str, turn_number: int = 1) -> dict:
        """
        Process a single conversation turn.

        Args:
            session_id: Unique session identifier
            turn_number: Current turn number

        Returns:
            Dictionary with transcript, audio_path, text_response, and coaching
        """
        # === Step 1: Load and transcribe audio ===
        audio_path = f"/sessions/{session_id}/question.wav"
        print(f"Loading audio from {audio_path}")

        try:
            wav, sr = torchaudio.load(audio_path)
        except Exception as e:
            return {"error": f"Failed to load audio: {e}", "transcript": ""}

        # Transcribe using sequential generation (ASR mode)
        # Use "Perform ASR." system prompt per official examples
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

        transcript = self.processor.text.decode(torch.cat(transcript_tokens)) if transcript_tokens else ""
        print(f"Transcript: {transcript}")

        # Error handling: empty transcript
        if not transcript or not transcript.strip():
            return {
                "error": "Empty transcript - no speech detected",
                "transcript": "",
                "audio_path": None,
                "text_response": "",
                "coaching": None
            }

        # === Step 2: Detect situation ===
        detected = detect_situation(transcript, self.situations)
        print(f"Detected situation: {detected.situation_id}")

        # === Step 3: Select principle ===
        principle = select_principle(detected.applicable_principles, self.principles)

        # Error handling: use fallback if no principle found
        if principle is None:
            print("No matching principle found, using fallback")
            principle = FALLBACK_PRINCIPLE

        print(f"Selected principle: {principle.name}")

        # === Step 4: Generate response with principle ===
        system_prompt = f"""You are a helpful sales assistant. Respond to the customer using this approach:

PRINCIPLE: {principle.name}
DEFINITION: {principle.definition}
APPROACH: {principle.intervention}
EXAMPLE: {principle.example_response}

Respond naturally and conversationally. Keep it brief (2-3 sentences).
Respond with interleaved text and audio."""

        response_chat = ChatState(self.processor)

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

        for t in self.model.generate_interleaved(
            **response_chat,
            max_new_tokens=512,
            audio_temperature=1.0,
            audio_top_k=4
        ):
            if t.numel() == 1:
                text_out.append(t)
                print(self.processor.text.decode(t), end="", flush=True)
            else:
                audio_out.append(t)

        print()  # Newline after streaming

        # Decode response text
        response_text = self.processor.text.decode(torch.cat(text_out)) if text_out else ""

        # Decode and save audio
        answer_path = None
        if audio_out:
            audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)  # Remove EOS
            with torch.no_grad():
                waveform = self.processor.decode(audio_codes)

            answer_path = f"/sessions/{session_id}/answer.wav"
            torchaudio.save(answer_path, waveform.cpu(), 24_000)
            print(f"Saved response audio to {answer_path}")

            # Commit volume changes
            sessions_volume.commit()

        # === Step 5: Format coaching output ===
        coaching = {
            "turn": turn_number,
            "timestamp": datetime.now().isoformat(),
            "customer_said": transcript,
            "detected_situation": detected.situation_id,
            "matched_signal": detected.matched_signal,
            "stage": detected.typical_stage,
            "recommendation": {
                "principle": principle.name,
                "source": f"{principle.author}, {principle.book}, Ch.{principle.chapter}",
                "response": response_text,
                "why_it_works": principle.mechanism
            }
        }

        return {
            "transcript": transcript,
            "audio_path": f"{session_id}/answer.wav" if answer_path else None,
            "text_response": response_text,
            "coaching": coaching
        }
```

### File 12: src/client.py

```python
"""
Client - Main entrypoint for the voice chatbot

Records audio, sends to Modal for processing, plays response.
"""

import datetime
import time
import signal
from pathlib import Path

import modal

from .audio_recorder import AudioRecorder
from .audio_player import AudioPlayer
from .file_manager import FileManager
from .formatter import display_coaching_output, CoachingOutput

# Constants (matching server.py)
APP_NAME = "behavioral-sales-coach"
SESSIONS_VOLUME = "sales-coach-sessions"

# Create local app for entrypoint
app = modal.App(APP_NAME)


class RecordingTimeout(Exception):
    """Raised when recording times out."""
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise RecordingTimeout("Recording timed out")


def generate_session_id() -> str:
    """Generate unique session ID from timestamp."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def record_audio(max_duration: int = 60) -> Path:
    """
    Record audio from microphone with timeout.

    Args:
        max_duration: Maximum recording time in seconds (default 60)

    Returns:
        Path to recorded audio file
    """
    recorder = AudioRecorder(sample_rate=16000, channels=1)

    print("\n" + "="*50)
    print("SPEAK NOW - Recording will stop after 2s of silence")
    print(f"(Maximum recording time: {max_duration}s)")
    print("="*50 + "\n")

    recorder.start_recording(silence_duration=2.0, silence_threshold=0.02)

    # Set up timeout (Unix only - will be ignored on Windows)
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max_duration)
    except (AttributeError, ValueError):
        # SIGALRM not available on Windows
        pass

    try:
        # Wait for recording to complete
        start_time = time.time()
        while recorder.is_recording:
            time.sleep(0.1)
            # Manual timeout check for cross-platform support
            if time.time() - start_time > max_duration:
                print(f"\nMax recording time ({max_duration}s) reached.")
                break
    finally:
        # Cancel the alarm
        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass

    audio_data = recorder.stop_recording()

    # Save to temp file
    filename = "user_recording.wav"
    filepath = recorder.save_to_file(filename, audio_data)

    recorder.cleanup()

    return filepath


@app.local_entrypoint()
def main():
    """Main conversation loop."""
    print("\n" + "="*60)
    print("  BEHAVIORAL PSYCHOLOGY SALES COACH")
    print("  Phase 1 - End-to-End Pipeline")
    print("="*60)

    session_id = generate_session_id()
    print(f"\nSession ID: {session_id}")

    # Initialize components
    file_manager = FileManager(
        volume_name=SESSIONS_VOLUME,
        session_id=session_id
    )
    player = AudioPlayer()

    # Get the SalesCoach class from deployed Modal app
    SalesCoach = modal.Cls.from_name(APP_NAME, "SalesCoach")

    turn_number = 0

    try:
        while True:
            turn_number += 1
            print(f"\n--- Turn {turn_number} ---")

            # 1. Record audio
            try:
                audio_file = record_audio(max_duration=60)
            except RecordingTimeout:
                print("Recording timed out. Please try again.")
                continue

            # 2. Upload to Modal
            print("\nUploading audio...")
            file_manager.upload(str(audio_file), "question.wav")

            # 3. Process on Modal (GPU) using class method
            print("\nProcessing (this may take a moment on first run)...")
            result = SalesCoach().process_turn.remote(session_id, turn_number)

            # 4. Check for errors
            if "error" in result and result["error"]:
                print(f"\nError: {result['error']}")
                print("Please try speaking again.")
                continue

            # 5. Display coaching output
            if result.get("coaching"):
                coaching = CoachingOutput(
                    turn=result["coaching"]["turn"],
                    timestamp=result["coaching"]["timestamp"],
                    customer_said=result["coaching"]["customer_said"],
                    detected_situation=result["coaching"]["detected_situation"],
                    matched_signal=result["coaching"]["matched_signal"],
                    stage=result["coaching"]["stage"],
                    recommendation=result["coaching"]["recommendation"]
                )
                display_coaching_output(coaching)

            # 6. Download and play response
            if result.get("audio_path"):
                local_answer = f"answer_{session_id}_{turn_number}.wav"
                try:
                    file_manager.download("answer.wav", local_answer)
                    print("\nPlaying response...")
                    player.play(local_answer)
                except Exception as e:
                    print(f"\nCould not play audio: {e}")

            # 7. Ask to continue
            print("\n" + "-"*40)
            cont = input("Continue conversation? (y/n): ").strip().lower()
            if cont != 'y':
                break

    except KeyboardInterrupt:
        print("\n\nConversation ended by user.")

    finally:
        player.cleanup()
        print(f"\nSession {session_id} complete.")


if __name__ == "__main__":
    main()
```

---

## Setup Instructions

### Step 1: Create Modal Account
1. Go to https://modal.com
2. Sign up / Log in
3. Note: Free tier includes $30/month credits

### Step 2: Create HuggingFace Token
1. Go to https://huggingface.co/settings/tokens
2. Create new token with "Read" access
3. Copy the token (starts with `hf_`)

### Step 3: Install Modal CLI & Authenticate
```bash
# Install
pip install modal

# Authenticate (opens browser)
modal token new
```

### Step 4: Create Modal Secret
```bash
# Replace hf_xxx with your actual token
modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

### Step 5: Install Project Dependencies
```bash
cd /Users/kk/Documents/liquid-audio-model

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Or use uv
uv sync
```

### Step 6: Deploy to Modal
```bash
# Deploy the server function
modal deploy src/server.py
```

### Step 7: Run the Client
```bash
# Run the conversation loop
modal run src/client.py

# Or
python -m src.client
```

---

## Expected Output

When running, you should see:

```
==============================================================
  BEHAVIORAL PSYCHOLOGY SALES COACH
  Phase 1 - End-to-End Pipeline
==============================================================

Session ID: 20250119_153045

--- Turn 1 ---

==================================================
SPEAK NOW - Recording will stop after 2s of silence
==================================================

Recording... (will stop after 2.0s of silence)
Silence detected. Stopping...
Saved to user_recording.wav

Uploading audio...
Uploaded to /20250119_153045/question.wav

Processing (this may take a moment on first run)...

╭──────────────────────────────────────────────────────────────╮
│ Turn 1 - Coaching Output                                     │
├──────────────────────────────────────────────────────────────┤
│ turn: 1                                                      │
│ timestamp: '2025-01-19T15:30:55'                             │
│ customer_said: "that's too expensive i saw it on amazon"     │
│ detected_situation: online_price_checking                    │
│ matched_signal: "amazon has this cheaper"                    │
│ stage: objection_handling                                    │
│ recommendation:                                              │
│   principle: Reference Point                                 │
│   source: Kahneman, Thinking Fast and Slow, Ch.26            │
│   response: "I understand you're comparing prices..."        │
│   why_it_works: "Customers evaluate based on reference..."   │
╰──────────────────────────────────────────────────────────────╯

Downloading answer.wav (attempt 1)...
Downloaded to answer_20250119_153045_1.wav

Playing response...
Playing: answer_20250119_153045_1.wav
Playback complete

----------------------------------------
Continue conversation? (y/n):
```

---

## Troubleshooting

### "No module named 'src'"
```bash
# Make sure you're in the project directory and installed with -e
cd /Users/kk/Documents/liquid-audio-model
pip install -e .
```

### "Modal authentication failed"
```bash
modal token new
```

### "HuggingFace access denied"
```bash
# 1. Make sure you accepted model terms at HuggingFace
# 2. Recreate the secret
modal secret create huggingface-secret HF_TOKEN=hf_your_new_token
```

### "No microphone access"
- macOS: System Preferences > Security & Privacy > Privacy > Microphone
- Grant access to Terminal or your IDE

### "Data files not found on server"
The situations.json and principles.json need to be accessible on the Modal server. Options:
1. Bundle them in the Docker image
2. Upload to a Modal volume
3. Fetch from a URL

---

## Files Checklist

```
[x] PROJECT_PLAN.md (exists)
[x] PHASE1_IMPLEMENTATION.md (this file - exists)
[x] principles.json (exists)
[x] situations.json (exists)

[ ] pyproject.toml - CREATE
[ ] .gitignore - CREATE
[ ] .env.example - CREATE (template for environment variables)
[ ] src/__init__.py - CREATE
[ ] src/detector.py - CREATE
[ ] src/selector.py - CREATE
[ ] src/formatter.py - CREATE
[ ] src/audio_recorder.py - CREATE
[ ] src/audio_player.py - CREATE
[ ] src/file_manager.py - CREATE
[ ] src/modal_app.py - CREATE
[ ] src/server.py - CREATE
[ ] src/client.py - CREATE
```

---

## Next Agent Instructions

1. Read this document completely
2. Create all files listed in "Files to Create" section
3. Follow setup instructions
4. Test end-to-end
5. Fix any issues

**Important Notes**:
- Keep implementations simple (Phase 1)
- Don't add features not in this document
- Test each component before integration
- The data files (principles.json, situations.json) already exist

---

*Document created: 2025-01-19*
*For: Phase 1 Implementation*
