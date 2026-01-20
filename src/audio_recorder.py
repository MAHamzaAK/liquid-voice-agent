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
