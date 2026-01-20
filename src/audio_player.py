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
