"""
Client - Main entrypoint for the voice chatbot

Records audio, sends to Modal for processing, plays response.
"""

import datetime
import time
import signal
from pathlib import Path

import modal

# Handle both relative and absolute imports
try:
    from .audio_recorder import AudioRecorder
    from .audio_player import AudioPlayer
    from .file_manager import FileManager
    from .formatter import display_coaching_output, CoachingOutput
except ImportError:
    from audio_recorder import AudioRecorder
    from audio_player import AudioPlayer
    from file_manager import FileManager
    from formatter import display_coaching_output, CoachingOutput

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
