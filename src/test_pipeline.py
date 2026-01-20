"""
Test the full pipeline with a pre-recorded audio file.
Bypasses microphone to test the Modal processing.
"""

import modal

# Import local modules
from file_manager import FileManager
from formatter import display_coaching_output, CoachingOutput

APP_NAME = "behavioral-sales-coach"
SESSIONS_VOLUME = "sales-coach-sessions"


def test_pipeline(audio_file: str = "test_audio.wav"):
    """Test the pipeline with a specific audio file."""
    print("\n" + "="*60)
    print("  PIPELINE TEST (bypassing microphone)")
    print("="*60)

    session_id = "test_session"

    # Initialize
    file_manager = FileManager(
        volume_name=SESSIONS_VOLUME,
        session_id=session_id
    )

    # Get the SalesCoach class
    SalesCoach = modal.Cls.from_name(APP_NAME, "SalesCoach")

    # Upload test audio
    print(f"\n1. Uploading {audio_file}...")
    file_manager.upload(audio_file, "question.wav")

    # Process
    print("\n2. Processing on Modal GPU...")
    result = SalesCoach().process_turn.remote(session_id, 1)

    # Display results
    print("\n3. Results:")
    print("-"*40)
    print(f"Transcript: {result.get('transcript', 'N/A')}")
    print(f"Error: {result.get('error', 'None')}")

    if result.get("coaching"):
        print("\n4. Coaching output:")
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

    print("\n" + "="*60)
    print("Pipeline test complete!")
    print("="*60)


if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test_audio.wav"
    test_pipeline(audio_file)
