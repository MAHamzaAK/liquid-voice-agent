"""
Test script to debug audio recording.
Records audio and saves it locally so you can verify it captured your voice.
"""

import time
from pathlib import Path
from audio_recorder import AudioRecorder


def test_recording():
    """Record audio and save for verification."""
    print("\n" + "="*50)
    print("AUDIO RECORDING TEST")
    print("="*50)

    recorder = AudioRecorder(sample_rate=16000, channels=1)

    print("\nSpeak NOW - Recording for up to 10 seconds")
    print("(Will stop after 2 seconds of silence)")
    print("-"*50)

    recorder.start_recording(silence_duration=2.0, silence_threshold=0.02)

    start_time = time.time()
    while recorder.is_recording:
        time.sleep(0.1)
        elapsed = time.time() - start_time
        if elapsed > 10:
            print("\nMax time reached")
            break

    audio_data = recorder.stop_recording()

    # Save the recording
    output_file = "test_recording.wav"
    filepath = recorder.save_to_file(output_file, audio_data)

    recorder.cleanup()

    # Show stats
    duration = len(audio_data) / 16000  # 16kHz sample rate
    print(f"\nRecording stats:")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Samples: {len(audio_data)}")
    print(f"  Saved to: {filepath.absolute()}")

    print("\n" + "="*50)
    print("NEXT STEP: Play the file to verify your voice was captured:")
    print(f"  afplay {filepath.absolute()}")
    print("="*50)


if __name__ == "__main__":
    test_recording()
