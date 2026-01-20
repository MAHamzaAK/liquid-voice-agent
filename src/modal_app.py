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
        .add_local_file("situations.json", "/app/situations.json")
        .add_local_file("principles.json", "/app/principles.json")
    )


def get_volume(name: str) -> modal.Volume:
    """Get or create a Modal volume."""
    return modal.Volume.from_name(name, create_if_missing=True)


def get_secrets() -> list[modal.Secret]:
    """Get required secrets (HuggingFace token)."""
    return [modal.Secret.from_name("huggingface-secret")]
