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
        self._initialized = False

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

        # Upload using batch_upload API with force=True to overwrite existing files
        with self.volume.batch_upload(force=True) as batch:
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
