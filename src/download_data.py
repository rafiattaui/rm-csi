from pathlib import Path
from huggingface_hub import enable_progress_bars, snapshot_download

# Anchors path to project root relative to this file's location
SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_DIR = SCRIPT_DIR.parent / "data" / "HAR-3"

enable_progress_bars()

folder_path = snapshot_download(
    repo_id="foysalhaque/CSI-BFI-HAR-Dataset",
    repo_type="dataset",
    allow_patterns="HAR-3/*",
    local_dir=str(TARGET_DIR),
)

print(f"Downloaded to {folder_path}")