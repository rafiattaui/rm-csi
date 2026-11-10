from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

# Anchor path to the project root directory regardless of execution location
SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_DIR = SCRIPT_DIR.parent / "data" / "HAR-3"

folder_path = snapshot_download(
    repo_id="foysalhaque/CSI-BFI-HAR-Dataset",
    repo_type="dataset",
    allow_patterns="HAR-3/*",
    local_dir=str(TARGET_DIR),
    tqdm_class=tqdm,
)

print(f"Downloaded to {folder_path}")