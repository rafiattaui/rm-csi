from huggingface_hub import snapshot_download

folder_path = snapshot_download(
    repo_id="foysalhaque/CSI-BFI-HAR-Dataset",
    repo_type="dataset",
    allow_patterns="HAR-3/*",
    local_dir="data/HAR-3"
)

print(f"Downloaded to {folder_path}")