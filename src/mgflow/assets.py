import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPO = "shy0423/MGFlow"
DATA_REPO = "shy0423/MGFlow-T2I"


def download(output, post_trained=False):
    patterns = ["Reference/**", "Checkpoints/ImageNet/Base/**", "Evaluation/**"]
    if post_trained:
        patterns += ["Checkpoints/ImageNet/Post-trained/**", "Checkpoints/T2I/**"]
    snapshot_download(MODEL_REPO, local_dir=output, allow_patterns=patterns)
    destination = Path(output) / "Encoders"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(Path(output) / "Reference/Encoders.tar") as archive:
        archive.extractall(destination, filter="data")


def download_data(output, images=False):
    patterns = ["metadata.jsonl", "ATTRIBUTION.md"] + (["data/**"] if images else [])
    snapshot_download(DATA_REPO, repo_type="dataset", local_dir=output, allow_patterns=patterns)
