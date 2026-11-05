"""Prepare the reference CPU baseline from a fresh clone (Python 3.12 + Git)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".venv"
PYTHON = ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
NOTEBOOK = "notebooks/02_dataset_preparation.ipynb"
SOURCE = "https://github.com/skaczylo/Football-Image-Dataset.git"
REVISION = "264212872f1bfa6a9a3413f298793474da741aec"
DATA = ROOT / "data/raw/Football-Image-Dataset"
DATA_DIGEST = "fef162344d918cb32a4ecf410c69e7876396712a681ca38b6cd2f4a9e6004a30"


def run(*args):
    subprocess.run(args, cwd=ROOT, check=True)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_data(root):
    # Include relative names and contents; allow Windows text line endings.
    parts = sorted(p.relative_to(root).as_posix() for folder in ("images", "labels")
                   for p in (root / folder).rglob("*") if p.is_file())
    h = hashlib.sha256()
    for name in [*parts, "data.yaml"]:
        content = (root / name).read_bytes()
        if name.endswith((".txt", ".yaml")):
            content = content.replace(b"\r\n", b"\n")
        h.update(name.encode() + b"\0" + hashlib.sha256(content).digest())
    if h.hexdigest() != DATA_DIGEST:
        raise ValueError(f"{root}: inputs differ from the reference dataset. "
                         "Keep a backup or move the directory aside and rerun setup.")


def dataset():
    if DATA.exists():
        verify_data(DATA)
        return
    DATA.parent.mkdir(parents=True, exist_ok=True)
    # Fetch only the pinned snapshot. Publish the checkout only after validation.
    with tempfile.TemporaryDirectory(dir=DATA.parent) as work:
        checkout = Path(work) / "files"
        gitdir = Path(work) / "git"
        checkout.mkdir()
        git = ["git", "-c", "core.longpaths=true", "-c", "core.autocrlf=false",
               f"--git-dir={gitdir}", f"--work-tree={checkout}"]
        run(*git, "init", "-q")
        run(*git, "fetch", "--depth=1", SOURCE, REVISION)
        run(*git, "checkout", "--detach", "FETCH_HEAD")
        verify_data(checkout)
        checkout.rename(DATA)


def weights():
    models = ROOT / "models"
    manifest = json.loads((models / "MANIFEST.json").read_text(encoding="utf-8"))
    entry, = [e for e in manifest["files"] if e["file"] == "yolo26n.pt"]
    target = models / entry["file"]

    def verify(p):
        if p.stat().st_size != entry["bytes"] or digest(p) != entry["sha256"]:
            raise ValueError(f"Unexpected model size or SHA-256: {p}. "
                             "Inspect the file; existing weights were not overwritten.")

    if target.exists():
        verify(target)
        return
    with tempfile.TemporaryDirectory(dir=models) as work:
        tmp = Path(work) / target.name
        with urllib.request.urlopen(entry["source_url"], timeout=90) as response, tmp.open("wb") as f:
            shutil.copyfileobj(response, f)
        verify(tmp)
        tmp.rename(target)


def environment():
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Use Python 3.12: py -3.12 scripts/setup.py")
    if not PYTHON.is_file():
        if ENV.exists():
            raise ValueError(f"Incomplete environment at {ENV}. Move it aside and rerun.")
        run(sys.executable, "-m", "venv", ENV)
    run(PYTHON, "-c", "import sys; sys.exit(0 if sys.version_info[:2] == (3,12) "
        "else 'Existing .venv is not Python 3.12. Move it aside and rerun.')")
    pins = (ROOT / "requirements-cpu.txt").read_text().splitlines()
    torch_pins = [p for p in pins if p.startswith(("torch==", "torchvision=="))]
    if len(torch_pins) != 2:
        raise ValueError("Missing torch/torchvision version in requirements-cpu.txt")
    run(PYTHON, "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu", *torch_pins)
    run(PYTHON, "-m", "pip", "install", "-r", "requirements-cpu.txt", "-r", "requirements-notebook.txt")
    run(PYTHON, "-m", "pip", "check")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="setup", choices=["setup", "lab"])
    args = parser.parse_args()
    if args.action == "lab":
        if not PYTHON.is_file():
            raise FileNotFoundError("Run setup before opening JupyterLab.")
        run(PYTHON, "-m", "jupyterlab", NOTEBOOK)
        return
    environment()
    print("Checking the complete football dataset and its original splits ...", flush=True)
    dataset()
    print("Checking YOLO26n against models/MANIFEST.json ...", flush=True)
    weights()
    run(PYTHON, "-c", "import torch, torchvision, jupyterlab, ipykernel; from ultralytics import YOLO; "
        "m=YOLO('models/yolo26n.pt'); print('Loaded:', m.task, '| torch:', torch.__version__)")
    print("Ready: 512 train + 100 validation images. Open Notebook 02 with scripts/setup.py lab.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Setup failed: {exc}")
