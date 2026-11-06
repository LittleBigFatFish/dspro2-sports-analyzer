# Sports Analyzer

Football object detection for HSLU Data Science Project 2, autumn 2026. Johan Roth and Timo Tran.

## From a fresh clone to the first notebook

**Prerequisites:** Git, 64-bit Python 3.12 with venv/pip (Windows: the `py` launcher), internet access and several GB of free disk space. The reference setup uses the CPU; no NVIDIA driver, W&B account, uv or activated environment is required. On Windows choose a short local checkout path, e.g. C:\src, to avoid long-path errors.

**1. Get the project.** In PowerShell:

```powershell
git clone https://github.com/LittleBigFatFish/dspro2-sports-analyzer.git
cd dspro2-sports-analyzer
```

**2. Prepare everything once and open the notebook.** Run the next two commands separately from the project root. Wait for **Ready** before running the second:

```powershell
py -3.12 scripts/setup.py
py -3.12 scripts/setup.py lab
```

On Linux x86-64, use `python3.12` in place of `py -3.12`. The script uses the project interpreter directly. A CPU environment is created in `.venv`; keep a separate CUDA environment for GPU experiments.

**3. Explore.** JupyterLab opens [02_dataset_preparation.ipynb](notebooks/02_dataset_preparation.ipynb). Select the Python 3 kernel offered by this server and **Kernel → Restart Kernel and Run All Cells**. The first audit should report **612 images: 512 train, 100 validation; 10,136 boxes**. Continue through the class distributions, box sizes and original frames with annotations. Section 6 runs YOLO26n on the validation split at two resolutions and can take longer on a laptop. Generated reports go into `outputs/dataset_preparation/<run>/`. Stop the server with Ctrl+C in its terminal. Later, only the `lab` command is needed.

**PyCharm alternative:** run setup once, open the project, select `.venv\Scripts\python.exe` as interpreter and notebook kernel, then run Notebook 02. For a faster first inference open [Notebook 01](notebooks/01_exploration_and_baseline.ipynb), which selects seven images reproducibly from the original training split and reads their matching labels in place. No sample directories, image copies or manual pairing are needed. Both notebooks use the same raw data; the source train/validation split is preserved.

## What setup prepares

| Local path | Contents |
| --- | --- |
| `.venv/` | CPU PyTorch and Torchvision from the PyTorch CPU index, pinned project dependencies, JupyterLab and kernel |
| `data/raw/Football-Image-Dataset/` | Full [Football Image Dataset](https://github.com/skaczylo/Football-Image-Dataset) by skaczylo (CC BY 4.0), including source README and license on download. Revision `264212872f1bfa6a9a3413f298793474da741aec`, as recorded in the datasheet |
| `models/yolo26n.pt` | Pretrained detection weights, verified against the size and SHA-256 in [models/MANIFEST.json](models/MANIFEST.json) before loading |

Setup verifies the reference image and label contents on repeat runs. Existing inputs with different contents are reported rather than replaced. Data, weights and results stay local. The notebooks run inference; this setup does not train a model.

## Beyond the first notebook

- [Baseline method and interpretation](BASELINE_SETUP.md): class mapping, matching, metrics and run manifests.
- Reviewed datasheet: [Markdown](docs/datasheet/datasheet.md) · [PDF](docs/datasheet/datasheet_draft.pdf) · [LaTeX](docs/datasheet/datasheet_draft.tex) · [BibTeX](docs/datasheet/references.bib). The earlier detailed version remains available below.
- [Datasheet (PDF)](docs/datasheet/main.pdf) · [LaTeX source](docs/datasheet/main.tex): six datasets, provenance, splits and known limitations.
- [Model comparison](configs/model_comparison.yaml) and [scoring](configs/scoring.yaml): separate experiments for YOLOv8n, YOLO12n and YOLO26n. Other datasets, model weights and historical audit/report paths in those configs are **not** prerequisites for either onboarding notebook. Reproducing those experiments requires their inputs and matching audit reports.
- The web app, video tracking and tactical metrics are separate project goals. Tasks are managed in GitHub Issues and Projects.

## If setup stops

- **Python not found:** check `py -3.12 --version`. A different default Python is fine. Do not reuse a .venv copied from another machine.
- **Download interrupted:** rerun setup. Allow access to GitHub and its asset hosts, PyPI and the PyTorch CPU package index. The failing command or path is printed; no TLS bypass is needed.
- **Existing data or model differs:** keep a backup and move the reported file/directory aside before rerunning. The bootstrap expects the original dataset and weights, not corrected labels or your own fine-tuned checkpoint.
- **Browser did not open:** use the local URL printed by JupyterLab. For an import error in PyCharm, check that the notebook uses this repository's .venv.
