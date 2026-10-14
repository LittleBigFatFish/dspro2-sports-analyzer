# Sports Analyzer

A computer vision project developed for the **Data Science Project 2
(DSPRO2)** module at **Lucerne University of Applied Sciences and Arts
(HSLU)**.

**Team:** Johan Roth and Timo Tran  
**Semester:** Autumn 2026

## Project idea

The project explores how computer vision can support the analysis of
football images and videos.

Our first objective is to apply a pretrained YOLO model to football
images and visualize its predictions using bounding boxes, class labels
and confidence scores.

We plan to compare YOLOv8, YOLO12 and YOLO26 and investigate their
strengths and limitations on football data. Further development will
depend on the quality of the results and the available project time.

The system is intended to support human interpretation. Model
predictions may contain errors and should not be treated as objective
ground truth.

## Current status

The Windows CPU development environment has been set up and checked.

Completed checks:

- Python virtual environment created.
- PyTorch, Torchvision and Ultralytics imported successfully.
- Package dependencies checked with `pip check`.
- A basic PyTorch calculation executed successfully on the CPU.

Football image inference, the web application and model evaluation
have not yet been implemented.

## Planned next steps

- Select a small sample of football images.
- Run YOLO26 inference and display bounding boxes.
- Build a simple image-upload web interface.
- Explore dataset annotations and class mappings.
- Compare YOLOv8, YOLO12 and YOLO26.
- Extend processing to short football videos.

Depending on the results, we may explore fine-tuning, object tracking,
team assignment and basic football metrics such as estimated ball
possession.

## Development environment

The initial setup was verified with:

| Component | Version |
| --- | --- |
| Operating system | Windows |
| Python | 3.12.10 |
| PyTorch | 2.14.1+cpu |
| Torchvision | 0.29.1+cpu |
| Ultralytics | 8.4.177 |

The CPU environment supports development and initial experiments on
a laptop. A GPU environment for training and larger experiments will
be configured separately.

Package versions for the initial CPU environment are recorded in
`requirements-cpu.txt`.

## Setup on Windows

### Prerequisites

- Git
- Python 3.12, with the Windows Python launcher available as `py`
- Access to this repository
- Internet access for installing dependencies

Clone the repository using its GitHub clone URL, then open a PowerShell
terminal in the repository root.

Execute the following commands individually. If a command fails,
resolve the error before continuing.

### 1. Create a virtual environment

```powershell
py -3.12 -m venv .venv
```

### 2. Install the CPU build of PyTorch

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install the remaining dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

PyTorch is installed separately because `requirements-cpu.txt` records
package versions but does not record the CPU package index.

The commands explicitly use the project interpreter, so activating
the virtual environment is optional.

### 4. Verify the installation

Check dependency compatibility:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

Expected output:

```text
No broken requirements found.
```

Check imports, package versions and a CPU calculation:

```powershell
.\.venv\Scripts\python.exe -c "import torch; import torchvision; import ultralytics; from ultralytics import YOLO; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('Ultralytics:', ultralytics.__version__); print('CUDA build:', torch.version.cuda); print('CPU test:', (torch.tensor([2.0], device='cpu') * 3).item())"
```

For this CPU setup, the final two lines should be:

```text
CUDA build: None
CPU test: 6.0
```

This verifies the environment. It does not download model weights or
perform object detection.

### 5. Configure PyCharm

Select the existing interpreter at:

```text
<repository-root>\.venv\Scripts\python.exe
```

Use this interpreter for the project and its run configurations.

## Repository files

| Path | Purpose |
| --- | --- |
| `README.md` | Project overview and setup instructions |
| `requirements-cpu.txt` | Recorded dependencies for the Windows CPU environment |
| `.gitignore` | Excludes local environments and other local files |

The local `.venv/` and `.idea/` directories are not committed.

Additional source code, dataset documentation and evaluation files
will be added as the project develops.

## Data and evaluation

A project dataset has not yet been selected.

Before using a dataset, we will review its source, usage conditions,
available classes and annotation format. Dataset characteristics and
limitations will be documented in the Data Sheet.

For model comparisons, we will use consistent input data and document
the hardware, model weights and inference settings.

Quantitative detection evaluation requires suitable ground-truth
annotations. Visual inspection alone does not establish detection
accuracy.

For video datasets, we will aim to separate training, validation and
test data by match to avoid leakage between closely related frames.

## Experiment tracking

Weights & Biases is planned for recording model configurations,
dataset versions, metrics and experiment results.

Experiment tracking has not yet been configured.

## Collaboration

Tasks are managed through GitHub Issues and GitHub Projects:

**Backlog → Ready → In Progress → In Review → Done**

Changes are developed on separate branches and reviewed through pull
requests. Each team member creates their own local virtual environment.

Before closing the setup issue, the other team member should verify
the installation from a fresh clone.

## Results and limitations

Has to be explored yet.

## References

- [Ultralytics documentation](https://docs.ultralytics.com/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [Weights & Biases documentation](https://docs.wandb.ai/)