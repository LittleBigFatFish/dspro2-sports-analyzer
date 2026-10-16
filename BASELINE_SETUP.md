# Baseline learning package

This is a proposed implementation for review and adaptation, not a completed coursework submission.

## Integrate

Copy the contents into your existing repository root, preserving directory structure. Review collisions before replacing any file. The package does not replace README.md, .gitignore, requirements-cpu.txt, sports_analyzer/__init__.py or your data_audit.py. New annotations/__init__.py is included. Your existing project package initializer should remain in place.

Install notebook tools from the project terminal:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_baseline.py -v
```

Open notebooks/01_exploration_and_baseline.ipynb in PyCharm and select the project .venv interpreter as kernel. Restart the kernel and run all cells. If notebook execution is unavailable in your IDE, install JupyterLab in this same environment and launch it with `python -m jupyterlab` using the .venv interpreter.

## Required local files

- data/samples/: the seven original JPG images (not prediction overlays).
- data/sample-labels/: source labels with exactly matching image stems.
- data/source-metadata/data.yaml: original dataset configuration. Change the config if named data.yml.
- models/yolo26n.pt: weights already downloaded during the CLI test.

Change only configs/baseline.yaml for paths and experiment settings. All paths resolve against the repository root. Only JPG/JPEG/PNG inputs are included; this initial notebook is image-only and CPU-only. It never downloads weights automatically.

## Workflow and outputs

1. Validate images, labels, classes, geometry and orientation.
2. Show originals and source annotations; chart class counts and original pixel box sizes.
3. Record input, label, source code and checkpoint hashes, versions, Git state and configuration.
4. Warm up once per input size; run 640 and 1280 at confidence 0.25.
5. Map player/referee to person and sports ball to ball.
6. Match per image and class, descending confidence, highest IoU among unmatched GT, threshold 0.5 inclusive.
7. Export per-class and per-image TP/FP/FN, precision/recall, timings, raw predictions, matching events and figures.

Outputs go to outputs/baseline/<UTC timestamp>/. Each manifest is marked incomplete until the last code cell succeeds. For reproducibility restart/run all rather than rerunning only selected export cells. Never interpret an incomplete run as a successful evaluation.

Missing labels fail; empty labels are allowed and reported. Out-of-bounds boxes fail except for a 1e-6 rounding tolerance (warned, unchanged). Sides below 5 pixels produce warnings, not deletions. Source files are never modified. Ground truth with unreviewed empty annotations or missing out-of-pitch persons can make precision misleading. This is explicitly a development-reference evaluation, not full-scene detection accuracy.

## Validation performed here

- Four unit tests passed: coordinate conversion and invalid data, duplicate prediction accounting, missed/wrong-class predictions, role mapping.
- Notebook schema and every code cell compiled successfully.
- Every code cell was run in sequence on a synthetic image/labels with a mocked detector. Metrics, exports and complete manifest verified; a generated ground-truth figure visually inspected.
- Real YOLO inference and the real seven-image notebook were NOT executed here: complete source data, labels and local weights were unavailable. Notebook is delivered without fabricated outputs. Real evaluation must be run in the user's verified CPU environment.

## Limits and next steps

This is fixed-threshold precision/recall, not mAP. Prediction-call timings exclude model loading, plotting and saving; no rigorous latency benchmark is claimed. Seven selected images are development evidence only. Untested generalization, annotation scope, possible near-duplicate frames and pretrained-data overlap remain limitations. Keep a later test split untouched and use a standard evaluator for mAP.

Dependencies for notebook tools are bounded, not a complete lock. Record resolved versions after installation. Do not silently replace the validated CPU environment snapshot. Add W&B only after the local run succeeds; the JSON/CSV outputs already provide traceable local experiment records.

Official inference API: https://docs.ultralytics.com/modes/predict/
Dataset: https://github.com/skaczylo/Football-Image-Dataset
