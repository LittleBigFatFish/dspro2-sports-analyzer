# Pretrained football baseline

For installation, data, weights and notebook startup, use the [README quickstart](README.md#from-a-fresh-clone-to-the-first-notebook). The following describes Notebook 01 on seven seeded training images read directly from the full dataset. Notebook 02 explores the complete dataset and also evaluates the validation split.

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

## Limits and next steps

This is fixed-threshold precision/recall, not mAP. Prediction-call timings exclude model loading, plotting and saving; no rigorous latency benchmark is claimed. Seven selected images are development evidence only. Untested generalization, annotation scope, possible near-duplicate frames and pretrained-data overlap remain limitations. Keep a later test split untouched and use a standard evaluator for mAP.

Dependencies for notebook tools are bounded, not a complete lock. Record resolved versions after installation. Do not silently replace the validated CPU environment snapshot. Add W&B only after the local run succeeds; the JSON/CSV outputs already provide traceable local experiment records.

Official inference API: https://docs.ultralytics.com/modes/predict/
Dataset: https://github.com/skaczylo/Football-Image-Dataset
