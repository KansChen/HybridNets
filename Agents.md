# HybridNets – Codex Agent Guide

This `agents.md` is for AI agents (like Codex) aiding developers or reviewers unfamiliar with HybridNets. It explains where to find things and how technical parts are implemented.

---

## 🗂️ Project Structure Overview

Root-level files:

- **`backbone.py`** – defines the CNN backbone (e.g. EfficientNet) and anchor configurations.
- **`train.py` / `train_ddp.py`** – training pipelines: single-GPU and DistributedDataParallel (DDP).
- **`val.py` / `val_ddp.py`** – validation scripts (single and multi-GPU).
- **`hybridnets_test.py`** / **`hybridnets_test_videos.py`** – inference scripts for images and videos.
- **`export.py`** – exports ONNX model + `.npy` anchors.
- **`hubconf.py`** – PyTorch Hub entrypoint.
  
Folders:

- **`encoders/`** – third-party backbones (from segmentation_models.pytorch).
- **`hybridnets/`**  
  - `model.py` – detection + segmentation heads, BiFPN, multi-task fusion.  
  - `autoanchor.py` – auto-generated anchors via k‑means.  
  - `dataset.py` – BDD100K data loader and augmentations.  
  - `loss.py` – losses: focal, Tversky/Dice for segmentation; SIoU for detection.  
- **`projects/`** – e.g. `bdd100k.yml` defines anchor, path, normalization, etc.
- **`utils/`** – helpers: `constants.py`, `plot.py`, `smp_metrics.py`, shared preprocessing & postprocessing.
- **`ros/`** – ROS C++ integration for planning/detection.
- **`demo/`, `tutorial/`** – usage examples and notebooks.

---

## 🚀 Technical Workflow for Agent Users

### 1. Backbone + Feature Pyramid

- `backbone.py` initializes EfficientNet and BiFPN layers.
- `autoanchor.py` computes dataset-specific anchors with k-means (this is called in training startup).

### 2. Multi-task Heads

- In `model.py`, detect head (box + class), segmentation head (drivable + lane) branch from BiFPN.
- Agent should inspect the `forward()` to understand tensor shapes & splits.

### 3. Loss Functions

- Detection: SIoU for boxes + BCE for confidence/class.
- Segmentation: Binary BCE + Tversky/Dice.
- Code for multiplication and balancing is in `loss.py`.

### 4. Training & Validation

- `train.py` handles loading dataset, optimizer, scheduler.
- Supports freezing backbone or heads via CLI args.
- `val.py` loads checkpoint, runs metrics, visualizes output with `plot.py`.

### 5. Inference & Export

- `hybridnets_test*.py` shows how to pre-/postprocess and overlay outputs.
- `export.py` exports ONNX including anchors, enabling C++/ROS integration.
- `hubconf.py` supports loading via `torch.hub`.

### 6. ROS Integration

- C++ modules in `ros/` subscribe to topics, run inference via exported model, and handle planning.

---

## 🧪 Code Generation & API Expectations

When generating new code:

- **Type hints** in Python 3.7+ are required.
- Functions should include docstrings.
- API must be consistent: e.g. `train()`, `validate()`, `infer_image()`, `infer_video()`.
- CLI should support flags like `--freeze-backbone`, `--multi-task-weight`, `--anchors-recalc`.

---

## ✅ Pull Request & CI Requirements

1. **Implementation clarity**: new modules should include tests & docstrings.
2. **Demo update**: if API/UI changed, update `demo/` and `tutorial/`.

---

## 👨‍💻 Agent Q&A Examples

**Q**: *How are anchors derived?*  
**A**: Agent inspects `autoanchor.py`, which k‑means clusters BDD100K bounding boxes and writes `.npy` anchors for each pyramid level. Training reads these defaults unless overridden via `--anchors`.

**Q**: *What architecture is used for segmentation?*  
**A**: In `model.py`, a U-Net-like decoder with BiFPN cross-feature fusion outputs two masks (drivable & lane), using Tversky/BCE losses from `loss.py`.

**Q**: *How to switch to distributed training?*  
**A**: Use `train_ddp.py`; agent should auto-generate the `torch.distributed.launch` call and wrap model/scheduler into `DistributedDataParallel`.

**Q**: *Can I replace the backbone?*  
**A**: Yes — choose a new encoder from `encoders/`, import it, and adjust `backbone.py`. You likely also need to run `autoanchor.py` to regenerate anchors.

---

## 📚 References & Performance

HybridNets achieves SOTA on BDD100K for detection (77.3 mAP), drivable-area segmentation (~90.5 mIoU), and lane detection (~31.6 mIoU) at ~12M params + 15.6 GFLOPs :contentReference[oaicite:1]{index=1}. The paper details BiFPN, SIoU loss, and anchor auto-generation :contentReference[oaicite:2]{index=2}.
