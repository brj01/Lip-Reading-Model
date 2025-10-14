# Lip-Reading-Model

This repository is organized into three parts to streamline the lip-reading data pipeline: extraction, cleaning, and annotations.

**Repository Structure**
- `PART 1/` — Video Extraction
  - `split_clip_from_anno.py` — Crop per-clip regions from full videos using annotation JSONs; saves mp4 + 16 kHz wav.
  - `auto_single_scene.py` — Scene split + A/B diarization; performs segment cropping/export.
  - `auto_process_batch_scene.py` — Batch launcher for `auto_single_scene.py` with per-job logs.
  - `auto_single_yolo.py` — YOLO + SyncNet + diarization variant for segmentation; cropping/export utilities.
  - `auto_process_batch_yolo.py` — Batch launcher for `auto_single_yolo.py`.

- `PART 2/` — Video Cleaning
  - `filter_light.py` — Detects over-bright/dark sequences via luminance thresholds; multiprocessing.
  - `para_fliter.py` — Parallel launcher for `filter_light.py` partitions.
  - `refine_ID.py` — DeepFace (ArcFace + yolov8) cross-verification to catch A/B identity swaps.
  - `auto_reID_batch.py` — Batch launcher for `refine_ID.py`.

- `PART 3/Annotations/` — Annotations
  - `traversal_clips.py` — Multithreaded lister to generate `.mp4`/`.wav` path text files.
  - `auto_single_asr.py` — Whisper transcription to JSON (sharded by index/total).
  - `auto_asr_batch.py` — Batch launcher for local ASR workers (expects `process_local_asr.py`).
  - `auto_single_dwpose.py` — DWpose (ONNX det+pose) keypoint extraction to `.npy` per video.
  - `auto_dwpose_batch.py` — Batch launcher for DWpose.
  - `auto_collect_scene.py` — QC helper to list mp4s missing scene JSON files.
  - `dwpose/` — ONNX inference and utilities for DWpose (detector, pose, drawing).

**Important Notes**
- Paths and weights:
  - Many scripts reference absolute paths like `/data/...` and weights via relative paths such as `../weights`. Adjust these for your environment.
  - If running a script from a different working directory, batch launchers have been updated to resolve their companion scripts using `__file__` to avoid path issues.
- Dependencies (non-exhaustive): `ffmpeg`, `opencv-python`, `deepface`, `onnxruntime`, `ultralytics`, `whisper`, `pydub`, `torchaudio`, `av`, `scenedetect`, `tqdm`, `numpy`.
- OS considerations:
  - Some batch scripts include `nohup`/Linux-style commands. On Windows, launch without `nohup` or adapt to PowerShell background jobs if needed.

**Quick Start (Examples)**
- List all mp4s under a root:
  - `python "PART 3/Annotations/traversal_clips.py"`
- Crop clips from annotations:
  - `python "PART 1/split_clip_from_anno.py"`
- Detect lighting issues:
  - `python "PART 2/filter_light.py" --video_list_path <txt> --save_dir <out>`
- Refine A/B identities with DeepFace:
  - `python "PART 2/refine_ID.py" --video_list_path <txt> --output <out_dir>`
- Extract DWpose keypoints:
  - `python "PART 3/Annotations/auto_single_dwpose.py" --video_list_path <txt> --save_dir <out>`

If you want me to add per-script usage docs or align default paths to your machine, let me know the target directories for videos, annotations, and weights.
