# 3DEKD
This repository provides the official implementation of the paper:

> **3DEKD: 3D Explanation-based Knowledge Distillation for Pillar-based 3D Object Detection**  
> Accepted to *IEEE Signal Processing Letters (SPL), 2025*

---

## 🔍 Overview

3DEKD is a novel explanation-based knowledge distillation method designed to improve both **interpretability** and **accuracy** in LiDAR-based 3D object detection.  
It integrates 3D attribution maps into the distillation process to help the student model focus on meaningful regions in sparse point cloud data.

<p align="center">
  <img src="docs/figure 1.png" alt="3DEKD Architecture" width="700"/>
</p>

---

## 📌 Key Features

- 💡 Attribution map generation via [OccAM's Laser](https://arxiv.org/abs/2203.14335)
- 🧠 Pillar-wise feature selection for sparse region refinement
- 🔄 Pillar-aligned explanation transfer using cosine similarity loss
- 📈 Up to **+2.09%** 3D mAP and **+0.84%** BEV mAP improvement on KITTI

---

## 🗂️ Folder Structure

3DEKD/
├── cfgs/ # YAML config files for experiments
├── datasets/ # Dataset processing scripts
├── models/ # Model architecture + distillation logic
├── tools/ # Training, evaluation, and visualization scripts
├── utils/ # Helper functions
└── docs/ # Figures and docs (e.g., paper figure, attribution examples)
