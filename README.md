# 3DEKD: 3D Explanation-based Knowledge Distillation for Pillar-based 3D Object Detection
This repository provides the official implementation of the paper:

> **3DEKD: 3D Explanation-based Knowledge Distillation for Pillar-based 3D Object Detection**  
> Accepted to *IEEE Signal Processing Letters (SPL), 2025*




## Train
```
python occam_demo_pointpillar.py --model_cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../ckpt/pointpillar_7728.pth --imagesets_path ../data/kitti/ImageSets/ --dataset_path ../data/kitti/ --nr_it 3000
```
