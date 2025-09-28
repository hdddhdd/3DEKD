#!/bin/bash

set -e  # 에러 발생 시 즉시 중단

# echo "=============================="
# echo "🚀 첫 번째 실험: train_occam.py (4x)"
# echo "=============================="
# python train_occam.py --cfg_file cfgs/kd/kd_pointpillar_4x.yaml --extra_tag MM_ppillar4x_nusc_top10 --epochs 100 --batch_size 8
# # python train_occam.py --cfg_file cfgs/kd/kd_pointpillar_4x.yaml --extra_tag MM_ppillar4x_nusc_top50 --epochs 1 --batch_size 8
# # python test.py --cfg_file cfgs/nuscenes_models/pointpillar_4x.yaml --batch_size 8 --workers 4 --ckpt ../output/kd/kd_pointpillar_4x/MM_ppillar4x_nusc_top10/ckpt/checkpoint_epoch_30.pth 

# echo "=============================="
# echo "📦 두 번째 실험: train.py (Student only, 4x)"
# echo "=============================="
# python train.py --cfg_file cfgs/nuscenes_models/pointpillar_4x.yaml --epochs 100 --batch_size 8 --workers 4
# # # python train.py --cfg_file cfgs/nuscenes_models/pointpillar_4x.yaml --epochs 32 --batch_size 8 --workers 4


# echo "=============================="
# echo "🔥 세 번째 실험: train_fitnet.py (FitNet 4x)"
# echo "=============================="
# python train_fitnet.py --cfg_file cfgs/kd/kd_pointpillar_4x.yaml --epochs 100 --extra_tag fitnet_pointpillar_4x --batch_size 8 --workers 4 --max_waiting_mins 0
# # python train_fitnet.py --cfg_file cfgs/kd/kd_pointpillar_4x.yaml --epochs 32 --extra_tag fitnet_pointpillar_4x --batch_size 8 --workers 4 --max_waiting_mins 0



# # echo "=============================="
# # echo "📦 추가 실험: train_occam.py (Student only, 4x)"
# # echo "=============================="
# # # python train_occam.py --cfg_file cfgs/kd/kd_pointpillar_4x.yaml --extra_tag MM_ppillar4x_nusc_top75 --epochs 60 --batch_size 8
# # python train_occam.py --cfg_file cfgs/kd/kd_pointpillar_4x.yaml --extra_tag MM_ppillar4x_nusc_top25 --epochs 30 --batch_size 8


# echo "=============================="
# echo "📦 네 번째 실험: train_crd.py (Student only, 4x)"
# echo "=============================="
# python train_crd.py --cfg_file cfgs/kd/kd_pointpillar_4x.yaml --epochs 100 --extra_tag crd_ppillar4x --batch_size 8 --workers 4





echo "=============================="
echo " TESTING "
python test.py --cfg_file cfgs/nuscenes_models/pointpillar_4x.yaml --batch_size 8 --workers 4 --ckpt ../output/kd/kd_pointpillar_4x/MM_ppillar4x_nusc_top10/ckpt/checkpoint_epoch_100.pth 
python test.py --cfg_file cfgs/nuscenes_models/pointpillar_4x.yaml --ckpt ../output/kd/kd_pointpillar_4x/fitnet_pointpillar_4x/ckpt/checkpoint_epoch_100.pth 
python test.py --cfg_file cfgs/nuscenes_models/pointpillar_4x.yaml --ckpt ../output/nuscenes_models/pointpillar_4x/default/ckpt/checkpoint_epoch_100.pth
python test.py --cfg_file cfgs/nuscenes_models/pointpillar_4x.yaml --batch_size 8 --workers 4 --ckpt ../output/kd/kd_pointpillar_4x/crd_ppillar4x/ckpt/checkpoint_epoch_100.pth
# python test.py --cfg_file cfgs/nuscenes_models/pointpillar_4x.yaml --ckpt ../output/kd/kd_pointpillar_4x/fitnet_pointpillar_4x/ckpt/checkpoint_epoch_30.pth 



echo "✅ 모든 실험이 완료되었습니다."
