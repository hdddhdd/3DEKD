
echo "=============================="
echo " TESTING "
python test.py --cfg_file cfgs/nuscenes_models/pointpillar_4x.yaml --batch_size 8 --workers 4 --ckpt ../output/kd/kd_pointpillar_4x/MM_ppillar4x_nusc_top25/ckpt/checkpoint_epoch_30.pth 
python test.py --cfg_file cfgs/nuscenes_models/pointpillar_4x.yaml --batch_size 8 --workers 4 --ckpt ../output/kd/kd_pointpillar_4x/MM_ppillar4x_nusc_top50/ckpt/checkpoint_epoch_30.pth 
python test.py --cfg_file cfgs/nuscenes_models/pointpillar_4x.yaml --batch_size 8 --workers 4 --ckpt ../output/kd/kd_pointpillar_4x/MM_ppillar4x_nusc_top75/ckpt/checkpoint_epoch_30.pth 
python test.py --cfg_file cfgs/nuscenes_models/pointpillar_4x.yaml --ckpt ../output/kd/kd_pointpillar_4x/fitnet_pointpillar_4x/ckpt/checkpoint_epoch_31.pth 
python test.py --cfg_file cfgs/nuscenes_models/pointpillar_4x.yaml --ckpt ../output/nuscenes_models/pointpillar_4x/default/ckpt/checkpoint_epoch_31.pth

# python test.py --cfg_file cfgs/nuscenes_models/pointpillar_4x.yaml --ckpt ../output/kd/kd_pointpillar_4x/fitnet_pointpillar_4x/ckpt/checkpoint_epoch_30.pth 

