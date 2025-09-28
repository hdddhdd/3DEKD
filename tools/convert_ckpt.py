import torch

# 기존 MMDetection3D 체크포인트 로드
ckpt = torch.load('../ckpt/teacher_pointpillar_kitti.pth')

# 변환: OpenPCDet이 요구하는 구조로 저장
converted_ckpt = {'model_state': ckpt['state_dict']}
torch.save(converted_ckpt, '../ckpt/teacher_pointpillar_kitti_openpcdet.pth')
