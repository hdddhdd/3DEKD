import argparse
import numpy as np
import torch
from pathlib import Path
from thop import profile
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.datasets import KittiDataset
def move_to_device(sample_data, device):
    # sample_data는 딕셔너리이므로 그 안에 있는 텐서들을 모두 device로 이동
    for key in sample_data:
        if isinstance(sample_data[key], torch.Tensor):  # 텐서일 경우에만 이동
            sample_data[key] = sample_data[key].to(device)
    return sample_data


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/centerpointpillar.yaml',
                        help='specify the config for demo')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()

    # 데이터셋 객체 생성
    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH),
        training=False
    )

    # 모델 구성 및 빌드
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.eval()
    
    # 모델을 GPU로 이동 (사용 가능한 경우)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 데이터셋에서 샘플 데이터 로드 (val쪽에서 가져옴.)
    sample_data = dataset[0]  # 첫 번째 샘플 데이터 로드, 실제 사용 시에는 반복문 안에서 처리
    # sample_data = move_to_device(sample_data, device)
    print('sample_data keys:', sample_data.keys())
    print('sample_data example:', sample_data)
    
    print('sample_data: ', sample_data)
    sample_data = dataset.collate_batch([sample_data])  # 배치 데이터로 변환

    # GPU에 데이터 로드
    for key, value in sample_data.items():
        if isinstance(value, torch.Tensor):
            sample_data[key] = value.to(device)


    # FLOPS와 파라미터 계산
    flops, params = profile(model, inputs=(sample_data,), verbose=False)
    logger.info(f"Computed FLOPS: {flops / 1e9:.2f} GFLOPS")
    logger.info(f"Total Parameters: {params / 1e6:.2f} M")

if __name__ == '__main__':
    main()
