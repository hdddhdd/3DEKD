import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import os

# Input 및 Output 폴더 경로 설정
pcl_folder = '../data/nuscenes/v1.0-trainval/pcl_ppillar_nusc/train/'
attribution_folder = '../data/nuscenes/v1.0-trainval/attrmap_ppillar_nusc/train/'
output_folder = './ply_outputs/'

# 출력 폴더가 없다면 생성
os.makedirs(output_folder, exist_ok=True)

# 폴더 내의 모든 npy 파일 목록을 불러옴
pcl_files = [f for f in os.listdir(pcl_folder) if f.endswith('.npy')]

for pcl_file in pcl_files:
    pcl_filename = os.path.join(pcl_folder, pcl_file)
    attribution_filename = os.path.join(attribution_folder, pcl_file.replace("pcl.npy", "attr.npy"))
    output_ply_filename = os.path.join(output_folder, pcl_file.replace(".npy", ".ply"))

    # Attribution map을 불러옴
    attr_map = np.load(attribution_filename)

    # 객체별 정규화 수행
    normalized_attr_map = np.zeros_like(attr_map)
    for i in range(attr_map.shape[0]):
        min_val = attr_map[i].min()
        max_val = attr_map[i].max()
        if max_val > min_val:
            normalized_attr_map[i] = (attr_map[i] - min_val) / (max_val - min_val)
        else:
            normalized_attr_map[i] = attr_map[i]

    # 각 포인트의 최대 중요도 값 계산
    max_importance_normalized = np.max(normalized_attr_map, axis=0)

    # PCL 파일 불러오기
    pcl_data = np.load(pcl_filename)

    # Color map 생성
    cmap = plt.get_cmap('jet')
    colors = cmap(max_importance_normalized)

    # PLY 파일로 저장
    vertex = np.array([(pcl_data[i, 0], pcl_data[i, 1], pcl_data[i, 2], *colors[i, :3]*255)
                        for i in range(len(pcl_data))],
                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(output_ply_filename)

    print(f'Saved: {output_ply_filename}')