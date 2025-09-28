import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse
import os
import glob

'''
python script.py --attr_dir ../data/data/ATTRfortimecheck/train --pcl_dir ../data/data/PCLfortimecheck/train

'''

def apply_turbo_colormap(values):
    """
    Attribution 값에 turbo colormap을 적용하여 RGB 색상 반환
    
    Args:
        values: attribution 값 배열 (1D numpy array), 원본 값 그대로
    
    Returns:
        colors: RGB 색상 배열 (N, 3), 값은 0-255 범위
    """
    # matplotlib의 turbo colormap 사용
    cmap = plt.cm.turbo
    
    # 원본 값 그대로 사용 (정규화 없이)
    normalized_values = values
    
    # colormap 적용
    colors = cmap(normalized_values)[:, :3]  # RGB만 사용 (alpha 제외)
    
    # 0-255 범위로 변환
    colors_255 = (colors * 255).astype(np.uint8)
    
    return colors_255

def save_ply(points, colors, filename):
    """
    Point cloud와 색상을 PLY 파일로 저장
    
    Args:
        points: 3D 좌표 배열 (N, 3) - x, y, z만 사용
        colors: RGB 색상 배열 (N, 3), 0-255 범위
        filename: 저장할 파일명
    """
    n_points = points.shape[0]
    
    with open(filename, 'w') as f:
        # PLY 헤더 작성
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 점 데이터 작성
        for i in range(n_points):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                   f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")

def process_single_file(attr_path, pcl_path, output_path):
    """
    단일 attribution map과 point cloud 파일을 처리하여 PLY로 저장
    
    Args:
        attr_path: attribution map npy 파일 경로 (shape: n_objects, n_points)
        pcl_path: point cloud npy 파일 경로 (shape: n_points, 4)
        output_path: 출력 PLY 파일 경로
    """
    print(f"Loading attribution map: {attr_path}")
    attribution_map = np.load(attr_path)  # shape: (n_objects, n_points)
    
    print(f"Loading point cloud: {pcl_path}")
    points_with_intensity = np.load(pcl_path)  # shape: (n_points, 4) - x,y,z,intensity
    
    print(f"Attribution map shape: {attribution_map.shape}")
    print(f"Point cloud shape: {points_with_intensity.shape}")
    
    # Attribution map이 2D인지 확인
    if attribution_map.ndim != 2:
        raise ValueError(f"Attribution map should be 2D, got shape {attribution_map.shape}")
    
    n_objects, n_points = attribution_map.shape
    
    # Point cloud에서 x, y, z 좌표만 추출
    points_xyz = points_with_intensity[:, :3]  # shape: (n_points, 3)
    
    # 크기 체크
    if n_points != points_xyz.shape[0]:
        raise ValueError(f"Attribution map points ({n_points}) and "
                        f"point cloud points ({points_xyz.shape[0]}) don't match")
    
    print(f"Number of objects: {n_objects}")
    print(f"Number of points: {n_points}")
    
    # 각 포인트에 대해 모든 객체 중 최대 attribution 값 계산
    print("Computing max attribution values across objects...")
    scene_attribution = np.max(attribution_map, axis=0)  # shape: (n_points,)
    
    print(f"Scene attribution shape: {scene_attribution.shape}")
    print(f"Scene attribution range: {scene_attribution.min():.6f} ~ {scene_attribution.max():.6f}")
    
    # Turbo colormap 적용 (정규화 후)
    print("Applying turbo colormap...")
    colors = apply_turbo_colormap(scene_attribution)
    
    # PLY 파일 저장
    print(f"Saving PLY file: {output_path}")
    save_ply(points_xyz, colors, output_path)
    
    print(f"Successfully saved PLY file with {n_points} points")
    
    return scene_attribution

def main():
    parser = argparse.ArgumentParser(description='Convert multi-object attribution map and point cloud to PLY')
    parser.add_argument('--attr_dir', type=str, required=True,
                       help='Directory containing attribution map npy files')
    parser.add_argument('--pcl_dir', type=str, required=True,
                       help='Directory containing point cloud npy files')
    parser.add_argument('--output_dir', type=str, default='ply_output',
                       help='Output directory for PLY files')
    parser.add_argument('--attr_pattern', type=str, default='*_attr.npy',
                       help='Pattern for attribution map files')
    parser.add_argument('--pcl_pattern', type=str, default='*_pcl.npy',
                       help='Pattern for point cloud files')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Attribution map 파일들 찾기
    attr_files = sorted(glob.glob(os.path.join(args.attr_dir, args.attr_pattern)))
    
    if not attr_files:
        print(f"No attribution files found in {args.attr_dir} with pattern {args.attr_pattern}")
        return
    
    print(f"Found {len(attr_files)} attribution files")
    
    processed = 0
    for attr_path in attr_files:
        try:
            # 파일명에서 기본 이름 추출 (예: 000036_attr.npy -> 000036)
            attr_filename = os.path.basename(attr_path)
            base_name = attr_filename.replace('_attr.npy', '')
            
            # 대응하는 point cloud 파일 찾기
            pcl_filename = f"{base_name}_pcl.npy"
            pcl_path = os.path.join(args.pcl_dir, pcl_filename)
            
            if not os.path.exists(pcl_path):
                print(f"Warning: Point cloud file not found: {pcl_path}")
                continue
            
            # 출력 파일명
            output_filename = f"{base_name}_attribution.ply"
            output_path = os.path.join(args.output_dir, output_filename)
            
            # 처리
            scene_attribution = process_single_file(attr_path, pcl_path, output_path)
            processed += 1
            
            print("-" * 60)
            
        except Exception as e:
            print(f"Error processing {attr_path}: {str(e)}")
            continue
    
    print(f"Successfully processed {processed}/{len(attr_files)} files")

if __name__ == '__main__':
    # 사용 예시 (명령행 인자 없이 직접 실행하는 경우)
    if len(os.sys.argv) == 1:
        print("Usage examples:")
        print("python script.py --attr_dir ../data/data/ATTRfortimecheck/train --pcl_dir ../data/data/PCLfortimecheck/train")
        print("python script.py --attr_dir attr_folder --pcl_dir pcl_folder --output_dir my_ply_files")
        print("python script.py --attr_dir attr_folder --pcl_dir pcl_folder --attr_pattern '*_attribution.npy' --pcl_pattern '*_points.npy'")
        print()
        print("Direct usage in Python:")
        print("from script import process_single_file")
        print("process_single_file('000036_attr.npy', '000036_pcl.npy', '000036_attribution.ply')")
    else:
        main()