import numpy as np
import argparse
import os
import glob

def check_single_file(file_path):
    """단일 npy 파일의 정보 출력"""
    try:
        data = np.load(file_path)
        print(f"File: {file_path}")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Size: {data.size} elements")
        
        if data.size > 0:
            print(f"  Min: {data.min()}")
            print(f"  Max: {data.max()}")
            print(f"  Mean: {data.mean():.6f}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Check shape and info of npy files')
    parser.add_argument('path', type=str, 
                       help='Path to npy file or directory containing npy files')
    parser.add_argument('--pattern', type=str, default='*.npy',
                       help='File pattern for directory search (default: *.npy)')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        # 단일 파일 처리
        check_single_file(args.path)
    elif os.path.isdir(args.path):
        # 디렉토리 내 모든 npy 파일 처리
        npy_files = sorted(glob.glob(os.path.join(args.path, args.pattern)))
        
        if not npy_files:
            print(f"No npy files found in {args.path} with pattern {args.pattern}")
            return
        
        print(f"Found {len(npy_files)} npy files:")
        print("=" * 50)
        
        for file_path in npy_files:
            check_single_file(file_path)
    else:
        print(f"Path does not exist: {args.path}")

if __name__ == '__main__':
    # 명령행 인자 없이 직접 실행하는 경우의 예시
    if len(os.sys.argv) == 1:
        print("Usage examples:")
        print("python check_npy.py file.npy")
        print("python check_npy.py directory/")
        print("python check_npy.py directory/ --pattern '*_scene.npy'")
        print("python check_npy.py directory/ --pattern '*_points.npy'")
        print()
        print("Or use directly in Python:")
        print("import numpy as np")
        print("data = np.load('file.npy')")
        print("print(data.shape)")
    else:
        main()