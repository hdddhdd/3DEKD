# kitti로 변환필요
# python occam_demo.py \
#     --cfg_file cfgs/kitti_models/pointpillar.yaml \
#     --occam_cfg_file cfgs/occam_configs/kitti_pointpillar.yaml \
#     --ckpt path/to/pretrained/model.pth \
#     --dataset_path path/to/dataset \
#     --imagesets_path path/to/ImageSets \
#     --batch_size 8 \
#     --workers 4
import argparse
import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 1번 GPU 사용

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from occam_utils.occam import OccAM

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str,
                        default='cfgs/kitti_models/pointpillar.yaml',
                        help='dataset/model config for the demo')
    parser.add_argument('--occam_cfg_file', type=str,
                        default='cfgs/occam_configs/kitti_ppillar.yaml',
                        help='specify the OccAM config')
    parser.add_argument('--ckpt', type=str, default=None, required=True,
                        help='path to pretrained model parameters')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for OccAM creation')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers for dataloader')
    parser.add_argument('--nr_it', type=int, default=3000,
                        help='number of sub-sampling iterations N')
    parser.add_argument('--dataset_path', type=str, default='../data/kitti/',
                        help='Path to the dataset containing training data')
    parser.add_argument('--imagesets_path', type=str, default='../data/kitti/ImageSets/',
                        help='Path to the ImageSets folder containing train.txt')
    parser.add_argument('--attrmap_save_dir', type=str, default='attrmap_ppillar_kitti',
                        help='Directory to save attribution maps')
    parser.add_argument('--pcl_save_dir', type=str, default='pcl_ppillar_kitti',
                        help='Directory to save preprocessed point clouds')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg_from_yaml_file(args.occam_cfg_file, cfg)

    return args, cfg

def load_file_list(imagesets_path, filename):
    file_path = os.path.join(imagesets_path, filename)
    with open(file_path, 'r') as f:
        file_list = f.readlines()
    file_list = [line.strip() for line in file_list]
    return file_list

def main():
    args, config = parse_config()
    logger = common_utils.create_logger()
    logger.info('------------------------ OccAM Demo -------------------------')

    occam = OccAM(data_config=config.DATA_CONFIG, model_config=config.MODEL,
                  occam_config=config.OCCAM, class_names=config.CLASS_NAMES,
                  model_ckpt_path=args.ckpt, nr_it=args.nr_it, logger=logger)

    # Load train and val files
    train_files = load_file_list(args.imagesets_path, 'train.txt')

    for file_type, file_list in [('train', train_files)]:
        logger.info(f'Number of {file_type} files to analyze: {len(file_list)}')

        for idx, file_name in enumerate(file_list):
            source_file_path = os.path.join(args.dataset_path, 'training','velodyne', file_name + '.bin')
            if not os.path.exists(source_file_path):
                logger.warning(f'File {source_file_path} does not exist, skipping.')
                continue

            logger.info(f'Processing {file_type} file {idx + 1}/{len(file_list)}: {file_name}')
            
            # Pre-sampling shape
            pcl = occam.load_and_preprocess_pcl(source_file_path)
            logger.info(f'Original point cloud shape: {pcl.shape}')

            # Ego vehicle 제거: x, y가 -1~1 사이인 포인트 제거
            xy_mask = ~((np.abs(pcl[:, 0]) < 2) & (np.abs(pcl[:, 1]) < 2))
            pcl = pcl[xy_mask]
            logger.info(f'pcl after ego vehicle removal: {pcl.shape}')

            logger.info(f'Original point cloud shape: {pcl.shape}')

            # Get detections to analyze (in full pcl)
            base_det = occam.get_base_predictions(pcl=pcl)
            base_det_boxes, base_det_labels, base_det_scores = base_det

            logger.info('Number of detected objects to analyze: ' + str(base_det_labels.shape[0]))

            
            unique_labels = np.unique(base_det_labels)
            # print('unique_labels: ', unique_labels)
            # print('CLASS_NAMES:', config.CLASS_NAMES)
            # print('max label index:', max(unique_labels))
            # print('unique_labels:', unique_labels)

            # 보정된 클래스 매핑
            detected_classes = []
            for lbl in unique_labels:
                corrected_lbl = int(lbl) - 1
                if 0 <= corrected_lbl < len(config.CLASS_NAMES):
                    detected_classes.append(config.CLASS_NAMES[corrected_lbl])
                else:
                    logger.warning(f'⚠️ Unknown label index {lbl} (corrected: {corrected_lbl}) detected — skipping')

            logger.info('Detected classes: ' + ', '.join(detected_classes))



            logger.info('Start attribution map computation:')
            attr_maps, result_mask = occam.compute_attribution_maps(
                pcl=pcl, base_det_boxes=base_det_boxes,
                base_det_labels=base_det_labels, batch_size=args.batch_size,
                num_workers=args.workers)
            print('!!!!!! attr_maps.shape: ', attr_maps.shape)
            # Post-sampling shapes
            logger.info(f'Attribution map shape: {attr_maps.shape}')
            logger.info(f'Result mask shape: {result_mask.shape}')

            # Save the computed attribution maps
            # attr_map_save_path = os.path.join(args.dataset_path, 'attrmap_ppillar_nusc', file_type, file_name + '_attr.npy')
            # pcl_save_path = os.path.join(args.dataset_path, 'pcl_ppillar_nusc', file_type, file_name + '_pcl.npy')
            attr_map_save_path = os.path.join(args.dataset_path, args.attrmap_save_dir, file_type, file_name + '_attr.npy')
            pcl_save_path = os.path.join(args.dataset_path, args.pcl_save_dir, file_type, file_name + '_pcl.npy')
            
            os.makedirs(os.path.dirname(attr_map_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(pcl_save_path), exist_ok=True)
            
            np.save(attr_map_save_path, attr_maps)
            np.save(pcl_save_path, pcl)

            logger.info(f'Saved attribution map for {file_type} file {file_name} to {attr_map_save_path}')
            logger.info(f'Saved pcl for {file_type} file {file_name} to {pcl_save_path}')

    logger.info('DONE')

if __name__ == '__main__':
    main()
