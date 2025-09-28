import argparse
import os
import numpy as np
import time

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from occam_utils.occam import OccAM

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar.yaml')
    parser.add_argument('--occam_cfg_file', type=str, default='cfgs/occam_configs/kitti_ppillar.yaml')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--nr_it', type=int, default=3000)
    parser.add_argument('--dataset_path', type=str, default='../data/kitti/')
    parser.add_argument('--imagesets_path', type=str, default='../data/kitti/ImageSets/')
    parser.add_argument('--attrmap_save_dir', type=str, default='attrmap_ppillar_kitti')
    parser.add_argument('--pcl_save_dir', type=str, default='pcl_ppillar_kitti')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg_from_yaml_file(args.occam_cfg_file, cfg)

    return args, cfg

def load_file_list(imagesets_path, filename):
    file_path = os.path.join(imagesets_path, filename)
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def main():
    args, config = parse_config()
    logger = common_utils.create_logger()
    logger.info('------------------------ OccAM Demo with Timing -------------------------')

    occam = OccAM(data_config=config.DATA_CONFIG, model_config=config.MODEL,
                  occam_config=config.OCCAM, class_names=config.CLASS_NAMES,
                  model_ckpt_path=args.ckpt, nr_it=args.nr_it, logger=logger)

    train_files = load_file_list(args.imagesets_path, 'occam_vis.txt')
    processing_times = []
    total_start_time = time.time()

    for idx, file_name in enumerate(train_files):
        source_file_path = os.path.join(args.dataset_path, 'training', 'velodyne', file_name + '.bin')
        if not os.path.exists(source_file_path):
            logger.warning(f'File {source_file_path} does not exist, skipping.')
            continue

        logger.info(f'Processing file {idx + 1}/{len(train_files)}: {file_name}')

        start_time = time.time()

        pcl = occam.load_and_preprocess_pcl(source_file_path)
        xy_mask = ~((np.abs(pcl[:, 0]) < 2) & (np.abs(pcl[:, 1]) < 2))
        pcl = pcl[xy_mask]

        base_det = occam.get_base_predictions(pcl=pcl)
        base_det_boxes, base_det_labels, base_det_scores = base_det

        attr_maps, result_mask = occam.compute_attribution_maps(
            pcl=pcl, base_det_boxes=base_det_boxes,
            base_det_labels=base_det_labels, batch_size=args.batch_size,
            num_workers=args.workers)

        attr_map_save_path = os.path.join(args.dataset_path, args.attrmap_save_dir, 'train', file_name + '_attr.npy')
        pcl_save_path = os.path.join(args.dataset_path, args.pcl_save_dir, 'train', file_name + '_pcl.npy')

        os.makedirs(os.path.dirname(attr_map_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(pcl_save_path), exist_ok=True)

        np.save(attr_map_save_path, attr_maps)
        np.save(pcl_save_path, pcl)

        logger.info(f"Saved attribution map: {attr_map_save_path}")
        logger.info(f"Saved PCL: {pcl_save_path}")

        proc_time = time.time() - start_time
        processing_times.append(proc_time)
        logger.info(f"File processing time: {proc_time:.2f} seconds")

        if (idx + 1) % 10 == 0 or (idx + 1) == len(train_files):
            avg_time_so_far = np.mean(processing_times)
            logger.info(f"Progress: {idx + 1}/{len(train_files)} files processed. Average time so far: {avg_time_so_far:.2f} s")

    total_time = time.time() - total_start_time
    if processing_times:
        stats_file = os.path.join(args.dataset_path, args.attrmap_save_dir, 'processing_time_stats.txt')
        with open(stats_file, 'w') as f:
            f.write("OccAM Processing Time Statistics\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total files: {len(processing_times)}\n")
            f.write(f"Total time: {total_time:.2f}s ({total_time / 60:.1f}min)\n")
            f.write(f"Avg time per file: {np.mean(processing_times):.2f}s\n")
            f.write(f"Std dev: {np.std(processing_times):.2f}s\n")
            f.write(f"Min time: {np.min(processing_times):.2f}s\n")
            f.write(f"Max time: {np.max(processing_times):.2f}s\n")
            f.write(f"Est. time for 1000 files: {np.mean(processing_times) * 1000 / 60:.1f}min\n")
            f.write("\nIndividual file times:\n")
            for i, t in enumerate(processing_times):
                f.write(f"{i+1:4d}. {train_files[i]}: {t:.2f}s\n")

        logger.info(f"Processing time statistics saved to: {stats_file}")

    logger.info('DONE')

if __name__ == '__main__':
    main()
