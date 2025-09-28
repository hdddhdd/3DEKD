import copy
import torch
import open3d
import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from torch.utils.data import DataLoader

from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from scipy.spatial.transform import Rotation

from occam_utils.occam_datasets import BaseDataset, OccamInferenceDataset
class OccAM(object):
    def __init__(self, data_config, model_config, occam_config, class_names,
                 model_ckpt_path, nr_it, logger):
        # 기존 초기화 코드...
        
        # 디버깅 관련 속성 추가
        self.debug_mode = False
        self.debug_data = {
            'similarities': [],
            'distances': [],
            'kept_ratios': [],
            'object_classes': [],
            'iteration_info': []
        }

        self.data_config = data_config
        self.model_config = model_config
        self.occam_config = occam_config
        self.class_names = class_names
        self.logger = logger
        self.nr_it = nr_it

        self.base_dataset = BaseDataset(data_config=self.data_config,
                                        class_names=self.class_names,
                                        occam_config=self.occam_config)

        self.model = build_network(model_cfg=self.model_config,
                                   num_class=len(self.class_names),
                                   dataset=self.base_dataset)
        self.model.load_params_from_file(filename=model_ckpt_path,
                                         logger=logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()

    def load_and_preprocess_pcl(self, source_file_path):
        """
        load given point cloud file and preprocess data according OpenPCDet
        data config using the base dataset

        Parameters
        ----------
        source_file_path : str
            path to point cloud to analyze (bin or npy)

        Returns
        -------
        pcl : ndarray (N, 4)
            preprocessed point cloud (x, y, z, intensity)
        """
        pcl = self.base_dataset.load_and_preprocess_pcl(source_file_path)
        return pcl

    def get_base_predictions(self, pcl):
        """
        get all K detections in full point cloud for which attribution maps will
        be determined

        Parameters
        ----------
        pcl : ndarray (N, 4)
            preprocessed point cloud (x, y, z, intensity)

        Returns
        -------
        base_det_boxes : ndarray (K, 7)
            bounding box parameters of detected objects
        base_det_labels : ndarray (K)
            labels of detected objects
        base_det_scores : ndarray (K)
            confidence scores for detected objects
        """
        input_dict = {
            'points': pcl
        }

        data_dict = self.base_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.base_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        with torch.no_grad():
            base_pred_dict, _ = self.model.forward(data_dict)

        base_det_boxes = base_pred_dict[0]['pred_boxes'].cpu().numpy()
        base_det_labels = base_pred_dict[0]['pred_labels'].cpu().numpy()
        base_det_scores = base_pred_dict[0]['pred_scores'].cpu().numpy()


        # 추가
        score_thresh = 0.1
        mask = base_det_scores > score_thresh
        base_det_boxes = base_det_boxes[mask]
        base_det_labels = base_det_labels[mask]
        base_det_scores = base_det_scores[mask]


        return base_det_boxes, base_det_labels, base_det_scores

    def merge_detections_in_batch(self, det_dicts):
        """
        In order to efficiently determine the confidence score for
        all detections in a batch they are merged.

        Parameters
        ----------
        det_dicts : list
            list of M dicts containing the detections in the M samples within
            the batch (pred boxes, pred scores, pred labels)

        Returns
        -------
        pert_det_boxes : ndarray (L, 7)
            bounding boxes of all L detections in the M samples
        pert_det_labels : ndarray (L)
            labels of all L detections in the M samples
        pert_det_scores : ndarray (L)
            scores of all L detections in the M samples
        batch_ids : ndarray (L)
            Mapping of the detections to the individual samples within the batch
        """
        batch_ids = []

        data_dict = defaultdict(list)
        for batch_id, cur_sample in enumerate(det_dicts):
            batch_ids.append(
                np.ones(cur_sample['pred_labels'].shape[0], dtype=int)
                * batch_id)

            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_ids = np.concatenate(batch_ids, axis=0)

        merged_dict = {}
        for key, val in data_dict.items():
            if key in ['pred_boxes', 'pred_scores', 'pred_labels']:
                merged_data = []
                for data in val:
                    data = data.cpu().numpy()
                    merged_data.append(data)
                merged_dict[key] = np.concatenate(merged_data, axis=0)

        pert_det_boxes = merged_dict['pred_boxes']
        pert_det_labels = merged_dict['pred_labels']
        pert_det_scores = merged_dict['pred_scores']
        return pert_det_boxes, pert_det_labels, pert_det_scores, batch_ids

    def compute_iou(self, base_boxes, pert_boxes):
        """
        3D IoU between base and perturbed detections
        """
        base_boxes = torch.from_numpy(base_boxes)
        pert_boxes = torch.from_numpy(pert_boxes)
        base_boxes, pert_boxes = base_boxes.cuda(), pert_boxes.cuda()
        iou = boxes_iou3d_gpu(base_boxes, pert_boxes)
        iou = iou.cpu().numpy()
        return iou

    def compute_translation_score(self, base_boxes, pert_boxes):
        """
        translation score (see paper for details)
        """
        translation_error = np.linalg.norm(
            base_boxes[:, :3][:, None, :] - pert_boxes[:, :3], axis=2)
        translation_score = 1 - translation_error
        translation_score[translation_score < 0] = 0
        return translation_score

    def compute_orientation_score(self, base_boxes, pert_boxes):
        """
        orientation score (see paper for details)
        """
        boxes_a = copy.deepcopy(base_boxes)
        boxes_b = copy.deepcopy(pert_boxes)

        boxes_a[:, 6] = boxes_a[:, 6] % (2 * math.pi)
        boxes_a[boxes_a[:, 6] > math.pi, 6] -= 2 * math.pi
        boxes_a[boxes_a[:, 6] < -math.pi, 6] += 2 * math.pi
        boxes_b[:, 6] = boxes_b[:, 6] % (2 * math.pi)
        boxes_b[boxes_b[:, 6] > math.pi, 6] -= 2 * math.pi
        boxes_b[boxes_b[:, 6] < -math.pi, 6] += 2 * math.pi
        orientation_error_ = np.abs(
            boxes_a[:, 6][:, None] - boxes_b[:, 6][None, :])
        orientation_error__ = 2 * math.pi - np.abs(
            boxes_a[:, 6][:, None] - boxes_b[:, 6][None, :])
        orientation_error = np.concatenate(
            (orientation_error_[:, :, None], orientation_error__[:, :, None]),
            axis=2)
        orientation_error = np.min(orientation_error, axis=2)
        orientation_score = 1 - orientation_error
        orientation_score[orientation_score < 0] = 0
        return orientation_score

    def compute_scale_score(self, base_boxes, pert_boxes):
        """
        scale score (see paper for details)
        """
        boxes_centered_a = copy.deepcopy(base_boxes)
        boxes_centered_b = copy.deepcopy(pert_boxes)
        boxes_centered_a[:, :3] = 0
        boxes_centered_a[:, 6] = 0
        boxes_centered_b[:, :3] = 0
        boxes_centered_b[:, 6] = 0
        scale_score = self.compute_iou(boxes_centered_a, boxes_centered_b)
        scale_score[scale_score < 0] = 0
        return scale_score

    def get_similarity_matrix(self, base_det_boxes, base_det_labels,
                              pert_det_boxes, pert_det_labels, pert_det_scores):
        """
        compute similarity score between the base detections in the full
        point cloud and the detections in the perturbed samples

        Parameters
        ----------
        base_det_boxes : (K, 7)
            bounding boxes of detected objects in full pcl
        base_det_labels : (K)
            class labels of detected objects in full pcl
        pert_det_boxes : ndarray (L, 7)
            bounding boxes of all L detections in the perturbed samples of the batch
        pert_det_labels : ndarray (L)
            labels of all L detections in the perturbed samples of the batch
        pert_det_scores : ndarray (L)
            scores of all L detections in the perturbed samples of the batch
        Returns
        -------
        sim_scores : ndarray (K, L)
            similarity score between all K detections in the full pcl and
            the L detections in the perturbed samples within the batch
        """
        # similarity score is only greater zero if boxes overlap
        s_overlap = self.compute_iou(base_det_boxes, pert_det_boxes) > 0
        s_overlap = s_overlap.astype(np.float32)

        # similarity score is only greater zero for boxes of same class
        s_class = base_det_labels[:, None] == pert_det_labels[None, :]
        s_class = s_class.astype(np.float32)

        # confidence score is directly used (see paper)
        s_conf = np.repeat(pert_det_scores[None, :], base_det_boxes.shape[0], axis=0)

        s_transl = self.compute_translation_score(base_det_boxes, pert_det_boxes)

        s_orient = self.compute_orientation_score(base_det_boxes, pert_det_boxes)

        s_score = self.compute_scale_score(base_det_boxes, pert_det_boxes)

        # sim_scores = s_overlap * s_conf * s_transl * s_orient * s_score * s_class



        # 수정
        sim_scores = s_overlap * s_class * s_conf * (
            0.4 * s_transl + 0.3 * s_orient + 0.3 * s_score)
        return sim_scores


    def compute_attribution_maps(self, pcl, base_det_boxes, base_det_labels,
                                batch_size, num_workers):
        # 디버깅 모드 체크
        if hasattr(self, 'debug_mode') and self.debug_mode:
            self.logger.info("🔍 Debug mode activated for attribution map computation")
            # 각 객체의 거리 미리 계산
            object_distances = np.linalg.norm(base_det_boxes[:, :2], axis=1)
        
        attr_maps = np.zeros((base_det_labels.shape[0], pcl.shape[0]))
        sampling_map = np.zeros(pcl.shape[0])
        result_cur_mask = np.zeros(pcl.shape[0], dtype=np.int32)

        occam_inference_dataset = OccamInferenceDataset(
            data_config=self.data_config, class_names=self.class_names,
            occam_config=self.occam_config, pcl=pcl, nr_it=self.nr_it,
            logger=self.logger
        )

        dataloader = DataLoader(
            occam_inference_dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, shuffle=False,
            collate_fn=occam_inference_dataset.collate_batch, drop_last=False,
            sampler=None, timeout=0
        )

        progress_bar = tqdm.tqdm(
            total=self.nr_it, leave=True, desc='OccAM computation',
            dynamic_ncols=True)
        
        result_cur_mask = np.full(pcl.shape[0], -1, dtype=np.int32)

        with torch.no_grad():
            for i, batch_dict in enumerate(dataloader):
                load_data_to_gpu(batch_dict)
                pert_pred_dicts, _ = self.model.forward(batch_dict)

                pert_det_boxes, pert_det_labels, pert_det_scores, batch_ids = \
                    self.merge_detections_in_batch(pert_pred_dicts)

                similarity_matrix = self.get_similarity_matrix(
                    base_det_boxes, base_det_labels,
                    pert_det_boxes, pert_det_labels, pert_det_scores)


                # for i, sim in enumerate(similarity_matrix):
                #     self.logger.info(f"[DEBUG] base obj {i} max similarity: {sim.max():.3f}")

                cur_batch_size = len(pert_pred_dicts)
                
                # ===== 디버깅 정보 수집 추가 =====
                if self.debug_mode and i % 10 == 0:  # 매 10번째 배치마다
                    self._collect_debug_info(
                        iteration=i * batch_size,
                        similarity_matrix=similarity_matrix,
                        object_distances=object_distances,
                        base_det_labels=base_det_labels,
                        batch_dict=batch_dict,
                        pert_det_boxes=pert_det_boxes,
                        pert_det_labels=pert_det_labels
                    )
                
                for j in range(cur_batch_size):
                    cur_mask = batch_dict['mask'][j, :].cpu().numpy()
                    result_cur_mask[cur_mask > 0] = 1
                    result_cur_mask[(cur_mask == 0) & (result_cur_mask == -1)] = 0
                    sampling_map += cur_mask

                    batch_sample_mask = batch_ids == j
                    if np.sum(batch_sample_mask) > 0:
                        max_score = np.max(
                            similarity_matrix[:, batch_sample_mask], axis=1)
                        attr_maps += max_score[:, None] * cur_mask

                progress_bar.update(n=cur_batch_size)

        # Normalize using occurrences
        attr_maps[:, sampling_map > 0] /= sampling_map[sampling_map > 0]
        
        # 디버깅 최종 요약
        if self.debug_mode:
            self.logger.info(f"🎯 Base detections: {len(base_det_labels)} objects")
            for i, (box, label) in enumerate(zip(base_det_boxes[:5], base_det_labels[:5])):
                distance = np.linalg.norm(box[:2])
                self.logger.info(f"  Object {i}: class={label}, distance={distance:.1f}m, center={box[:3]}")

        return attr_maps, result_cur_mask





    ####### attribute map만 시각화 #######
    def visualize_attr_map(self, points, box, attr_map, draw_origin=True):
        # 컬러맵 설정
        turbo_cmap = plt.get_cmap('turbo')
        attr_map_scaled = attr_map - attr_map.min()
        attr_map_scaled /= attr_map_scaled.max()
        color = turbo_cmap(attr_map_scaled)[:, :3]

        # 포인트 클라우드 생성
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        pts.colors = open3d.utility.Vector3dVector(color)

        # 바운딩 박스 생성
        rot_mat = Rotation.from_rotvec([0, 0, box[6]]).as_matrix()
        bb = open3d.geometry.OrientedBoundingBox(box[:3], rot_mat, box[3:6])
        bb.color = (1.0, 0.0, 1.0)

        # 바운딩 박스 생성 코드 필요함. 

        open3d.io.write_point_cloud("../../output_occam/attribute_map_with_bb.ply", pts)
        print("Point cloud saved as 'attribute_map.ply'")
    


###############
    def _collect_debug_info(self, iteration, similarity_matrix, object_distances, 
                        base_det_labels, batch_dict, pert_det_boxes, pert_det_labels):
        """디버깅 정보 수집"""
        for obj_idx in range(len(base_det_labels)):
            # 각 객체에 대한 최대 similarity
            if similarity_matrix.size > 0 and similarity_matrix[obj_idx].size > 0:
                max_sim = similarity_matrix[obj_idx].max()
                mean_sim = similarity_matrix[obj_idx].mean()
            else:
                max_sim = 0
                mean_sim = 0
            
            self.debug_data['similarities'].append(max_sim)
            self.debug_data['distances'].append(object_distances[obj_idx])
            self.debug_data['object_classes'].append(base_det_labels[obj_idx])
            
            # kept ratio 계산
            masks = batch_dict['mask'].cpu().numpy()
            avg_kept_ratio = masks.mean()
            self.debug_data['kept_ratios'].append(avg_kept_ratio)
            
            # iteration 정보 저장
            self.debug_data['iteration_info'].append({
                'iteration': iteration,
                'obj_idx': obj_idx,
                'max_sim': max_sim,
                'mean_sim': mean_sim,
                'n_pert_detections': len(pert_det_labels)
            })
        
        # 진행 상황 출력
        if iteration % 100 == 0:
            self._print_debug_stats(iteration)




    def _print_debug_stats(self, iteration):
        """현재까지의 디버깅 통계 출력"""
        if len(self.debug_data['similarities']) == 0:
            return
        
        distances = np.array(self.debug_data['distances'])
        similarities = np.array(self.debug_data['similarities'])
        kept_ratios = np.array(self.debug_data['kept_ratios'])
        
        self.logger.info(f"\n📊 Debug Stats at iteration {iteration}/{self.nr_it}")
        
        # 전체 통계
        self.logger.info(f"  Overall stats:")
        self.logger.info(f"    Mean similarity: {similarities.mean():.3f} (±{similarities.std():.3f})")
        self.logger.info(f"    Mean kept ratio: {kept_ratios.mean():.3f} (±{kept_ratios.std():.3f})")
        self.logger.info(f"    Zero similarity ratio: {(similarities == 0).mean():.3f}")
        
        # 거리별 similarity 통계
        self.logger.info(f"  Distance-based stats:")
        for dist_range in [(0, 20), (20, 40), (40, 60), (60, 80)]:
            mask = (distances >= dist_range[0]) & (distances < dist_range[1])
            if mask.sum() > 0:
                mean_sim = similarities[mask].mean()
                std_sim = similarities[mask].std()
                zero_ratio = (similarities[mask] == 0).mean()
                self.logger.info(f"    {dist_range[0]}-{dist_range[1]}m: "
                            f"mean={mean_sim:.3f} (±{std_sim:.3f}), "
                            f"zero_ratio={zero_ratio:.3f}, n={mask.sum()}")




    def _print_final_stats(self, attr_maps, sampling_map):
        """최종 attribution map 통계"""
        self.logger.info(f"  Attribution map shape: {attr_maps.shape}")
        self.logger.info(f"  Points with attribution > 0: {(attr_maps > 0).sum(axis=1).mean():.1f}")
        self.logger.info(f"  Mean sampling per point: {sampling_map.mean():.1f}")
        self.logger.info(f"  Points never sampled: {(sampling_map == 0).sum()}")
        
        # 각 객체별 attribution 통계
        for i in range(min(5, len(attr_maps))):  # 처음 5개 객체만
            attr_values = attr_maps[i][attr_maps[i] > 0]
            if len(attr_values) > 0:
                self.logger.info(f"  Object {i}: mean_attr={attr_values.mean():.3f}, "
                            f"max_attr={attr_values.max():.3f}, "
                            f"n_points={len(attr_values)}")

    def get_debug_summary(self):
        """디버깅 결과 요약 반환"""
        if not self.debug_mode or len(self.debug_data['similarities']) == 0:
            return None
        
        distances = np.array(self.debug_data['distances'])
        similarities = np.array(self.debug_data['similarities'])
        kept_ratios = np.array(self.debug_data['kept_ratios'])
        
        summary = {
            'total_samples': len(similarities),
            'mean_similarity': float(similarities.mean()),
            'std_similarity': float(similarities.std()),
            'mean_kept_ratio': float(kept_ratios.mean()),
            'zero_similarity_ratio': float((similarities == 0).mean()),
            'distance_stats': {}
        }
        
        for dist_range in [(0, 20), (20, 40), (40, 60), (60, 80)]:
            mask = (distances >= dist_range[0]) & (distances < dist_range[1])
            if mask.sum() > 0:
                summary['distance_stats'][f"{dist_range[0]}-{dist_range[1]}m"] = {
                    'count': int(mask.sum()),
                    'mean_sim': float(similarities[mask].mean()),
                    'std_sim': float(similarities[mask].std()),
                    'zero_ratio': float((similarities[mask] == 0).mean())
                }
        
        return summary

    def save_debug_data(self, filepath):
        """디버깅 데이터를 파일로 저장"""
        if self.debug_mode and len(self.debug_data['similarities']) > 0:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.debug_data, f)
            self.logger.info(f"Debug data saved to {filepath}")

    def plot_debug_analysis(self, save_path='debug_analysis.png'):
        """디버깅 결과 시각화"""
        if not self.debug_mode or len(self.debug_data['similarities']) == 0:
            return
        
        import matplotlib.pyplot as plt
        
        distances = np.array(self.debug_data['distances'])
        similarities = np.array(self.debug_data['similarities'])
        kept_ratios = np.array(self.debug_data['kept_ratios'])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Distance vs Similarity scatter plot
        ax = axes[0, 0]
        scatter = ax.scatter(distances, similarities, alpha=0.5, c=kept_ratios, cmap='viridis')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Similarity vs Distance')
        plt.colorbar(scatter, ax=ax, label='Kept Ratio')
        
        # 2. Similarity distribution
        ax = axes[0, 1]
        ax.hist(similarities, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(similarities.mean(), color='red', linestyle='--', label=f'Mean: {similarities.mean():.3f}')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Count')
        ax.set_title('Similarity Distribution')
        ax.legend()
        
        # 3. Distance-based box plot
        ax = axes[1, 0]
        dist_bins = [(0, 20), (20, 40), (40, 60), (60, 80)]
        box_data = []
        labels = []
        for dist_range in dist_bins:
            mask = (distances >= dist_range[0]) & (distances < dist_range[1])
            if mask.sum() > 0:
                box_data.append(similarities[mask])
                labels.append(f'{dist_range[0]}-{dist_range[1]}m')
        ax.boxplot(box_data, labels=labels)
        ax.set_ylabel('Similarity Score')
        ax.set_title('Similarity by Distance Range')
        
        # 4. Kept ratio over iterations
        ax = axes[1, 1]
        iterations = [info['iteration'] for info in self.debug_data['iteration_info']]
        iteration_sims = [info['max_sim'] for info in self.debug_data['iteration_info']]
        ax.scatter(iterations[:1000], iteration_sims[:1000], alpha=0.3, s=10)  # 처음 1000개만
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Max Similarity')
        ax.set_title('Similarity Progress')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        self.logger.info(f"Debug analysis plot saved to {save_path}")