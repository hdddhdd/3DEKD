'''
FOR TRAINING AND INFERENCE
# '''
# 수정 후 - unique 하지 않게 (ver3)

#0314 논문용

from collections import namedtuple

import numpy as np
import torch
import open3d as o3d
import os
from .detectors import build_detector

try:
    import kornia
except:
    pass 
    print('Warning: kornia is not installed. This package is only required by CaDDN')


def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model

def build_teacher_network(cfg, args, train_set, dist, logger):
    teacher_model = build_network(model_cfg=cfg.MODEL_TEACHER, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    teacher_model.cuda()

    for param_k in teacher_model.parameters():
        param_k.requires_grad = False  # not update by gradient

    teacher_model.train()

    if args.teacher_ckpt is not None:
        logger.info('Loading teacher parameters >>>>>>')
        teacher_model.load_params_from_file(filename=args.teacher_ckpt, to_cpu=dist, logger=logger)

    teacher_model.is_teacher = True
    for cur_module in teacher_model.module_list:
        cur_module.is_teacher = True
        cur_module.kd = True

    return teacher_model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if key == 'camera_imgs':
            batch_dict[key] = val.cuda()
        elif not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'image_paths','ori_shape','img_process_infos']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator_original():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func


import torch.nn.functional as F
import torch
import numpy as np
import os
from collections import namedtuple

'''teacher -> occam 을 위해 추가'''
def compute_attribution_maps_on_the_fly(teacher_model, pcl, det_boxes, det_labels, batch_size, num_workers):
    """
    실시간으로 어트리뷰션 맵을 생성합니다.
    Args:
        teacher_model: OccAM 객체
        pcl: 포인트 클라우드 데이터
        det_boxes: 탐지된 박스
        det_labels: 탐지된 클래스 라벨
        batch_size: 배치 크기
        num_workers: 병렬 작업자 수
    Returns:
        attr_maps: 생성된 어트리뷰션 맵
    """
    # compute_attribution_maps: attr map 하나만 반환
    attr_maps, _ = teacher_model.compute_attribution_maps(
        pcl=pcl,
        base_det_boxes=det_boxes,
        base_det_labels=det_labels,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return attr_maps

import open3d as o3d

def save_selected_points_to_ply(pcl, selected_indices, ply_path):
    """
    선택된 포인트들을 .ply 파일로 저장하여 시각화할 수 있도록 함.
    
    Args:
        pcl (torch.Tensor): [K, 4] 형태의 포인트 클라우드 데이터 (x, y, z, intensity)
        selected_indices (torch.Tensor): 선택된 포인트들의 인덱스 (상위 10%)
        ply_path (str): 저장할 .ply 파일 경로
    """
    pcl_np = pcl.cpu().numpy()  # GPU 텐서를 NumPy 배열로 변환
    colors = np.ones((pcl_np.shape[0], 3))  # 모든 포인트를 기본 흰색 (RGB=[1,1,1])으로 설정

    if selected_indices.numel() > 0:  # 선택된 포인트가 있을 경우
        colors[selected_indices.cpu().numpy()] = [0, 1, 0]  # 상위 10% 포인트는 녹색 (RGB=[0,1,0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcl_np[:, :3])  # (x, y, z) 좌표 추가
    point_cloud.colors = o3d.utility.Vector3dVector(colors)  # 색상 추가

    o3d.io.write_point_cloud(ply_path, point_cloud)
    # print(f"[INFO] Saved selected points to: {ply_path}")



'''
FOR TRAINING AND INFERENCE
# '''

# 수정 후 - 3번 (unique제거)

from collections import namedtuple
from scipy.spatial import KDTree

def model_fn_decorator(distill_cfg=None):
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        # print('batch_dict================')
        # print(batch_dict.keys())
        # print(batch_dict['frame_id'])

        ret_dict, tb_dict, disp_dict = model(batch_dict)
        loss = ret_dict['loss'].mean()

        if distill_cfg is not None and distill_cfg.get('ENABLED', False):
            attr_map_dir = distill_cfg['ATTR_MAP_DIR']
            pcl_dir = distill_cfg['PCL_DIR']
            lambda_distill = distill_cfg['LAMBDA_DISTILL']
            batch_size = batch_dict['batch_size']
            total_distill_loss = 0.0

            for i in range(batch_size):
                frame_id = batch_dict['frame_id'][i]
                # print('frame_id: ', frame_id)

                attr_map = torch.tensor(np.load(os.path.join(attr_map_dir, f"{frame_id}_attr.npy")), dtype=torch.float32).cuda()
                # print('attr_map: ', attr_map)
                # print('path: f"{frame_id}_attr.npy")')
                pcl = torch.tensor(np.load(os.path.join(pcl_dir, f"{frame_id}_pcl.npy")), dtype=torch.float32).cuda()

                # Top 10% 포인트 선택 (빠른 방식)
                max_teacher_importance = attr_map.max(dim=0)[0]
                top_10_percent_values, top_10_percent_indices = torch.topk(max_teacher_importance, int(0.10 * max_teacher_importance.shape[0]), largest=True)
                # print('top_10_percent_indices.shape: ', top_10_percent_indices.shape)

                # 🔹 KDTree를 사용한 Pillar 매칭
                pillar_centers = (batch_dict['real_coords'][:, 1:4] + batch_dict['real_coords'][:, 4:]) / 2
                pillar_tree = KDTree(pillar_centers.cpu().numpy())

                # 🔹 각 Top 10% 포인트에 대해 가장 가까운 Pillar 찾기 (중복 제거 X)
                _, nearest_pillar_indices = pillar_tree.query(pcl[top_10_percent_indices, :3].cpu().numpy(), k=1)

                # 🔹 선택된 Pillar에 해당하는 학생 모델 Feature 가져오기 (중복 제거 X)
                # print('batch_dict.keys(): ', batch_dict.keys())
                student_features = batch_dict['pillar_features'][nearest_pillar_indices]  # Shape: [M, D]

                # 🔹 각 Pillar에 속한 포인트들의 중요도를 평균화
                teacher_importance = max_teacher_importance[top_10_percent_indices]  # Shape: [M]
                weighted_features = student_features * teacher_importance.unsqueeze(-1)  # Shape: [M, D]

                # 🔹 Distillation Loss 계산 (포인트별로 계산 후 평균)
                cosine_similarities = F.cosine_similarity(weighted_features, student_features, dim=-1)  # Shape: [M]
                distill_losses = 1 - cosine_similarities
                # print('distill_losses.shape: ', distill_losses.shape)
                # print('distill_losses: ', distill_losses)


                if len(distill_losses) > 0:
                    total_distill_loss += distill_losses.mean()

            loss += lambda_distill * (total_distill_loss / batch_size)
            tb_dict['distill_loss'] = total_distill_loss / batch_size

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func




import torch.nn.functional as F
from collections import namedtuple
def model_fn_fitnet(teacher_model, distill_cfg=None):
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)

        teacher_model.eval()
        with torch.no_grad():
            _ = teacher_model(batch_dict)

        ret_dict, tb_dict, disp_dict = model(batch_dict)  # Student forward

        detection_loss = ret_dict['loss'].mean()

        if distill_cfg is not None and distill_cfg.get('ENABLED', False):
            lambda_distill = distill_cfg['LAMBDA_DISTILL']

            teacher_features = teacher_model.get_intermediate_features(batch_dict)
            student_features = model.get_intermediate_features(batch_dict)

            if teacher_features.shape != student_features.shape:
                min_channels = min(teacher_features.shape[1], student_features.shape[1])
                teacher_features = teacher_features[:, :min_channels]
                student_features = student_features[:, :min_channels]

            # distillation_loss = torch.mean((student_features - teacher_features) ** 2)
            distillation_loss = F.mse_loss(student_features, teacher_features)  # MSE Loss를 사용한 출력 비교

            tb_dict['distill_loss'] = distillation_loss.item()
            # print('distillation loss :', distillation_loss)
            loss = detection_loss + lambda_distill * distillation_loss
        else:
            loss = detection_loss

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func

from .crd import CRD
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

def model_fn_crd(teacher_model, crd_module, distill_cfg=None):
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)

        # 1. Teacher forward pass (no grad)
        teacher_model.eval()
        with torch.no_grad():
            _ = teacher_model(batch_dict)

        # 2. Student forward pass
        ret_dict, tb_dict, disp_dict = model(batch_dict)
        detection_loss = ret_dict['loss'].mean()

        if distill_cfg is not None and distill_cfg.get('ENABLED', False):
            lambda_crd = 0.01
            idx = batch_dict['idx'].long()
            sample_idx = batch_dict['sample_idx'].long()

            # 3. Feature 가져오기 (hook 기반)
            feat_s = model.get_intermediate_features(batch_dict)  # [B, C, H, W]
            feat_t = teacher_model.get_intermediate_features(batch_dict)  # [B, C, H, W]

            # 🔥 Spatial 평균 pooling → [B, C]
            feat_s = feat_s.mean(dim=[2, 3])  # [B, C]
            feat_t = feat_t.mean(dim=[2, 3])  # [B, C]

            # 🔥 Index tensor GPU 이동
            idx = idx.to(feat_s.device)
            sample_idx = sample_idx.to(feat_s.device)

            # 4. CRD loss 계산
            distill_loss = crd_module(feat_s, feat_t, idx, sample_idx)
            tb_dict['distill_loss'] = distill_loss.item()

            # 5. 총 loss 계산
            loss = detection_loss + lambda_crd * distill_loss
        else:
            loss = detection_loss

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
# # # 1. import 및 CRD 구성 요소 준비
# from .crd import CRD
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import namedtuple

# # 2. model_fn_crd 정의 (FitNet 구조 그대로 유지)
# def model_fn_crd(teacher_model, crd_module, distill_cfg=None):
#     ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

#     def model_func(model, batch_dict):
#         load_data_to_gpu(batch_dict)

#         # 1. Teacher forward pass
#         teacher_model.eval()
#         with torch.no_grad():
#             _ = teacher_model(batch_dict)

#         # 2. Student forward pass
#         ret_dict, tb_dict, disp_dict = model(batch_dict)
#         detection_loss = ret_dict['loss'].mean()

#         if distill_cfg is not None and distill_cfg.get('ENABLED', False):
#             lambda_crd = 0.01
#             idx = batch_dict['idx']
#             sample_idx = batch_dict['sample_idx']


#             # 3. Feature 가져오기 (hook 기반)
#             # feature: [11770, 1, C] → squeeze: [11770, C]
#             feat_s = model.get_intermediate_features(batch_dict).squeeze(1)
#             feat_t = teacher_model.get_intermediate_features(batch_dict).squeeze(1)
#             # 🔥 반드시 Long 타입으로 변환 + GPU 이동
#             idx = idx.long().to(feat_s.device)
#             sample_idx = sample_idx.long().to(feat_s.device)
#             # 🔥 각 voxel이 어떤 batch에 속하는지
#             batch_inds = batch_dict['voxel_coords'][:, 0]  # shape: [11770]

#             B = batch_dict['batch_size']
#             pooled_feat_s, pooled_feat_t = [], []
#             print(feat_s.shape)
#             print(feat_t.shape)
#             for b in range(B):
#                 mask = (batch_inds == b)
#                 pooled_feat_s.append(feat_s[mask].mean(dim=0, keepdim=True))  # [1, C] ####
#                 pooled_feat_t.append(feat_t[mask].mean(dim=0, keepdim=True))  # [1, C]

#             feat_s = torch.cat(pooled_feat_s, dim=0)  # [B, C]
#             feat_t = torch.cat(pooled_feat_t, dim=0)  # [B, C]

#             distill_loss = crd_module(feat_s, feat_t, idx, sample_idx) 
#             tb_dict['distill_loss'] = distill_loss.item()
#             loss = detection_loss + lambda_crd * distill_loss
#         else:
#             loss = detection_loss

#         return ModelReturn(loss, tb_dict, disp_dict)

#     return model_func

# 수정 전
# def model_fn_decorator(distill_cfg=None):
#     ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

#     def model_func(model, batch_dict):
#         load_data_to_gpu(batch_dict)  # Load data to GPU
#         # print('batch_dict================')
#         # print(batch_dict.keys())
#         # dict_keys(['frame_id', 'calib', 'gt_boxes', 'points', 'trans_lidar_to_cam', 'trans_cam_to_img', 'flip_x', 'noise_rot', 'noise_scale', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 'real_coords', 'image_shape', 'batch_size'])
#         ret_dict, tb_dict, disp_dict = model(batch_dict)  # Forward pass
#         loss = ret_dict['loss'].mean()  # Base loss computation

#         if distill_cfg is not None and distill_cfg.get('ENABLED', False):
#             attr_map_dir = distill_cfg['ATTR_MAP_DIR']
#             lambda_distill = distill_cfg['LAMBDA_DISTILL']
#             batch_size = batch_dict['batch_size']

#             total_distill_loss = 0.0

#             for i in range(batch_size):
#                 frame_id = batch_dict['frame_id'][i]
#                 attr_map_path = os.path.join(attr_map_dir, f"{frame_id}_attr.npy")

#                 if not os.path.isfile(attr_map_path):
#                     raise FileNotFoundError(f"Attribution map file not found: {attr_map_path}")

#                 attr_map = np.load(attr_map_path)  # Load attribution map
#                 attr_map = torch.tensor(attr_map, dtype=torch.float32).cuda()

#                 if 'pillar_features' in batch_dict:
#                     # print('pillar_features')
#                     batch_mask = (batch_dict['voxel_coords'][:, 0] == i)
#                     voxel_coords = batch_dict['voxel_coords'][batch_mask]
#                     real_coords = batch_dict['real_coords'][batch_mask]

#                     student_pillar_features = batch_dict['pillar_features'][batch_mask]

#                     norms = torch.norm(student_pillar_features, dim=1) #ex. 8050
#                     # print('norms.shape: ', norms.shape)
#                     top_indices = torch.topk(norms, k=min(130, norms.size(0)), largest=True).indices ## 원본
#                     # top_indices = torch.topk(norms, k=min(100, norms.size(0)), largest=True).indices ## 원본
#                     # top_indices = torch.topk(norms, k=min(100, norms.size(0)), largest=False).indices ## 작은거부터 하도록 수정

#                     distill_losses = []

#                     for j in top_indices:
#                         real_coords_row = real_coords[j][1:]  # Coordinate range for the pillar
#                         if len(real_coords_row) != 6:
#                             continue

#                         x_min, y_min, z_min, x_max, y_max, z_max = real_coords_row

#                         points_xyz = batch_dict['points'][:, 1:4]
#                         x_mask = (points_xyz[:, 0] >= x_min) & (points_xyz[:, 0] <= x_max)
#                         y_mask = (points_xyz[:, 1] >= y_min) & (points_xyz[:, 1] <= y_max)
#                         z_mask = (points_xyz[:, 2] >= z_min) & (points_xyz[:, 2] <= z_max)

#                         mask = x_mask & y_mask & z_mask

#                         if mask.any():
#                             selected_points_idx = mask.nonzero(as_tuple=True)[0]

#                             batch_offset = i * attr_map.shape[1]
#                             adjusted_selected_points_idx = selected_points_idx - batch_offset
#                             valid_idx_mask = (adjusted_selected_points_idx >= 0) & (adjusted_selected_points_idx < attr_map.shape[1])
#                             valid_selected_points_idx = adjusted_selected_points_idx[valid_idx_mask]

#                             if valid_selected_points_idx.numel() > 0:
#                                 teacher_importance = attr_map[:, valid_selected_points_idx].max(dim=1)[0]                               
#                                 student_feature = student_pillar_features[j]

#                                 # 기존 방법
#                                 teacher_importance_avg = teacher_importance.mean()
#                                 weighted_feature = student_feature * teacher_importance_avg
                
#                                 cosine_similarity = F.cosine_similarity(weighted_feature.unsqueeze(0), student_feature.unsqueeze(0), dim=-1)
#                                 distill_loss = 1 - cosine_similarity.mean()

#                                 distill_losses.append(distill_loss)
#                             # else:
#                             #     print(f"No valid selected points found for pillar {j}. Skipping.") ###
#                         else:
#                             print(f"No points found in mask for pillar {j}.")

#                     if len(distill_losses) > 0:
#                         # print('total_distill_loss: ', total_distill_loss)
#                         # print('torch.stack(distill_losses).mean(): ', torch.stack(distill_losses).mean())

#                         total_distill_loss += torch.stack(distill_losses).mean()
#                         # print('total_distill_loss: ', total_distill_loss)

#             #########################################################
#             if batch_size > 0:
#                 total_distill_loss /= batch_size

#             loss += lambda_distill * total_distill_loss
#             tb_dict['distill_loss'] = total_distill_loss


#         if hasattr(model, 'update_global_step'):
#             model.update_global_step()
#         else:
#             model.module.update_global_step()

#         return ModelReturn(loss, tb_dict, disp_dict)

#     return model_func

