from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as torch_data
from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)

        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None

        # SparseKD 추가 처리
        if hasattr(self.data_processor, 'grid_size_tea'):
            self.grid_size_tea = self.data_processor.grid_size_tea
            self.voxel_size_tea = self.data_processor.voxel_size_tea

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    # pillar의 실제 좌표 계산
    def calculate_real_coords_pillar(self, voxel_coords):
        """
        Calculate the real coordinates for each voxel based on the voxel coordinates.
        Returns the full range of coordinates including min and max for each axis.

        Args:
            voxel_coords: (N, 3) array of voxel indices.

        Returns:
            voxel_boundaries: (N, 6) array of voxel boundaries where each row is
                            [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # print('voxel_coords.shape \\\\\\\\\\\\\\\\\\\\')
        # print(voxel_coords.shape)
        # print(' ')

        # voxel 좌표로부터 실제 공간의 시작 위치 (x_min, y_min, z_min) 계산
        if isinstance(voxel_coords, torch.Tensor):
            voxel_coords = voxel_coords.cpu().numpy()
        voxel_coords = voxel_coords[:, [2, 1, 0]]  # Swap z and x (z, y, x -> x, y, z)


        real_coords = voxel_coords * self.voxel_size + self.point_cloud_range[:3] 

        # print('voxel_coords: ', voxel_coords)
        # print('self.voxel_size: ', self.voxel_size)
        # print('self.point_cloud_range[:3]: ', self.point_cloud_range[:3])
        # print('real_coords: ', real_coords)

        # print('voxel_coords: ', voxel_coords)

        # 각 voxel의 x_max, y_max 계산, z_min과 z_max는 고정된 값 사용
        voxel_boundaries = []
        for real_coord in real_coords:
            
            x_min , y_min, _ = real_coord  # z는 따로 처리하기 때문에 real_coord에서 가져오지 않음
            x_max = x_min + self.voxel_size[0]
            y_max = y_min + self.voxel_size[1]

            # z_min과 z_max를 고정된 값으로 설정
            z_min = self.point_cloud_range[2]  # -3
            z_max = self.point_cloud_range[5]  # 1
            
            # voxel_boundaries 리스트에 추가
            # print('다음을 voxel_boundary에 추가')
            # print([x_min, y_min, z_min, x_max, y_max, z_max])

            voxel_boundaries.append([x_min, y_min, z_min, x_max, y_max, z_max])

        # 리스트를 numpy 배열로 변환
        voxel_boundaries = np.array(voxel_boundaries)
        return voxel_boundaries



    # pillar의 실제 좌표 계산
    def calculate_real_coords_voxel(self, voxel_coords):
        """
        Calculate the real coordinates for each voxel based on the voxel coordinates.
        Returns the full range of coordinates including min and max for each axis.

        Args:
            voxel_coords: (N, 3) array of voxel indices.

        Returns:
            voxel_boundaries: (N, 6) array of voxel boundaries where each row is
                            [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # print('voxel_coords.shape \\\\\\\\\\\\\\\\\\\\')
        # print(voxel_coords.shape)
        # print(' ')


        # voxel_coords가 텐서로 입력될 경우 Numpy로 변환
        if isinstance(voxel_coords, torch.Tensor):
            voxel_coords = voxel_coords.cpu().numpy()

        # voxel 좌표로부터 실제 공간의 시작 위치 (x_min, y_min, z_min) 계산
        if isinstance(voxel_coords, torch.Tensor):
            voxel_coords = voxel_coords.cpu().numpy()
        voxel_coords = voxel_coords[:, [2, 1, 0]]  # Swap z and x (z, y, x -> x, y, z)

        real_coords = voxel_coords * self.voxel_size + self.point_cloud_range[:3] 
        # voxel_coords: z, y, x-> x, y, z
        # self.voxel_size: 0.16, 0.16, 4
        # point_cloud_range: [0, -39.68, -3, 69.12, 39.68, 1],  self.point_cloud_range[:3]: 0(x), -39.68(y), -3(z)
        # print('*'*30)
        # print('voxel_coords: ', voxel_coords)
        # print('self.voxel_size: ', self.voxel_size)
        # print('self.point_cloud_range[:3]: ', self.point_cloud_range[:3])

        # print('voxel_coords: ', voxel_coords)

        # 각 voxel의 x_max, y_max 계산, z_min과 z_max는 고정된 값 사용
        voxel_boundaries = []
        for real_coord in real_coords:
            
            x_min , y_min, z_min = real_coord  # z는 따로 처리하기 때문에 real_coord에서 가져오지 않음
            x_max = x_min + self.voxel_size[0]
            y_max = y_min + self.voxel_size[1]

            # z_min과 z_max를 고정된 값으로 설정
            z_max = z_min + self.voxel_size[2]
           
            
            # voxel_boundaries 리스트에 추가
            # print('다음을 voxel_boundary에 추가')
            # print([x_min, y_min, z_min, x_max, y_max, z_max])

            voxel_boundaries.append([x_min, y_min, z_min, x_max, y_max, z_max])

        # 리스트를 numpy 배열로 변환
        voxel_boundaries = np.array(voxel_boundaries)
        return voxel_boundaries


    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            return {
                'name': np.zeros(num_samples), 
                'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]), 
                'pred_labels': np.zeros(num_samples)
            }

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()

            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        self._merge_all_iters_to_one_epoch = merge
        if merge:
            self.total_epochs = epochs

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def prepare_data(self, data_dict):
        # print('prepare_data!!')
        # print('data_dict[points].shape: ', data_dict['points'].shape)

        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            if 'calib' in data_dict:
                calib = data_dict['calib']
            data_dict = self.data_augmentor.forward(
                data_dict={**data_dict, 'gt_boxes_mask': gt_boxes_mask}
            )
            if 'calib' in data_dict:
                data_dict['calib'] = calib

        if 'gt_boxes' in data_dict:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        if 'points' in data_dict:
            data_dict = self.point_feature_encoder.forward(data_dict)
            # print('data_dict[points].shape: ', data_dict['points'].shape)

        data_dict = self.data_processor.forward(data_dict) ## 여기서 줄어들음!!!!!!
        # print('data_dict[points].shape: ', data_dict['points'].shape)

        # 추가 
        # 일단 voxel모델 하려고 수정했는데, 원래 pillar모델은 _pillar함수 써야함.
        # if 'voxel_coords' in data_dict: 
        #     print('voxel ~!!!!!!!!!!!!!!!')
        #     real_coords = self.calculate_real_coords_voxel(data_dict['voxel_coords'])
        #     data_dict['real_coords'] = real_coords


        ## PP!! pillar 모델 실험하려고 다시 수정함
        # print('여기서 pillar-coords 나오면 성공띠 data_dict.keys(): ', data_dict.keys())
        # data_dict.keys():  dict_keys(['frame_id', 'calib',
        # 'gt_names', 'gt_boxes', 'points', 'flip_x', 'noise_rot', 'noise_scale', 'use_lead_xyz', 
        #'voxels', 'voxel_coords', 'voxel_num_points', 'voxels_tea', 'voxel_coords_tea', 'voxel_num_points_tea'])
        # print('data_dict[voxel_coords] ', data_dict['voxel_coords'])
        if 'voxel_coords' in data_dict: 
            # print('voxel_coords 있노!!!!!!!!!!!!')
            # data_dict에 pillar_coords가 없어서 문제임. 
            # 그런데 여기는 데이터단이라 일단 voxel coords로 하고 한 번 더 필터링 해야할 것 같음. 
            # 엥 근데 real_coords 이렇게 하면 나중에 다른거랑 비교했을 때 더 작아짐..
            # 이거 data_dict 말고 batch_dict에 언제 추가되는지 체크
            # real_coords = self.calculate_real_coords_pillar(data_dict['voxel_coords']) #return voxel_boundaries
            # print('!!! data_dict[voxel_coords].shape: ', data_dict['voxel_coords'].shape)
            real_coords = self.calculate_real_coords_pillar(data_dict['voxel_coords']) #return voxel_boundaries
            data_dict['real_coords'] = real_coords # 추가하는 과정
        
        # print('### data_dict[real_coords].shape: ', data_dict['real_coords'].shape)
# 
        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points', 'voxels_tea', 'voxel_num_points_tea']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords', 'voxel_coords_tea', 'real_coords', 'pillar_coords']:
                    coors = [np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i) 
                             for i, coor in enumerate(val)]
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :len(val[k]), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print(f"Error in collate_batch: key={key}")
                raise TypeError

        ret['batch_size'] = batch_size


        
        # ✅ CRD용 sample_idx 추가
        nce_n = 1024       # CRD negative 개수
        dataset_size = 3712  # 전체 학습 샘플 수 (수정 필요!)
        sample_idx = []
        for _ in range(batch_size):
            pos = torch.randint(0, dataset_size, (1,))
            neg = torch.randint(0, nce_n, (nce_n,))  # 혹은 dataset_size
            sample_idx.append(torch.cat([pos, neg], dim=0))
        sample_idx = torch.stack(sample_idx, dim=0)
        ret['sample_idx'] = sample_idx

        return ret
