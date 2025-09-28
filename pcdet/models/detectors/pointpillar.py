
from .detector3d_template import Detector3DTemplate
import torch.nn as nn
import copy

class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        self.teacher_inter_feat = None
        self.student_inter_feat = None

        # Hook 등록
        self._register_hooks()

    def _register_hooks(self):
        target_block_idx = 2  # 필요 시 인덱스 조정

        # backbone_2d 또는 backbone 구조에서 hook 설정
        if hasattr(self, 'backbone_2d'):
            def hook_fn(module, input, output):
                if getattr(self, 'is_teacher', False):
                    self.teacher_inter_feat = output
                else:
                    self.student_inter_feat = output

            self.backbone_2d.blocks[target_block_idx].register_forward_hook(hook_fn)

        elif hasattr(self, 'backbone'):
            def hook_fn(module, input, output):
                if getattr(self, 'is_teacher', False):
                    self.teacher_inter_feat = output
                else:
                    self.student_inter_feat = output

            # 일반 backbone에는 blocks가 아니라 sequential일 수도 있으니 리스트로 접근
            if isinstance(self.backbone, nn.Sequential):
                self.backbone[target_block_idx].register_forward_hook(hook_fn)
            elif hasattr(self.backbone, 'blocks'):
                self.backbone.blocks[target_block_idx].register_forward_hook(hook_fn)

    def forward(self, batch_dict):
        # 수정: 중간 레이어의 출력을 저장하기 위해 변수 추가
        feature_map = None

        for idx, cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)

            if 'pillar_features' in batch_dict:
                feature_map = batch_dict['pillar_features']

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }

            self.feature_map = feature_map
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def get_intermediate_features(self, batch_dict):
        # hook으로 저장된 중간 feature 반환
        if getattr(self, 'is_teacher', False):
            # print("[HOOK] Teacher feature captured")
            return self.teacher_inter_feat
        else:
            # print("[HOOK] Student feature captured")
            return self.student_inter_feat



# from .detector3d_template import Detector3DTemplate

# class PointPillar(Detector3DTemplate):
#     def __init__(self, model_cfg, num_class, dataset):
#         super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
#         self.module_list = self.build_networks()

    # def forward(self, batch_dict):
    #     # 수정: 중간 레이어의 출력을 저장하기 위해 변수 추가
    #     feature_map = None

    #     for idx, cur_module in enumerate(self.module_list):
    #         batch_dict = cur_module(batch_dict)

    #         if 'pillar_features' in batch_dict:
    #             feature_map = batch_dict['pillar_features']

    #     if self.training:
    #         loss, tb_dict, disp_dict = self.get_training_loss()

    #         ret_dict = {
    #             'loss': loss
    #         }

    #         self.feature_map = feature_map
    #         return ret_dict, tb_dict, disp_dict
    #     else:
    #         pred_dicts, recall_dicts = self.post_processing(batch_dict)
    #         return pred_dicts, recall_dicts

#     def get_training_loss(self):
#         disp_dict = {}

#         loss_rpn, tb_dict = self.dense_head.get_loss()
#         tb_dict = {
#             'loss_rpn': loss_rpn.item(),
#             **tb_dict
#         }

#         loss = loss_rpn
#         return loss, tb_dict, disp_dict
