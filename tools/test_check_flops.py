import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
#, build_teacher_network
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--infer_time', action='store_true', default=False, help='')
    parser.add_argument('--cal_params', action='store_true', default=False, help='')
    parser.add_argument('--teacher_ckpt', type=str, default=None, help='checkpoint to start from')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg) ##
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
        #, save_to_file=args.save_to_file
    )


def train_eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, ckpt_dir, dist_test=False):
    # load checkpoint
    filename = os.path.join(ckpt_dir, 'checkpoint_epoch_%s.pth' % epoch_id)
    model.load_params_from_file(filename=filename, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=False
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


# def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
#     # evaluated ckpt record
#     ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
#     with open(ckpt_record_file, 'a'):
#         pass

#     # tensorboard log
#     if cfg.LOCAL_RANK == 0:
#         tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
#     total_time = 0
#     first_eval = True

#     while True:
#         # check whether there is checkpoint which is not evaluated
#         cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
#         if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
#             wait_second = 30
#             if cfg.LOCAL_RANK == 0:
#                 print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
#                       % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
#             time.sleep(wait_second)
#             total_time += 30
#             if total_time > args.max_waiting_mins * 60 and (first_eval is False):
#                 break
#             continue

#         total_time = 0
#         first_eval = False

#         model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
#         model.cuda()

#         # start evaluation
#         cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
#         tb_dict = eval_utils.eval_one_epoch(
#             cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
#             result_dir=cur_result_dir,
#         )
        
#         if cfg.LOCAL_RANK == 0:
#             for key, val in tb_dict.items():
#                 tb_log.add_scalar(key, val, cur_epoch_id)

#         # record this epoch which has been evaluated
#         with open(ckpt_record_file, 'a') as f:
#             print('%s' % cur_epoch_id, file=f)
#         logger.info('Epoch %s has been evaluated' % cur_epoch_id)


# print('*'*30)
# print(tb_dict)
# {'recall/roi_0.3': 0.0, 'recall/rcnn_0.3': 0.9255040437407449, 'recall/roi_0.5': 0.0, 'recall/rcnn_0.5': 0.8488438318715116, 'recall/roi_0.7': 0.0, 'recall/rcnn_0.7': 0.5652693928693473, 'Car_aos/easy_R40': 94.68994240924363, 'Car_aos/moderate_R40': 87.83438077859994, 'Car_aos/hard_R40': 85.21059805511891, 'Car_3d/easy_R40': 82.49271580553793, 'Car_3d/moderate_R40': 70.04273295812983, 'Car_3d/hard_R40': 67.10861842629198, 'Car_bev/easy_R40': 91.1753588581116, 'Car_bev/moderate_R40': 84.2644360777844, 'Car_bev/hard_R40': 81.57305972333596, 'Car_image/easy_R40': 94.74838296376117, 'Car_image/moderate_R40': 88.23343994233319, 'Car_image/hard_R40': 85.79611647952848, 'Pedestrian_aos/easy_R40': 44.585170225365225, 'Pedestrian_aos/moderate_R40': 40.099302716924576, 'Pedestrian_aos/hard_R40': 37.974316689421, 'Pedestrian_3d/easy_R40': 50.480894226782105, 'Pedestrian_3d/moderate_R40': 44.40950515577865, 'Pedestrian_3d/hard_R40': 40.604134748484384, 'Pedestrian_bev/easy_R40': 55.435339558952634, 'Pedestrian_bev/moderate_R40': 49.7291833487572, 'Pedestrian_bev/hard_R40': 46.27865840054181, 'Pedestrian_image/easy_R40': 61.69195377189881, 'Pedestrian_image/moderate_R40': 57.07051054542443, 'Pedestrian_image/hard_R40': 54.33853393535056, 'Cyclist_aos/easy_R40': 81.79675946638358, 'Cyclist_aos/moderate_R40': 68.10275041480696, 'Cyclist_aos/hard_R40': 63.91540977463909, 'Cyclist_3d/easy_R40': 72.67960509083503, 'Cyclist_3d/moderate_R40': 54.74519478616611, 'Cyclist_3d/hard_R40': 51.48538342651702, 'Cyclist_bev/easy_R40': 75.88162545389491, 'Cyclist_bev/moderate_R40': 57.97052017250749, 'Cyclist_bev/hard_R40': 54.75153341093135, 'Cyclist_image/easy_R40': 83.18044946013062, 'Cyclist_image/moderate_R40': 69.8200894471412, 'Cyclist_image/hard_R40': 65.63678266709026}

# def repeat_eval_ckpt_with_best(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
#     # 평가된 체크포인트 기록 파일
#     ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
#     with open(ckpt_record_file, 'a'):
#         pass

#     # TensorBoard 로그
#     if cfg.LOCAL_RANK == 0:
#         tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))

#     # 최고 성능 모델 정보
#     best_epoch_id = None
#     best_performance = None
#     best_metrics = {}

#     total_time = 0
#     first_eval = True

#     while True:
#         # 평가되지 않은 체크포인트 확인
#         cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
#         if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
#             wait_second = 30
#             if cfg.LOCAL_RANK == 0:
#                 print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
#                       % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
#             time.sleep(wait_second)
#             total_time += 30
#             if total_time > args.max_waiting_mins * 60 and (first_eval is False):
#                 break
#             continue

#         total_time = 0
#         first_eval = False

#         model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
#         model.cuda()

#         # 평가 시작
#         cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
#         tb_dict = eval_utils.eval_one_epoch(
#             cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
#             result_dir=cur_result_dir,
#         )

#         if cfg.LOCAL_RANK == 0:
#             for key, val in tb_dict.items():
#                 tb_log.add_scalar(key, val, cur_epoch_id)

#         # 성능 비교 및 업데이트

#         current_performance = tb_dict.get('recall/rcnn_0.3', 0) 
#         if best_performance is None or current_performance > best_performance:
#             best_epoch_id = cur_epoch_id
#             best_performance = current_performance
#             best_metrics = tb_dict
#             logger.info(f"New best model found at epoch {cur_epoch_id}: {current_performance}")

#         # 평가된 체크포인트 기록
#         with open(ckpt_record_file, 'a') as f:
#             print('%s' % cur_epoch_id, file=f)
#         logger.info('Epoch %s has been evaluated' % cur_epoch_id)

#     # 최종 최고 성능 출력
#     logger.info(f"Best model: Epoch {best_epoch_id} with performance {best_performance}")
#     for metric, value in best_metrics.items():
#         logger.info(f"{metric}: {value}")

#     return best_epoch_id, best_performance, best_metrics

def repeat_eval_ckpt_with_best(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # 평가된 체크포인트 기록 파일
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # TensorBoard 로그
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))

    # 최고 성능 모델 정보
    best_epoch_id = None
    best_performance = None
    best_metrics = {}

    total_time = 0
    first_eval = True

    while True:
        # 평가되지 않은 체크포인트 확인
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # 평가 시작
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir,
        )

        print('tb_dict')
        '''
        tb_dict.keys()
        dict_keys(['recall/roi_0.3', 'recall/rcnn_0.3', 'recall/roi_0.5', 'recall/rcnn_0.5', 'recall/roi_0.7', 'recall/rcnn_0.7', 
        'Car_aos/easy_R40', 'Car_aos/moderate_R40', 'Car_aos/hard_R40', 
        'Car_3d/easy_R40', 'Car_3d/moderate_R40', 'Car_3d/hard_R40',
        'Car_bev/easy_R40', 'Car_bev/moderate_R40', 'Car_bev/hard_R40', 
        'Car_image/easy_R40', 'Car_image/moderate_R40', 'Car_image/hard_R40', 
        'Pedestrian_aos/easy_R40', 'Pedestrian_aos/moderate_R40', 'Pedestrian_aos/hard_R40',
        'Pedestrian_3d/easy_R40', 'Pedestrian_3d/moderate_R40', 'Pedestrian_3d/hard_R40',
        'Pedestrian_bev/easy_R40', 'Pedestrian_bev/moderate_R40', 'Pedestrian_bev/hard_R40',
        'Pedestrian_image/easy_R40', 'Pedestrian_image/moderate_R40', 'Pedestrian_image/hard_R40',
        'Cyclist_aos/easy_R40', 'Cyclist_aos/moderate_R40', 'Cyclist_aos/hard_R40',
        'Cyclist_3d/easy_R40', 'Cyclist_3d/moderate_R40', 'Cyclist_3d/hard_R40',
        'Cyclist_bev/easy_R40', 'Cyclist_bev/moderate_R40', 'Cyclist_bev/hard_R40',
        'Cyclist_image/easy_R40', 'Cyclist_image/moderate_R40', 'Cyclist_image/hard_R40'])

        '''
        print(tb_dict.keys())

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # 클래스별 평균 계산 (특정 지표 제외)
        filtered_metrics = {k: v for k, v in tb_dict.items() if not k.startswith('recall/')}
        class_metrics = {}
        for key, value in filtered_metrics.items():
            class_name = key.split('_')[0]  # 클래스 이름 추출
            if class_name not in class_metrics:
                class_metrics[class_name] = []
            class_metrics[class_name].append(value)

        class_averages = {k: sum(v) / len(v) for k, v in class_metrics.items()}
        current_performance = sum(class_averages.values()) / len(class_averages)  # 클래스 평균의 평균

        # 최고 성능 모델 갱신
        if best_performance is None or current_performance > best_performance:
            best_epoch_id = cur_epoch_id
            best_performance = current_performance
            best_metrics = tb_dict
            logger.info(f"New best model found at epoch {cur_epoch_id}: {current_performance}")

        # 평가된 체크포인트 기록
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)

    # 최종 최고 성능 출력
    logger.info(f"Best model: Epoch {best_epoch_id} with performance {best_performance}")
    for metric, value in best_metrics.items():
        logger.info(f"{metric}: {value}")

    return best_epoch_id, best_performance, best_metrics

def main():
    args, cfg = parse_config() ##

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    logger.info(model)

    print('[test] dist_test: ', dist_test)
    with torch.no_grad():
        if args.eval_all:
            # repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
            repeat_eval_ckpt_with_best(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()