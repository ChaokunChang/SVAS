import os
import argparse
from os import path as osp
from models.yolo_utils import coco_names

def get_video_length(vr, opt=None):
    if opt is None or opt.length is None:
        return len(vr)
    else:
        return opt.length

def get_video_title(opt):
    opt.video = os.path.abspath(opt.video)
    videoname = os.path.basename(opt.video)
    suffix = videoname.split('.')[-1]
    return videoname[:-len(suffix)-1]

def get_video_meta_home(opt):
    opt.video = os.path.abspath(opt.video)
    datahome = os.path.dirname(opt.video)
    videoname = os.path.basename(opt.video)
    suffix = videoname.split('.')[-1]
    metahome = osp.join(datahome, videoname[:-len(suffix)-1]+"_"+opt.task)
    os.makedirs(metahome, exist_ok=True)
    return metahome

def get_label_path(opt):
    opt.video = os.path.abspath(opt.video)
    # metahome = get_video_meta_home(opt)
    # return osp.join(metahome, "../demo.labels.npy")
    datahome = os.path.dirname(opt.video)
    return osp.join(datahome, "demo.labels.npy")

def get_partition_paths(opt):
    opt.video = os.path.abspath(opt.video)
    metahome = get_video_meta_home(opt)
    split_path = osp.join(metahome, "partition/")
    os.makedirs(split_path, exist_ok=True)
    train_ids = osp.join(split_path, "train_ids.npy")
    valid_ids = osp.join(split_path, "valid_ids.npy")
    test_ids = osp.join(split_path, "test_ids.npy")
    return train_ids, valid_ids, test_ids

def get_checkpoint_dir(opt):
    opt.video = os.path.abspath(opt.video)
    metahome = get_video_meta_home(opt)
    ckpt_dir = osp.join(metahome, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir

# def get_proxy_checkpoint_path(opt):
#     ckpt_dir=get_checkpoint_dir(opt)
#     ckpt_final_path=osp.join(ckpt_dir, f'proxy_final.pt')
#     return ckpt_final_path

def get_tmp_results_path(opt):
    opt.video = os.path.abspath(opt.video)
    metahome = get_video_meta_home(opt)
    return osp.join(metahome, f'results_{opt.task}_chunk{opt.chunk_size}_gpux{opt.num_gpus}_dd{opt.diff_delay}_dt{opt.diff_thresh}')

def get_tmp_infos_path(opt):
    opt.video = os.path.abspath(opt.video)
    metahome = get_video_meta_home(opt)
    return osp.join(metahome, f'infos_{opt.task}_chunk{opt.chunk_size}_gpux{opt.num_gpus}_dd{opt.diff_delay}_dt{opt.diff_thresh}') 

def get_preprocessing_options():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    # Read video
    parser.add_argument("--video", type=str, default="data/videos/example.mp4", help="path to the video of interest")
    parser.add_argument("--length", type=int, default=None, help="specify the length of the video, full length by default")
    parser.add_argument("--offset", type=int, default=0, help="from where to read the video")
    parser.add_argument("--img_size", type=int, nargs=2, default=[416, 416])

    # Label data
    parser.add_argument("--skip_labeling", default=False, action="store_true")
    parser.add_argument("--labeling_model", type=str, default="yolov5x", choices=['yolov5x'], help="which oracle model to label the data")
    parser.add_argument("--labeling_thresh", type=float, default=0.25, help="score to determine the label from model output. Mostly not used.")
    parser.add_argument("--labeling_batch", type=int, default=8)

    # Split 
    parser.add_argument("--skip_split", default=False, action="store_true")
    parser.add_argument("--num_train", type=float, default=0.005, help="training set size of the CMDN")
    parser.add_argument("--num_valid", type=float, default=0.001, help="validation set size of the CMDN")

    # Proxy Training
    parser.add_argument("--skip_proxy_train", default=False, action="store_true")
    parser.add_argument("--proxy_train_model", type=str, default="tinyresnet18", choices=["tinyresnet18"])
    parser.add_argument("--proxy_train_num_classes", type=int, default=2)
    parser.add_argument("--proxy_pretrained_model", default=False, help="Whether to use pretrained model", action="store_true")
    parser.add_argument("--proxy_train_epochs", type=int, default=10)
    parser.add_argument("--proxy_train_batch_size", type=int, default=64)
    parser.add_argument("--proxy_train_lr", type=float, default=0.001)
    
    # Task
    parser.add_argument("--task", type=str, default="car-n1")
    parser.add_argument("--target_object", type=str, default='car', choices=coco_names)
    parser.add_argument("--target_object_thresh", type=float, default=0.25)
    parser.add_argument("--target_object_count", type=int, default=1)

    # Other
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--save", default=False, help="save intermediate results", action="store_true")

    opt, _ = parser.parse_known_args()
    opt.video = os.path.abspath(opt.video)
    
    task_name = opt.target_object + "-n" + str(opt.target_object_count)
    assert task_name == opt.task, f'{task_name} != {opt.task}, Rename your --task.'

    return opt

def get_select_options():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--video", type=str, default="data/videos/example.mp4",
                        help="path to the video of interest")
    # parser.add_argument("--length", type=int, default=108000,
    #                     help="specify the length of the video, full length by default")
    parser.add_argument("--length", type=int, default=None, help="specify the length of the video, full length by default")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--img_size", type=int, nargs=2, default=[416, 416])

    # Difference Detector
    parser.add_argument("--diff_model", type=str,
                        help="model of difference detector", default="minus", choices=['minus', 'hist'])
    parser.add_argument("--diff_thresh", type=float,
                        help="threshold of the difference detector", default=1e-5)
    parser.add_argument("--diff_delay", type=int,
                        help="distance to compute difference of two frames, can be seen as sampling rate", default=10)
    parser.add_argument("--diff_batch_size", type=int,
                        help="batch_size of difference detector", default=64)


    # Proxy Model
    parser.add_argument("--proxy_model_ckpt", type=str, default="data/videos/example/checkpoint_final.pt",
                        help="checkpoiny of pre-trained proxy model")
    parser.add_argument("--proxy_batch_size", type=int, default=64, help="batch size to inference proxy model")
    parser.add_argument("--proxy_score_upper", type=float, default=0.9, 
                        help="The frames whose proxy score is higher that it will be considered as results directly without oracle label.")
    parser.add_argument("--proxy_score_lower", type=float, default=0.1, 
                        help="The frames whose proxy score is lower that it will be considered as not results directly without oracle label.")

    # Oracle Model
    # A better design is oracle model is yolov5-2, which means a special model for yolov5 which output car only
    parser.add_argument("--oracle_model", type=str, default="yolov5x",
                        choices=['yolov5'], help="model of oracle")
    parser.add_argument("--oracle_batch_size", type=int, default=16)

    # Scheduler
    parser.add_argument("--chunk_size", type=int, default=128, help="The number of frames inside each task to be executed.")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpu to be used. equal to the number of servers.")

    # Worker
    # pass

    # Task
    parser.add_argument("--task", type=str, default="car-n1")
    parser.add_argument("--target_object", type=str, default='car', choices=coco_names)
    parser.add_argument("--target_object_thresh", type=float, default=0.25)
    parser.add_argument("--target_object_count", type=int, default=1)

    # Other
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save", default=False,
                        help="save intermediate results", action="store_true")

    opt, _ = parser.parse_known_args()
    opt.video = os.path.abspath(opt.video)

    task_name = opt.target_object + "-n" + str(opt.target_object_count)
    assert task_name == opt.task, f'{task_name} != {opt.task}, Rename your --task.'

    return opt

def get_options():
    return get_select_options()