import os, math
import yaml
import random
import argparse
import numpy as np
from os import path as osp
from dataloader.video_loader import DecordVideoReader, VideoLoader
from dataloader.label_loader import CachedGTLabelReader
from models.detector import YOLOv5

# def get_cached_gt_path(opt):
#     filename = f"{get_video_prefix(opt.video)}_{opt.udf}.npy"
#     return os.path.join(cached_gt_dir, filename)

# def get_video_prefix(video_path):
#     video_name = os.path.basename(video_path)
#     return os.path.splitext(video_name)[0]

# def get_split_path(opt):
#     video_prefix = get_video_prefix(opt.video)
#     return os.path.join(video_dir, video_prefix, "split")

def get_video_title(path_to_video):
    path_to_video = os.path.abspath(path_to_video)
    videoname = os.path.basename(path_to_video)
    suffix = videoname.split('.')[-1]
    return videoname[:-len(suffix)-1]

def get_video_meta_home(path_to_video):
    path_to_video = os.path.abspath(path_to_video)
    datahome = os.path.dirname(path_to_video)
    videoname = os.path.basename(path_to_video)
    suffix = videoname.split('.')[-1]
    metahome = os.path.join(datahome, videoname[:-len(suffix)-1])
    os.makedirs(metahome, exist_ok=True)
    return metahome

def get_label_path(path_to_video):
    path_to_video = os.path.abspath(path_to_video)
    metahome = get_video_meta_home(path_to_video)
    return os.path.join(metahome, get_video_title(path_to_video)+".label.npy")

def get_checkpoint_dir(path_to_video):
    path_to_video = os.path.abspath(path_to_video)
    metahome = get_video_meta_home(path_to_video)
    return os.path.join(metahome, "checkpoint") 

def get_video_length(vr, opt=None):
    if opt is None or opt.length is None:
        return len(vr)
    else:
        return opt.length

def data_labeling(opt):
    img_size = opt.img_size
    if opt.gpu is not None:
        device = f'cuda:{opt.gpu}'
    else:
        device = 'cpu'
    model = YOLOv5(model_type="yolov5x", long_side=img_size[0], device=device, fp16=False)
    vr = DecordVideoReader(opt.video, img_size=tuple(opt.img_size), gpu=opt.gpu, offset=opt.offset)
    vr_length = len(vr)
    batch_size = opt.batch_size
    # num_batch = math.ceil(vr_length / batch_size)
    results = []
    for i in range(0, vr_length, batch_size):
        batch_ids = [idx for idx in range(i, max(vr_length, i+batch_size))]
        frames = vr.get_batch(batch_ids)
        results += model.infer(frames)
    
    output_path = get_label_path(opt.video)
    np.save(output_path, np.array(results))
    

def get_partition_size(vr, opt):
    length = get_video_length(vr, opt)
    num_train = int(max(0.005 * length, 30000))
    num_valid = int(min(3000, num_train / 10))
    if opt is not None:
        if opt.num_train is not None:
            if opt.num_train <= 1:
                num_train = int(opt.num_train * length)
            else:
                num_train = int(opt.num_train)
        if opt.num_valid is not None:
            if opt.num_valid <= 1:
                num_valid = int(opt.num_valid * length)
            else:
                num_valid = int(opt.num_valid)
    return num_train, num_valid

def split_dataset(opt, vr, lr, save=False):
    random.seed(opt.random_seed)
    length = get_video_length(vr, opt)
    indices = list(range(length))
    num_train, num_valid = get_partition_size(vr, opt)
    random.shuffle(indices)
    train = indices[:num_train]
    valid = indices[num_train:num_train+num_valid]
    test = indices[num_train+num_valid:]

    train.sort()
    test.sort()
    valid.sort()

    # compute the label weight for training CMDN
    scores = lr.get_batch(train + valid)
    hist = np.ones([opt.max_score]) * 10 # to smoothn the weight
    for s in scores:
        hist[s] += 1

    sample_max = hist.max()
    weight = sample_max / hist

    train = np.array(train)
    valid = np.array(valid)
    test = np.array(test)
    weight = np.array(weight)

    if save:
        metahome = get_video_meta_home(opt.video)
        split_path = os.path.join(metahome, "partition/")
        os.makedirs(split_path, exist_ok=True)
        np.save(os.path.join(split_path, "train_ids.npy"), train)
        np.save(os.path.join(split_path, "valid_ids.npy"), valid)
        np.save(os.path.join(split_path, "test_ids.npy"), test)
        np.save(os.path.join(metahome, "score_weight.npy"), weight)

    return train, valid, test, weight

if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--video", type=str, default="videos/archie.mp4", help="path to the video of interest")
    parser.add_argument("--length", type=int, default=None, help="specify the length of the video, full length by default")
    parser.add_argument("--diff_thres", type=float, help="threshold of the difference detector")
    parser.add_argument("--num_train", type=float, default=0.005, help="training set size of the CMDN")
    parser.add_argument("--num_valid", type=float, default=3000, help="validation set size of the CMDN")
    parser.add_argument("--max_score", type=int, default=50, help="the maximum score")
    parser.add_argument("--udf", type=str, default="number_of_cars", help="the name of the scoring UDF")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--cmdn_train_epochs", type=int, default=10)
    parser.add_argument("--cmdn_train_batch", type=int, default=64)
    parser.add_argument("--cmdn_scan_batch", type=int, default=60)
    parser.add_argument("--oracle_batch", type=int, default=8)
    parser.add_argument("--conf_thres", type=float, default=0.9)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--window_samples", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--skip_train_cmdn", default=False, action="store_true")
    parser.add_argument("--skip_cmdn_scan", default=False, action="store_true")
    parser.add_argument("--skip_topk", default=False, action="store_true")
    parser.add_argument("--save", default=False, help="save intermediate results", action="store_true")
    opt, _ = parser.parse_known_args()

    vr = DecordVideoReader(opt.video, img_size=tuple(opt.img_size), offset=opt.offset)
    cached_gt_path = get_label_path(opt.video)
    if not os.path.exists(cached_gt_path):
        print("Cached Label doesn't exist, generating now...")
        data_labeling(opt)
    lr = CachedGTLabelReader(cached_gt_path, opt.offset)

    split_dataset(opt, vr, lr, True)
