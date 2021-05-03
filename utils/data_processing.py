import os, math
import yaml
from tqdm import tqdm
import random
import argparse
import numpy as np
from os import path as osp
from dataloader.video_loader import DecordVideoReader, VideoLoader
from dataloader.label_loader import CachedGTLabelReader
from models.detector import YOLOv5
from utils.parser import get_options

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

def get_tmp_results_path(path_to_video):
    path_to_video = os.path.abspath(path_to_video)
    metahome = get_video_meta_home(path_to_video)
    return os.path.join(metahome, "results") 

def get_video_length(vr, opt=None):
    if opt is None or opt.length is None:
        return len(vr)
    else:
        return opt.length

def data_labeling(opt):
    img_size = opt.img_size
    if opt.gpu is not None:
        device = f'cuda:{opt.gpu}'
        print("Using GPU ", opt.gpu)
    else:
        print("Using CPU as default ...")
        device = 'cpu'
    model = YOLOv5(model_type="yolov5x", long_side=img_size[0], device=device, fp16=False)
    vr = DecordVideoReader(opt.video, img_size=tuple(opt.img_size), gpu=opt.gpu, offset=opt.offset)
    if opt.length is not None:
        vr_length = min(len(vr), opt.length)
    else:
        vr_length = len(vr)
    # vr_length = 16
    batch_size = opt.batch_size
    results = []
    for i in tqdm(range(0, vr_length, batch_size)):
        batch_ids = [idx for idx in range(i, min(vr_length, i+batch_size))]
        # print("Batchs : ", batch_ids)
        frames = vr.get_batch(batch_ids)
        # print("DEBUG frame: ", frames)
        results.extend(model.infer(frames))
    # print("DEBUG TYPE: ", f'type={type(results[0])}, value={results}')
    for_save = np.array(results)
    # print("WHY??? ", f'type={type(for_save)}, value={for_save}')
    output_path = get_label_path(opt.video)
    np.save(output_path, np.array(results))
    

def get_partition_size(vr, opt):
    length = get_video_length(vr, opt)
    num_train = int(min(0.005 * length, 30000))
    num_valid = int(min(0.001 * length, 3000))
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
    print("Parition Size: ", f'train={num_train}, valid={num_valid}, test={length-num_valid-num_train}.')
    train = indices[:num_train]
    valid = indices[num_train:num_train+num_valid]
    test = indices[num_train+num_valid:]

    train.sort()
    test.sort()
    valid.sort()

    # compute the label weight for training CMDN
    # scores = lr.get_batch(train + valid)
    # hist = np.ones([opt.max_score]) * 10 # to smoothn the weight
    # for s in scores:
    #     hist[s] += 1

    # sample_max = hist.max()
    # weight = sample_max / hist

    train = np.array(train)
    valid = np.array(valid)
    test = np.array(test)
    # weight = np.array(weight)

    if save:
        metahome = get_video_meta_home(opt.video)
        split_path = os.path.join(metahome, "partition/")
        os.makedirs(split_path, exist_ok=True)
        np.save(os.path.join(split_path, "train_ids.npy"), train)
        np.save(os.path.join(split_path, "valid_ids.npy"), valid)
        np.save(os.path.join(split_path, "test_ids.npy"), test)
        # np.save(os.path.join(metahome, "score_weight.npy"), weight)

    return train, valid, test

if __name__ == "__main__":
    opt = get_options()

    vr = DecordVideoReader(opt.video, img_size=tuple(opt.img_size), offset=opt.offset)
    cached_gt_path = get_label_path(opt.video)
    if not os.path.exists(cached_gt_path):
        print("Cached Label doesn't exist, generating now...")
        data_labeling(opt)
    lr = CachedGTLabelReader(cached_gt_path, opt.offset)

    split_dataset(opt, vr, lr, True)
