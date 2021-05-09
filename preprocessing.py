import os
import math
import yaml
from tqdm import tqdm
import random
import argparse
import numpy as np
import torch
from os import path as osp

from dataloader.video_loader import DecordVideoReader, VideoLoader
from dataloader.label_loader import BinaryLabelReader
from models.detector import YOLOv5
from models import proxy
from models.yolo_utils import get_obj_id
from utils.parser import *

def data_labeling(opt, vr, device):
    img_size = opt.img_size
    model_type = opt.labeling_model
    model = YOLOv5(model_type=model_type,
                   long_side=img_size[0], device=device, fp16=False)
    if opt.length is not None:
        vr_length = min(len(vr), opt.length)
    else:
        vr_length = len(vr)
    batch_size = opt.labeling_batch
    results = []
    for i in tqdm(range(0, vr_length, batch_size), desc=f'Labeling data with {model_type}'):
        batch_ids = [idx for idx in range(i, min(vr_length, i+batch_size))]
        frames = vr.get_batch(batch_ids)
        results.extend(model.infer(frames))
    for_save = np.array(results)
    output_path = get_label_path(opt)
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
    print("Parition Size: ",
          f'train={num_train}, valid={num_valid}, test={length-num_valid-num_train}.')
    train = indices[:num_train]
    valid = indices[num_train:num_train+num_valid]
    test = indices[num_train+num_valid:]

    train.sort()
    test.sort()
    valid.sort()

    train = np.array(train)
    valid = np.array(valid)
    test = np.array(test)

    path_train, path_valid, path_test = get_partition_paths(opt)
    np.save(path_train, train)
    np.save(path_valid, valid)
    np.save(path_test, test)

    return train, valid, test


def evaluate(model, val_loader, device, bbx_thresh=0.2):
    ctx = torch.device(device)
    model.to(ctx)
    model.eval()

    num_pos = 0
    num_neg = 0

    pred_pos = 0
    pred_neg = 0

    tp = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(ctx)
            labels = labels.to(ctx)

            feat = model(images)
            _, preds = torch.max(feat, 1)

            num_pos += (labels == 1).sum()
            num_neg += (labels == 0).sum()

            pred_pos += (preds == 1).sum()
            pred_neg += (preds == 0).sum()

            tp += ((labels == 1) & (preds == labels)).sum()
            correct += (preds == labels).sum()
            total += len(labels)

    accuracy = correct / total
    precision = tp / pred_pos
    recall = tp / num_pos
    print('Summary:')
    print(f'  positive: {num_pos}')
    print(f'  negative: {num_neg}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')


def train(model, lr, num_epochs, train_loader, ckpt_final_path, device):
    ctx = torch.device(device)
    model.to(ctx)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # TODO: add lr scheduler

    for epoch in range(num_epochs):
        for images, labels in tqdm(train_loader, desc="Proxy Train - Epoch {}/{}".format(epoch+1, num_epochs)):
            images = images.to(ctx)
            labels = labels.to(ctx)

            feat = model(images)
            loss = criterion(feat, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), ckpt_final_path)
    print('Checkpoint saved to {}'.format(ckpt_final_path))


if __name__ == "__main__":
    opt = get_preprocessing_options()
    print("running optioins: ", opt)

    if opt.gpu is None:
        device = "cpu"
    else:
        device = f'cuda:{opt.gpu}'

    video_reader = DecordVideoReader(
        opt.video, gpu = opt.gpu, img_size=tuple(opt.img_size), offset=opt.offset)
    
    gt_label_path = get_label_path(opt)
    if not opt.skip_labeling and not os.path.exists(gt_label_path):
        print("Cached Label doesn't exist, generating now...")
        data_labeling(opt, video_reader, device)

    target_obj_id = get_obj_id(opt.target_object)
    label_reader = BinaryLabelReader(gt_label_path, opt.offset, target_obj_id, opt.target_object_thresh, opt.target_object_count)

    if not opt.skip_split:
        split_dataset(opt, video_reader, label_reader, True)

    path_train, path_valid, path_test = get_partition_paths(opt)
    train_idxs = np.load(path_train, allow_pickle=True)
    val_idxs = np.load(path_valid, allow_pickle=True)


    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    model_type=opt.proxy_train_model
    num_classes=opt.proxy_train_num_classes
    pretrained=opt.proxy_pretrained_model
    batch_size=opt.proxy_train_batch_size
    leaning_rate=opt.proxy_train_lr
    num_epochs=opt.proxy_train_epochs

    train_loader=VideoLoader(video_reader, label_reader, train_idxs,
                             batch_size, device, shuffle=True, enable_cache=True)
    val_loader=VideoLoader(video_reader, label_reader, val_idxs,
                           batch_size, device, shuffle=False, enable_cache=True)

    ckpt_dir=get_checkpoint_dir(opt)
    ckpt_final_path=osp.join(ckpt_dir, "proxy_final.pt")
    model=eval(f'proxy.{model_type}')(
        num_classes=num_classes, pretrained=pretrained)
    if opt.skip_proxy_train and os.path.exists(ckpt_final_path):
        model.load_state_dict(
            torch.load(ckpt_final_path, map_location=torch.device(device)))
    else:
        train(model, leaning_rate, num_epochs,
              train_loader, ckpt_final_path, device)
    evaluate(model, val_loader, device)
