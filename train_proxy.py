import os
import yaml
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from models import proxy
from dataloader.video_loader import DecordVideoReader,VideoLoader
from dataloader.label_loader import BinaryLabelReader
from os import path as osp
from utils.data_processing import get_video_meta_home, get_label_path, get_checkpoint_dir

def evaluate(model, val_loader, device, bbx_thresh = 0.2):
    ctx = torch.device(device)
    model.to(ctx)
    model.eval()

    num_pos = 0
    num_neg = 0

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

            tp += ((labels == 1) & (preds == labels)).sum()
            correct += (preds == labels).sum()
            total += len(labels)

    accuracy = correct / total
    precision = tp / num_pos
    print('Summary:')
    print(f'  positive: {num_pos}')
    print(f'  negative: {num_neg}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')


def train(model, lr, num_epochs, train_loader, ckpt_prefix, device):
    ctx = torch.device(device)
    model.to(ctx)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    #TODO: add lr scheduler

    for epoch in range(num_epochs):
        for images, labels in tqdm(train_loader, desc="Epoch {}/{}".format(epoch+1, num_epochs)):
            images = images.to(ctx)
            labels = labels.to(ctx)

            feat = model(images)
            loss = criterion(feat, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    os.makedirs(osp.dirname(ckpt_prefix), exist_ok=True)
    torch.save(model.state_dict(), '{}_final.pt'.format(ckpt_prefix))
    print('Checkpoint saved to {}_final.pt'.format(ckpt_prefix))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--video", type=str, default="data/videos/example.mp4", help="path to the video of interest")
    parser.add_argument("--length", type=int, default=None, help="specify the length of the video, full length by default")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--img_size", type=int, nargs=2, default=[416, 416])

    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)

    parser.add_argument("--proxy_train_model_type", type=str, default="tinyresnet18", choices=["tinyresnet18"])
    parser.add_argument("--proxy_train_num_classes", type=int, default=2)
    parser.add_argument("--proxy_pretrained_model", default=False, help="Whether to use pretrained model", action="store_true")
    parser.add_argument("--proxy_train_epochs", type=int, default=10)
    parser.add_argument("--proxy_train_batch_size", type=int, default=64)
    parser.add_argument("--proxy_train_lr", type=float, default=0.001)

    parser.add_argument("--test", default=False, help="Whether it is run for testing", action="store_true")
    opt, _ = parser.parse_known_args()
    opt.video = os.path.abspath(opt.video)
    print("OPTIONS: ", opt)

    random.seed(0)
    np.random.seed(opt.random_seed)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_type = opt.proxy_train_model_type
    num_classes = opt.proxy_train_num_classes
    pretrained = opt.proxy_pretrained_model
    batch_size = opt.proxy_train_batch_size
    lr = opt.proxy_train_lr
    num_epochs = opt.proxy_train_epochs

    if opt.gpu is None:
        device = "cpu"
    else:
        device = f'cuda:{opt.gpu}'

    cached_gt_path = get_label_path(opt.video)
    label_reader = BinaryLabelReader(cached_gt_path)

    video_reader = DecordVideoReader(
        video_file=opt.video,
        img_size=tuple(opt.img_size),
        gpu=opt.gpu
    )
    partiton_dir = osp.join(get_video_meta_home(opt.video), "partition")
    train_idxs = np.load(osp.join(partiton_dir, "train_ids.npy"), allow_pickle=True)
    val_idxs = np.load(osp.join(partiton_dir, "valid_ids.npy"), allow_pickle=True)

    train_loader = VideoLoader(video_reader, label_reader, train_idxs, batch_size, device, shuffle=True, enable_cache=True)
    val_loader = VideoLoader(video_reader, label_reader, val_idxs, batch_size, device, shuffle=False, enable_cache=True)

    ckpt_prefix = get_checkpoint_dir(opt.video)
    model = eval(f'proxy.{model_type}')(num_classes=num_classes, pretrained=pretrained)
    if opt.test:
        model.load_state_dict(
            torch.load('{}_final.pt'.format(ckpt_prefix), map_location=torch.device(device)))
    else:
        train(model, lr, num_epochs, train_loader, ckpt_prefix, device)
    evaluate(model, val_loader, device)
