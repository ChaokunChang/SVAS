import os
import yaml
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from models import proxy
from core.label_reader import NumpyLabelReader
from dataloader.video_loader import DecordVideoReader,VideoLoader
from os import path as osp


def evaluate(model, val_loader, device):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config for task')
    parser.add_argument('--test', action='store_true', help='evaluate checkpoint')
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    model_type = config['PROXY']['MODEL_TYPE']
    num_classes = config['PROXY']['NUM_CLASSES']
    pretrained = config['PROXY']['TRAIN']['PRETRAINED']

    video_reader = DecordVideoReader(
        video_file=config['SOURCE']['VIDEO'],
        img_size=config['PROXY']['IMG_SIZE'],
        device=config['DEVICE'],
        num_threads=4
    )

    train_idxs = np.load(config['SPLIT']['TRAIN_IDX_DIR'])
    val_idxs = np.load(config['SPLIT']['VAL_IDX_DIR'])

    device = config['DEVICE']
    batch_size = config['PROXY']['TRAIN']['BATCH_SIZE']
    lr = config['PROXY']['TRAIN']['LR']
    num_epochs = config['PROXY']['TRAIN']['EPOCHS']
    cache_dirs = config['PREPARE']['CACHE_DIR']

    for i, cache_dir in enumerate(cache_dirs):
        label_reader = NumpyLabelReader(numpy_dir=cache_dir)

        ckpt_prefix = config['PROXY']['TRAIN']['CKPT_PREFIX'][i]

        train_loader = VideoLoader(video_reader, label_reader, train_idxs, batch_size, device, shuffle=True, enable_cache=True)
        val_loader = VideoLoader(video_reader, label_reader, val_idxs, batch_size, device, shuffle=False, enable_cache=True)

        model = eval(f'proxy.{model_type}')(num_classes=num_classes, pretrained=pretrained)
        if args.test:
            model.load_state_dict(
                torch.load('{}_final.pt'.format(ckpt_prefix), map_location=torch.device(device)))
        else:
            train(model, lr, num_epochs, train_loader, ckpt_prefix, device)
        evaluate(model, val_loader, device)
