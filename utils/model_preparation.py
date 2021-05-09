import os
import yaml
import random
import argparse
import numpy as np
import torch
import tqdm
from os import path as osp
from dataloader.video_loader import DecordVideoReader
from utils.data_processing import *

def evaluate_video(models, mids, dataloader):
    for model in models:
        model.eval()
    mdn_loss_list = [0] * len(models)
    num_loss_list = [0] * len(models)
    mean_dict_list = [{} for m in models]
    var_dict_list = [{} for m in models]
    for imgs, scores in tqdm.tqdm(dataloader, desc="Detecting objects"):
        scores_cpu = scores.cpu()
        with torch.no_grad():
            for i, (mean_dict, var_dict, model) in enumerate(zip(mean_dict_list, var_dict_list, models)):
                _, mdn_output = model(imgs, scores=scores)
                pi, sigma, mu = mdn_output[2], mdn_output[3], mdn_output[4]
                mdn_loss_list[i] += mdn_output[1].item()
                num_loss_list[i] += 1
                mean = (mu * pi).sum(-1)
                var = ((sigma**2 + mu**2 - mean.unsqueeze(-1)**2) * pi).sum(-1)
                for i in range(len(mean)):
                    lab = scores_cpu[i].item()
                    if lab not in mean_dict:
                        mean_dict[lab] = [mean[i].item()]
                        var_dict[lab] = [var[i].item()]
                    else:
                        mean_dict[lab] += [mean[i].item()]
                        var_dict[lab] += [var[i].item()]

    for i, (mean_dict, var_dict) in enumerate(zip(mean_dict_list, var_dict_list)):
        print("profile of Model %d" % mids[i])
        print('K N Mean Var MSE')
        se_dict = dict()
        count_dict = dict()
        keys = [int(k) for k in mean_dict.keys()]
        keys.sort()
        for k in keys:
            samples = len(mean_dict[k])
            se = ((np.array(mean_dict[k]) - k) ** 2).sum()
            mean = np.mean(mean_dict[k])
            var = np.mean(var_dict[k])
            se_dict[k] = se
            count_dict[k] = samples
            print('{}: {} {:.2f} {:.2F} {:.2f}'.format(k, samples, mean, var, se / samples))
        se = np.array(list(se_dict.values()))
        count = np.array(list(count_dict.values()))
        print('MSE: {:.2f}'.format(se.sum() / count.sum()))
        print('NLL: {:.2f}'.format(mdn_loss_list[i] / num_loss_list[i]))
        mdn_loss_list[i] /= num_loss_list[i]
    return mdn_loss_list

def train_models(epochs, model_configs, mids, train_dataloader, valid_dataloader, weight, checkpoint_dir):
    for i, model_config in enumerate(model_configs):
        for module_def in model_config:
            if module_def["type"] == "hmdn":
                print("Model_%d: M: %s, H: %s, eps: %s" % (mids[i], module_def["M"], module_def["num_h"], module_def["eps"]))
                break

    models = [Darknet(model_config, weight).to(config.device) for model_config in model_configs]
    parameters = []
    for model in models:
        model.apply(weights_init_normal)
        model.load_darknet_weights("weights/yolov3-tiny.weights")
        model.train()
        parameters += model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        num_batchs = 0
        for imgs, scores in tqdm.tqdm(train_dataloader, desc="epoch %d/%d" % (epoch+1, epochs)):
            loss = 0.0
            for model in models:
                _, mdn_output = model(imgs, scores=scores)
                wta_loss = mdn_output[0]
                mdn_loss = mdn_output[1]
                if epoch < 5:
                    loss += wta_loss + 0.0 * mdn_loss
                else:
                    loss += 0.5 * wta_loss + mdn_loss

                # mdn metric
                total_loss += mdn_loss.item()
                num_batchs += 1
                model.seen += imgs.size(0)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("mdn_loss: %.3f" % (total_loss / num_batchs))

    nlls = evaluate_video(models, mids, valid_dataloader)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"cmdn_{mids[i]}.pth"))
    return np.array(nlls)

def train_cmdn(opt, vr, lr, train_idxs, valid_idxs, score_weight):
    torch.manual_seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(opt.random_seed)

    # Get data configuration
    model_group_config = parse_model_config(config.cmdn_config)
    model_configs = parse_model_group(model_group_config)
    checkpoint_dir = get_checkpoint_dir(opt)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_video_loader = VideoLoader(vr, train_idxs, lr, batch_size=opt.cmdn_train_batch)
    valid_video_loader = VideoLoader(vr, valid_idxs, lr, batch_size=opt.cmdn_train_batch)

    nlls = np.zeros([len(model_configs)])
    model_batch = len(model_configs)
    for i in range(int(math.ceil(len(model_configs) / model_batch))):
        model_config_batch = model_configs[i*model_batch: (i+1) * model_batch]
        nlls[i*model_batch: (i+1)*model_batch] = train_models(opt.cmdn_train_epochs, model_config_batch, range(i*model_batch, (i+1)*model_batch), train_video_loader, valid_video_loader, score_weight, checkpoint_dir)
    best_model = np.argmin(nlls)

    print("best_model: %d, nll: %0.3f" % (best_model, nlls[best_model]))
    best_model_path = os.path.join(checkpoint_dir, f"cmdn_{best_model}_best.pth")
    os.rename(os.path.join(checkpoint_dir, f"cmdn_{best_model}.pth"), best_model_path)
    return best_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--video", type=str, default="videos/archie.mp4", help="path to the video of interest")
    parser.add_argument("--length", type=int, default=None, help="specify the length of the video, full length by default")
    parser.add_argument("--diff_thresh", type=float, help="threshold of the difference detector")
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
    parser.add_argument("--udf_batch", type=int, default=8)
    parser.add_argument("--skip_train_cmdn", default=False, action="store_true")
    parser.add_argument("--skip_cmdn_scan", default=False, action="store_true")
    parser.add_argument("--skip_topk", default=False, action="store_true")
    parser.add_argument("--save", default=False, help="save intermediate results", action="store_true")
    opt, _ = parser.parse_known_args()
