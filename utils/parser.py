import os
import argparse

def get_options():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--video", type=str, default="data/videos/example.mp4", help="path to the video of interest")
    parser.add_argument("--length", type=int, default=None, help="specify the length of the video, full length by default")
    parser.add_argument("--diff_thres", type=float, help="threshold of the difference detector")
    parser.add_argument("--num_train", type=float, default=0.005, help="training set size of the CMDN")
    parser.add_argument("--num_valid", type=float, default=0.001, help="validation set size of the CMDN")
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
    opt.video = os.path.abspath(opt.video)
    return opt