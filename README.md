# SVAS
Scalable Video Analytic System, Course project for CSCI5570 Large Scale Data Processing System.

# Reproduce

## Set up
First, you should download our source code into your machine where there is a nvidia GPU installed(because we will use GPU).

To reduce the complexity of setting up the environment, we provided a docker image in docker hub. Please download it and run all experiments inside the containers based on our image.

Here is the commands for it. 
``` Bash
git clone https://github.com/ChaokunChang/SVAS svas
docker pull tigerchang/svas:latest
```

``` Bash
# ready to launch svas container
cd svas
bash docker/launch_svas.sh

# inside the container
# go to the working directory
cd /mnt/svas
# Setting up environment varibale
export PYTHONPATH=$PYTHONPATH:/mnt/svas
```

## Run experiments

Now you aleady launched a svas container, you can run our experiments inside the container.

### Dataset

We prepared three data set for reader: 

1. example.mp4 : A camera recording on the crossroad. There will be some cars cross it.
2. demo.mp4 : A new report recording where there will be journerlist, cars, etc.
3. zongyi.mp4 : A video where a lot of person playing together.

Notice that: As the `demo.mp4` and `zongyi.mp4` is too large (1G ~ 2G), we didn't put the video inside this repository. You can download these two video in the following Onece Drive [link](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155155266_link_cuhk_edu_hk/EsSffSbZqiBNpQiLZB1TqXIBbLNtX4BThA-MfrLYfJpFtw?e=mTji2L). *You can also use your own videos.*

### Investigation Experiments

Here are some scripts to help you check the throughput of data access, models. You can open it and run each command inside it to get the evalutation results. Or you can just use `bash` to run them directly (but it will cost more time, while you don't need to wait for finish because we only cares about throughput here).

``` Bash
cat experiments/data_access.sh # evalute data access
cat experiments/model_throughput.sh # evalute models' throughput
```

## Preprocessing

Before tun our queries, we must first pre-process our data and prepare our proxy model. The code are available in `preprocessing.py`.

Here is an example to pre-process example.mp4 and train a proxy model for task "select the frames where there are at least 1 car". 

``` Bash
python3 preprocessing.py --video data/videos/zongyi.mp4 \
        --task person-n2 --target_object person --target_object_count 2 \
        --gpu 0
# Or
python3 preprocessing.py --video data/videos/example.mp4 \
        --task car-n1 --target_object car --target_object_count 1 \
        --gpu 0
```

There will be 4x stages during the above procedure: (1) Data Labeling, which run oracle model on the full video to get the grount truth. This can be ignored in production. (2) Split Data, which split the data into train set, validation set and test set, the train set will be used to train our proxy model (3) Train our proxy model with train set. (4) Evaluate our proxy model with validation set.

*The first 3 stages can be skipped by option `--skip_labeling`, `--skip_split`, `skip_proxy_train`, respectively. We also support configuration for each stage, please refer to the `utils/parser.py` for details.*

After pre-processing, the ground truth label, the pre-trained proxy model, and ids for test set, will all be stored in folder `data/videos/example_car-n2/` automatically.

### End to end experiments

We support a lot of options to run a select query. For details, you can refer to the file `utils/parser.py`. Here is an simple example.

``` Bash
python3 dist_select.py  --video data/videos/zongyi.mp4 --length 141431 \
                        --task person-n2 --target_object person --target_object_count 2 \
                        --chunk_size 640 --diff_delay 10 --diff_thresh 1e-5 --num_gpus 4
# Or
python3 dist_select.py  --video data/videos/example.mp4 --length 108000 \
                        --task car-n1 --target_object car --target_object_count 1 \
                        --chunk_size 640 --diff_delay 10 --diff_thresh 1e-5 --num_gpus 4

```

Different configuration may have different acceleration, currently we doesn't support auto tuning, which will be future work.
