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
export PYTHONPATH=\$PYTHONPATH:/mnt/svas
```

## Run experiments

Now you aleady launched a svas container, you can run our experiments inside the container.

### Investigation Experiments

Here are some scripts to help you check the throughput of data access, models. You can open it and run each command inside it to get the evalutation results. Or you can just use `bash` to run them directly (but it will cost more time, while you don't need to wait for finish because we only cares about throughput here).

``` Bash
cat experiments/data_access.sh # evalute data access
cat model_throughput.sh # evalute models' throughput
```


### End to end experiments.

First, you need to train a proxy model. However, before we get it, we should firstly split our dataset to get training set, validation set, and test set. This is part of the data processing. 

``` Bash
# you can use "--vide xxxx" to specify video to process. By default we will use our example vide: data/videos/example.mp4
# U can also specify how to split the dataset, using options "--num_train" and "--num_valid", for more information, you can refer to utils/parser.py
python3 utils/data_processing.py
```

Then we start to train our proxy with scripts `train_proxy.py`. The trained proxy model will be stored along with the video data automatically.

``` Bash
# For options, refer to the train_proxy.py, --video can specify video files, etc.
python3 train_proxy.py
```

Now we can run our selection query with the following command:

``` Bash

python3 dist_select.py --chunk_size 640 --diff_delay 10 --diff_thresh 1e-5 --gpu 0 --num_gpus 1

```

For a specific video, you may need to change the parameters according to the characteristic of video.
