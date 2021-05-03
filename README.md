# SVAS
Scalable Video Analytic System, Course project for CSCI5570 Large Scale Data Processing System.

# Reproduce

``` Bash
git clone https://github.com/ChaokunChang/SVAS svas
docker pull tigerchang/svas:latest
```

``` Bash
cd svas
docker run -i -v $PWD:/mnt/svas tigerchange/svas /bin/sh
cd /mnt/svas
export PYTHONPATH=\$PYTHONPATH:/mnt/svas
```

``` Bash
bash experiments/data_access.sh # evalute data access
bash model_throughput.sh # evalute model throughput
python models/detecor.py # evalute Yolov5
```

