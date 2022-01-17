# PyTorch ResNet50 image classification with Bitfusion

The test code is from the nvidia github repo as below.  I just take a snapshot here.  
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/


## TL;DR





## Download Imagenet Dataset
You have several options:
1. Download the full imagenet dataset from http://image-net.org/download-images.  The dataset is very large. But unfortunately, this site doesn't provide new user registration.  You can still find ways to download it, such as from BitTorrent.  
2. Download imagenet small size subset from： https://github.com/fastai/imagenette.  In this scenario, we use the Imagenette 160 px. 

## Build the base resnet50 container image i.e. without bitfusion 

### Clone the repository
```
git clone https://github.com/alexhanl/machine-learning
cd machine-learning/PyTorch/ConvNets
```

### Build the container
```
docker build . -t nvidia_resnet50
```

### Performance of training
#### native mode
```
docker run --rm --gpus "device=1" -it -v /home/customer/pretrained:/root/.cache/torch/hub/checkpoints -v /home/customer/imagenette2-160:/imagenet --ipc=host  nvidia_resnet50
```

```
python ./main.py --arch resnet50 --epochs 3 --batch-size 256 --label-smoothing 0.1 --amp --static-loss-scale 256 --no-checkpoints /imagenet
```

#### with bitfusion

```
docker run --rm -it -v /home/user/pretrained:/root/.cache/torch/hub/checkpoints -v /home/user/imagenette2-160:/imagenet --ipc=host --privileged --pid=host nvidia_resnet50:bitfusion
```
```
bitfusion run --server_list '10.1.202.20:56001' -n 1 -p 1 -- python ./main.py --arch resnet50 --epochs 3 --batch-size 256 --label-smoothing 0.1 --amp --static-loss-scale 256 --no-checkpoints /imagenet
```



### Performance of inference
#### native mode
```
docker run --rm --gpus "device=1" -it -v /home/customer/pretrained:/root/.cache/torch/hub/checkpoints -v /home/customer/imagenette2-160:/imagenet --ipc=host  nvidia_resnet50

python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained --pretrained --batch-size 256 /imagenet
python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained --pretrained --batch-size 1 /imagenet

python ./main.py --arch resnext101-32x4d --evaluate --epochs 1 --pretrained --pretrained --batch-size 1 /imagenet
```

#### with bitfusion

```
docker run --rm -it -v /home/user/pretrained:/root/.cache/torch/hub/checkpoints -v /home/user/imagenette2-160:/imagenet --ipc=host --privileged --pid=host nvidia_resnet50:bitfusion
```
```
bitfusion run --server_list '10.1.202.20:56001' -n 1 -p 1 -- python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained --pretrained --batch-size 256 /imagenet

bitfusion run --server_list '10.1.202.20:56001' -n 1 -p 1 -- python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained --pretrained --batch-size 1 /imagenet


bitfusion run --server_list '10.1.202.20:56001' -n 1 -p 1 -- python ./main.py --arch resnext101-32x4d --evaluate --epochs 1 --pretrained --pretrained --batch-size 1 /imagenet
```






ping -M do -s 8500 10.1.202.20

ping -M do -s 8500 10.1.202.21


sudo ifconfig net1 mtu 9000 up

sudo ifconfig veth28476bf mtu 9000 up

ifconfig eth0 mtu 9000 up




find . -type f -name "*.yml" -exec sed -i 's/github.com/github.com.cnpmjs.org/g' {} \;

