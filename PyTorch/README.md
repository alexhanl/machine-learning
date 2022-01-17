# PyTorch ResNet50 image classification with Bitfusion

The test code is from the nvidia github repo as below.  I just take a snapshot here.  
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/


## 1. Download Imagenet Dataset
You have two options:
* Download the full imagenet dataset from http://image-net.org/download-images.  The dataset is very large. But unfortunately, this site doesn't provide new user registration. However，you can still find ways to download it, such as BitTorrent.  
* Download imagenet small size subset from： https://github.com/fastai/imagenette.  

In this document, we use the 2nd option, and use the smallest dataset: Imagenette 160 px. 
```
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -zxf imagenette2-160.tgz
```

## 2. Clone the repository
```
git clone https://github.com/alexhanl/machine-learning
cd ./machine-learning/PyTorch/ConvNets
```

## 3. Build the resnet50 container image
```
docker build . -t nvidia_resnet50
```

## 4. Build the resnet50 bitfusion container image

### Prepare the bitfusion related files
We assume that the VM itself has already been bitfusion-enabled. 

```
cd ~/machine-learning/PyTorch
wget https://packages.vmware.com/bitfusion/ubuntu/20.04/bitfusion-client-ubuntu2004_4.0.1-5_amd64.deb
sudo cp /etc/bitfusion/tls/ca.crt .
```

### Build the nvidia_resnet50:bitfusion image

```
cd ~/machine-learning/PyTorch
docker build . -t nvidia_resnet50:bitfusion
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

python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained --batch-size 256 /imagenet
python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained --batch-size 1 /imagenet

python ./main.py --arch resnext101-32x4d --evaluate --epochs 1 --pretrained --pretrained --batch-size 1 /imagenet
```

#### with bitfusion

```
docker run --rm -it -v /home/user/pretrained:/root/.cache/torch/hub/checkpoints -v /home/user/imagenette2-160:/imagenet --ipc=host --privileged --pid=host nvidia_resnet50:bitfusion
```
```
bitfusion run --server_list '10.1.202.20:56001' -n 1 -p 1 -- python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained --batch-size 256 /imagenet

bitfusion run --server_list '10.1.202.20:56001' -n 1 -p 1 -- python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained --batch-size 1 /imagenet


bitfusion run --server_list '10.1.202.20:56001' -n 1 -p 1 -- python ./main.py --arch resnext101-32x4d --evaluate --epochs 1 --pretrained --batch-size 1 /imagenet
```

bitfusion run --server_list '10.1.202.20:56001' -n 1 -m 1024 -- python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained --batch-size 1 /imagenet




ping -M do -s 8500 10.1.202.20

ping -M do -s 8500 10.1.202.21


sudo ifconfig net1 mtu 9000 up

sudo ifconfig veth888603c mtu 9000 up

ifconfig eth0 mtu 9000 up




find . -type f -name "*.yml" -exec sed -i 's/github.com/github.com.cnpmjs.org/g' {} \;

