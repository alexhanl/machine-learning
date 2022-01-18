# PyTorch ResNet50 image classification with Bitfusion

The test code is from the nvidia github repo as below.  I just take a snapshot here.  
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/


## 1. Download Imagenet Dataset
You have two options:
* Download the full imagenet dataset from http://image-net.org/download-images.  The dataset is more than 100 GB. But unfortunately, this site doesn't provide new user registration. However，you can still find ways to download it, such as BitTorrent.  
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

## Model Training and Inference

Download the pretrained model, which will be used in reference step. 

```
mkdir pretrained && cd pretrained
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/resnet50_pyt_amp/versions/20.06.0/zip -O resnet50_pyt_amp_20.06.0.zip
unzip resnet50_pyt_amp_20.06.0.zip
```

Let's start the container. 
```
docker run --rm -it -v /home/user/pretrained:/root/.cache/torch/hub/checkpoints -v /home/user/imagenette2-160:/imagenet --ipc=host --privileged --pid=host nvidia_resnet50:bitfusion
```

### Model Training 
In the container shell, execute the commandline. 
```
bitfusion run -n 1 -p 1 -- python ./main.py --arch resnet50 --epochs 3 --batch-size 256 --label-smoothing 0.1 --amp --static-loss-scale 256 --no-checkpoints /imagenet
```

### Model Inference
To simulate the batch inference
```
bitfusion run -n 1 -p 1 -- python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained --batch-size 256 /imagenet
```
To simulate the real time online inference (batch-size=1)
```
bitfusion run -n 1 -p 1 -- python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained --batch-size 1 /imagenet
```

## Experiment with native GPU
First of all, you need a VM with passthrough GPU.  In this document, we just use the bitfusion server VM.  

### Enable nvidia plugin in Bitfusion Server VM
The bitfusion server 4.0 is actually a Photon OS VM with docker CE installed. But there's no nvidia plugin, so we need to install the nvidia plugin for docker. This Photon OS kernerl is similar with CentOS 8.  So we follow the similar steps in: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

```
sudo systemctl start docker
sudo curl -s -L https://nvidia.github.io/nvidia-docker/centos8/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
sudo yum install nvidia-container-toolkit -y
sudo systemctl restart docker
sudo usermod -aG docker $USER
```

You need to logout and login again to use non-root user to run the docker commands, and use the following command to verify the installation. 
```
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Build the ResNet50 container images
Just repeat the steps 1，2，3 above. 

### Model Training

Download the pretrained resnet50 model, and start a container
```
docker run --rm --gpus "device=1" -it -v /home/customer/pretrained:/root/.cache/torch/hub/checkpoints -v /home/customer/imagenette2-160:/imagenet --ipc=host  nvidia_resnet50
```

Start the training process. 
```
python ./main.py --arch resnet50 --epochs 3 --batch-size 256 --label-smoothing 0.1 --amp --static-loss-scale 256 --no-checkpoints /imagenet
```

Start the inference process. 
Batch size is 256
```
python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained --batch-size 256 /imagenet
```
Batch size is 1
```
python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained --batch-size 1 /imagenet
```

## Bitfusion vs. native GPU: A Comparison
Apparently, the accuracy should be the same, so we just compare the throughput.  In this scenario, we compare the train/val.compute_ips, which means the the number of the images processed per second. 

The testbed has 2 esxi servers, each has 1 Bitfusion server with 2 V100 32GB GPUs passthroughed.  We also have several ubuntu 18.04 client VMs.  In this test, when we use bitfusion, we specify the --server_list to force the bitfusion client to connect to the bitfusion server on different ESXi host.  Thus the data has to be transfered by the cable.  We use 10G Ethernet and the MTU has been set to 9000 from end to end. 

Here's the test results. 


| test case | Training/Inference  | Model      | batch size | Bitfusion/Native | train/val.compute_ips (img/s) |
| --------- | ------------------  | --------   | ---------- | ---------------- | ----------------------------- |
| 1         | Training            | ResNet50   | 256        | Native           | 823.74                        |   
| 2         | Training            | ResNet50   | 256        | Bitfusion        | 800.49                        |          
| 3         | Inference           | ResNet50   | 256        | Native           | 1182.87                       |  
| 4         | Inference           | ResNet50   | 256        | Bitfusion        | 1159.00                       |  
| 5         | Inference           | ResNet50   | 1          | Native           | 122.37                        |  
| 6         | Inference           | ResNet50   | 1          | Bitfusion        | 76.18                         |  
| 7         | Inference           | resnext101 | 1          | Native           | 68.58                         |  
| 8         | Inference           | resnext101 | 1          | Bitfusion        | 47.41                         |  

From the above data, we have the findings. 
- When the batch size is 256, the performance degradation is minor, The bitfusion throughput is about about 97% of the native GPU.  (800.49/823.74 = 97.18%, 1159.00/1182.87 = 97.98%)

- When the batch size is 1 to simulate the real time inference, the performance degradation is noticeable. 
    - for ResNet50 model about 62.25% (76.18/122.37 = 62.25)
    - for resnext101 model about 69.13%  (47.41/68.58 = 69.13). 

From the inference scenario, we also find that the result of resnext101 is better than ResNet50, because resnext101 is larger and more complex. 

However, we can benefit from bitfusion, because actually in a ResNet50 inference, 2 GB is GPU memory is sufficient.   Therefore, the remaining 30 GB memory of V100 GPU can be used by other processes at the sametime. 
