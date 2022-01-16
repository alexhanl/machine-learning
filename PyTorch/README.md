# PyTorch ResNet50 image classification with Bitfusion

The test code is from the nvidia github repo as below.  I take a snapshot here. 
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/

## Test Data
You have several choices:
1. Imagenet small subset download： https://github.com/fastai/imagenette



ping -M do -s 8500 10.1.202.20

ping -M do -s 8500 10.1.202.21


sudo ifconfig net1 mtu 9000 up

sudo ifconfig ens160 mtu 9000 up

ifconfig eth0 mtu 9000 up


docker run --gpus all -it -v /home/customer/imagenette2-160:/imagenet --ipc=host --privileged --pid=host nvidia_resnet50
docker run --gpus all -it -v /home/customer/imagenette2:/imagenet --ipc=host --privileged --pid=host nvidia_resnet50

docker run -it -v /home/user/pretrained:/root/.cache/torch/hub/checkpoints -v /home/user/imagenette2-160:/imagenet --ipc=host --privileged --pid=host nvidia_resnet50:bitfusion
docker run -it -v /home/user/imagenette2:/imagenet --ipc=host --privileged --pid=host nvidia_resnet50:bitfusion


python ./main.py --arch resnet50  --batch-size 256 --label-smoothing 0.1 --amp --static-loss-scale 256 --no-checkpoints /imagenet


bitfusion run --server_list '10.1.202.20:56001' -n 1 -p 1 -- python ./main.py --arch resnet50  --batch-size 256 --label-smoothing 0.1 --amp --static-loss-scale 256 --no-checkpoints /imagenet



bitfusion run --server_list '10.1.202.21:56001' -n 1 -p 1 -- python ./main.py --arch resnet50 --evaluate --epochs 1 --pretrained --pretrained -b 256 /imagenet


