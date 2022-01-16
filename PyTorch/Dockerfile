#ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.03-py3
#ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.07-py3
ARG FROM_IMAGE_NAME=nvidia_resnet50:latest

FROM ${FROM_IMAGE_NAME}
LABEL maintainer="someone@something.com"

RUN apt-get update

# for noninteractive set the timezone
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install -y tzdata
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN dpkg-reconfigure -f noninteractive tzdata

# install necessary tools for toubleshooting, comment out the below line in production
RUN apt-get install -y iputils-ping iproute2 net-tools 


WORKDIR /workspace/bitfusion

# You need to download the correct deb from vmware repo https://packages.vmware.com/bitfusion/ubuntu/
COPY bitfusion-client-ubuntu2004_4.0.1-5_amd64.deb  bitfusion-client.deb
# Install the bitfusion client
RUN apt-get install -y ./bitfusion-client.deb

# Copy the token files
RUN mkdir -p /root/.bitfusion
COPY ./.bitfusion/client.yaml /root/.bitfusion/client.yaml
RUN chmod 400 /root/.bitfusion/client.yaml 
COPY ./.bitfusion/servers.conf /etc/bitfusion/servers.conf
RUN mkdir -p /etc/bitfusion/tls
COPY ca.crt /etc/bitfusion/tls/ca.crt

RUN usermod -aG bitfusion root

WORKDIR /workspace/rn50
