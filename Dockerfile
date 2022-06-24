FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
# RUN apt-get install -y wget
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i cuda-keyring_1.0-1_all.deb

# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# RUN apt-get update -y
# RUN apt-get -y --purge remove "*cublas*" "cuda*" "nsight*" 
# RUN apt-get install -y wget

# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
# RUN dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
# RUN apt-get -y install gnupg2
# RUN apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
# RUN apt-get update
# RUN apt-get -y install cuda


RUN apt update
RUN apt install -y wget software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.8 python3-pip python3.8-dev

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.8 get-pip.py

RUN pip3.8 install torch==1.9.0 torchvision
RUN pip3.8 install einops shapely timm yacs tensorboardX ftfy prettytable pymongo transformers
RUN pip3.8 install transformers

RUN apt install -y build-essential

COPY ./ /code
WORKDIR /code

RUN python3.8 setup.py build develop --user
# RUN mkdir MODEL

RUN wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth -O MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth

RUN pip3.8 install opencv-python nltk inflect scipy pycocotools
RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN python3.8 test.py