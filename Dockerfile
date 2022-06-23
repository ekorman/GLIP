FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

# RUN apt-get install -y wget
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update -y
# RUN apt-get --purge remove "*cublas*" "cuda*" "nsight*" 
RUN apt-get install -y wget

# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
# RUN dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
# RUN apt-get -y install gnupg2
# RUN apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
# RUN apt-get update
# RUN apt-get -y install cuda-10-2

COPY ./ /code
WORKDIR /code

RUN pip install torch==1.9.0 torchvision  torchaudio
RUN pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo transformers
RUN pip install transformers
RUN apt install -y build-essential
RUN python setup.py build develop --user
# RUN mkdir MODEL

RUN wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth -O MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth

RUN pip install opencv-python nltk inflect scipy pycocotools
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python test.py