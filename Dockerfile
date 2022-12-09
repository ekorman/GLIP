FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

RUN apt update
RUN apt install -y wget software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.8 python3-pip python3.8-dev

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.8 get-pip.py

RUN pip3.8 install -U setuptools
RUN pip3.8 install torch==1.9.0 numpy

RUN apt install -y build-essential

RUN mkdir /code

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./glip /code/glip
COPY ./setup.py /code/setup.py

WORKDIR /code

RUN python3.8 setup.py build develop --user
# RUN pip3.8 install -v .
