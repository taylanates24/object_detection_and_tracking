FROM nvcr.io/nvidia/pytorch:22.12-py3

ARG CUDA=11.8
ARG TORCH_VERSION=1.8.0
ARG TORCHVISION_VERSION=0.9.0

ENV DEBIAN_FRONTEND=noninteractive

### update apt and install libs
RUN apt-get update &&\
    apt-get install -y vim cmake libsm6 libxext6 libxrender-dev libgl1-mesa-glx git

### create folder
#RUN mkdir ~/space &&\
#    mkdir /root/.pip

### set pip source
# COPY ./pip.conf /root/.pip

### pytorch
#RUN pip3 install torch==${TORCH_VERSION}+cu${CUDA//./} torchvision==${TORCHVISION_VERSION}+cu${CUDA//./} -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

### install mmcv
RUN pip3 install pytest-runner
RUN pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${TORCH_VERSION}/index.html

### git mmdetection
RUN git clone --depth=1 https://github.com/open-mmlab/mmdetection.git /root/space/mmdetection
RUN pip3 install numpy==1.20
### install mmdetection
RUN cd /root/space/mmdetection &&\
    pip3 install -r requirements.txt &&\
    python3 setup.py develop

### git amirstan plugin
RUN git clone --depth=1 https://github.com/grimoire/amirstan_plugin.git /root/space/amirstan_plugin &&\
    cd /root/space/amirstan_plugin &&\
    git submodule update --init --progress --depth=1

### install amirstan plugin
RUN cd /root/space/amirstan_plugin &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make -j10

ENV AMIRSTAN_LIBRARY_PATH=/root/space/amirstan_plugin/build/lib

### git torch2trt_dynamic
RUN git clone --depth=1 https://github.com/grimoire/torch2trt_dynamic.git /root/space/torch2trt_dynamic

### install torch2trt_dynamic
RUN cd /root/space/torch2trt_dynamic &&\
    python3 setup.py develop

### git mmdetection-to-tensorrt
RUN git clone --depth=1 https://github.com/grimoire/mmdetection-to-tensorrt.git /root/space/mmdetection-to-tensorrt

### install mmdetection-to-tensorrt
RUN cd /root/space/mmdetection-to-tensorrt &&\
    python3 setup.py develop

RUN python -m pip install --upgrade pip setuptools
RUN pip install tensorflow==2.2.0 \
      Pillow \
      matplotlib \
      moviepy \
      scipy \ 
      opencv-python \
      object-detection

RUN pip install scikit-learn==0.22.2 \
                opencv-python==4.6.0.66 \
                opencv==4.6.0 \ 
                opencv-python-headless==4.6.0.66
RUN apt install -y libgtk2.0-dev pkg-config

WORKDIR /root/space

CMD [ "--help" ]
ENTRYPOINT [ "mmdet2trt" ]