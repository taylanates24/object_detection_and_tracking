FROM nvcr.io/nvidia/pytorch:22.12-py3

ARG CUDA=11.6
ARG TORCH_VERSION=1.8.0
ARG TORCHVISION_VERSION=0.9.0

ENV DEBIAN_FRONTEND=noninteractive

### update apt and install libs
RUN apt-get update &&\
    apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx git wget libssl-dev libopencv-dev libspdlog-dev p7zip-full --no-install-recommends &&\
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
RUN pip install --upgrade pip
RUN pip install -U openmim
RUN pip3 install torch torchvision torchaudio
### install mmcv
RUN pip3 install pytest-runner
RUN pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${TORCH_VERSION}/index.html

### git mmdetection
RUN pip install --upgrade pip
RUN git clone --depth=1 https://github.com/open-mmlab/mmdetection.git /root/space/mmdetection
RUN pip3 install numpy
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
RUN pip install Pillow \
      matplotlib

RUN pip install --no-cache-dir \
    grad-cam==1.4.6 \
    tensorboard==2.10.1 \
    fiftyone==0.17.2 \
    fast-coco-eval \
    opencv-python==4.6.0.66 \
    scipy==1.9.3 \
    scikit-learn==1.1.3 \
    scikit-image==0.19.3 \
    torch-optimizer==0.3.0 \
    pytorch-optimizer==2.2.0 \
    clearml==1.9.1 \
    setuptools==58.2.0 \
    numpy==1.23.5
RUN pip install tensorrt
RUN pip install scikit-learn==0.22.2.post1
WORKDIR /root/space

CMD [ "--help" ]
ENTRYPOINT [ "mmdet2trt" ]