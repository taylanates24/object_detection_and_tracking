FROM nvcr.io/nvidia/pytorch:22.12-py3


ARG CUDA_VERSION=11.3
ARG PYTHON_VERSION=3.8
ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0
ARG MMCV_VERSION=1.7.1
ARG MMDET_VERSION=2.28.1

ENV TORCH_CUDA_ARCH_LIST=${CUDA_VERSION}

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

### pytorch
RUN /opt/conda/bin/conda install pytorch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} -c pytorch
ENV CUDA_VERSION=11.3
ENV TORCH_CUDA_ARCH_LIST=${CUDA_VERSION}
RUN conda install cudatoolkit=${CUDA_VERSION} -c pytorch

### install mmcv-full
RUN /opt/conda/bin/pip install mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA_VERSION//./}/torch${TORCH_VERSION}/index.html

# Copy tensorrt to conda environment
RUN cp -r /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* /opt/conda/lib/python${PYTHON_VERSION}/site-packages/

# ------------- MMDET  -------------

WORKDIR /
RUN wget -O mmdet.tar.gz https://github.com/open-mmlab/mmdetection/archive/refs/tags/v${MMDET_VERSION}.tar.gz && \
    mkdir mmdetection && \
    tar -zxvf mmdet.tar.gz --directory mmdetection && \
    cd /mmdetection && \
    mv */* . && \
    rm /mmdet.tar.gz

WORKDIR /mmdetection
RUN pip install --no-cache-dir -r requirements/build.txt
RUN pip install --no-cache-dir -e .



# -------------MM TENSORRT------------
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

RUN pip install --no-cache-dir \
    opencv-python==4.6.0.66 \
    scipy==1.9.3 \
    scikit-learn==0.22.2 \
    scikit-image==0.19.3 \
    setuptools==58.2.0 \
    numpy==1.23.5

WORKDIR /workspace