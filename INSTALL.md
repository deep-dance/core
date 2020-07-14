# Prerequisites

## AMD GPU

Running on AMD GPU's needs a functioning [ROCm stack](https://rocmdocs.amd.com/en/latest/) in order to support GPU accelaration
for ML libraries such as pytorch and tensorflow. Builds of these libraries are available with the ROCm backend enabled.

### Install Ubuntu 18.04.4

ROCm is not supposed to work with the prorietary AMD driver. After the operation system is installed, verify the open-source kernel module `amdgpu` is loaded:

```
kmod list | grep amdgpu
```

### Setup ROCm

- Follow the installation guide: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#ubuntu
- (Optional) Install ROCm validation suite

### Setup ML libraries

#### pytorch

##### Docker

- Follow the installation guide: https://rocmdocs.amd.com/en/latest/Deep_learning/Deep-learning.html#recommended-install-using-published-pytorch-rocm-docker-image
- Install missing OpenCV and requirements
- Create new image from current container using `docker commit`

##### From source

###### 1. Clone forked PyTorch repository on the host

```
cd ~
git clone https://github.com/zirkular/pytorch.git
cd pytorch
git submodule init
git submodule update
```

###### 2. Build PyTorch docker image

```
cd pytorch/docker/caffe2/jenkins
sudo ./build.sh py3.6-gcc-rocmdeb-conda-opencv-ubuntu18.04
```

###### 3. Start a docker container using the new image:

```
sudo docker run -it -v $HOME:/data --privileged --rm --device=/dev/kfd --device=/dev/dri --group-add video <image_id>
```

Reference: https://rocmdocs.amd.com/en/latest/Deep_learning/Deep-learning.html#option-2-install-using-pytorch-upstream-docker-file.

## Nvidia GPU

**Note:** CUDA 10.0+ requires at least CMake 3.12.2+
```
wget -c "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin"
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

```
CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
wget -c ${CUDNN_URL}
sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig
```

### GPU monitoring

```
watch -n 0.5 nvidia-smi
```

# High-level libraries

## detectron2

### Installation

Our [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
has step-by-step instructions that install detectron2.
The [Dockerfile](docker)
also installs detectron2 with a few simple commands.

#### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.4
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- [pycocotools](https://github.com/cocodataset/cocoapi). Install it by `pip install pycocotools>=2.0.1`.
- OpenCV, optional, needed by demo and visualization


#### Build Detectron2 from Source

gcc & g++ ≥ 5 are required. [ninja](https://ninja-build.org/) is recommended for faster build.
After having them, run:
```
# Install from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# Or if you are on macOS
CC=clang CXX=clang++ python -m pip install ......
```

Please refer to [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) for further information in the installation process.

## OpenPose

```
sudo apt install libhdf5-dev \
    protobuf-compiler \
    libopencv-dev \
    libatlas-base-dev \
    libboost-all-dev \
    libopencv-dev
```

### AMD GPU

```
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose
mkdir build && cd build
cmake .. -DGPU_MODE=OPENCL
```

