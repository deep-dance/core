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

## Nvidia GPU

TBD
