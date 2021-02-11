## Setting up the docker container

To setup the docker container you first have to install docker and the nvidia toolkit. Please follow [the instructions here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

To build the docker container run 
```
sudo docker build --build-arg UID=$(id -u) -t deep-dance:v0 .
```

To run the container do 
```
sudo docker run --gpus all -it -p 6666:8888 -v /path to folder /core:/home/deep-dance/core deep-dance:v0 bash
```

-p 6666:8888 exposes the jupyter port and is optional. To start a jupyter notebook server in the container do
```
jupyter notebook --ip 0.0.0.0 --no-browser
```
you can then access the notebook server on your host via a browser at
```
localhost:8888/tree
```
for the correct token check the terminal output in docker container.

The -v option mounts the ../core folder to the container. 


Note: You still have to download the pretrained models for VideoPose3D as described in the main readme file in point 4.
