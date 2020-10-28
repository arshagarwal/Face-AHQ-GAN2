## Steps to run using docker using Nvidia gpus:
1. Install docker using [docker installation guide](https://docs.docker.com/engine/install/ubuntu/).
2. Install nvidia-docker2 using [nvidia-docker2 installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

**Note: Ensure to check the latest versions of nvidia cuda runtime and coroborrate with pytorch cuda requirements, this guide was uploaded on 4/9/2020.**

3. Build the image using `docker build -f docker -t slimgan:gpu`.
4. Run the Container using `docker run --gpus all --shm-size 10G -it slimgan:gpu`.
5.Run the `nvidia-smi` to check if gpus work.   
6. Run the following commands to train and test the network: 
```
# Train
python main.py --img_dir ../data/celeba_hq/train --iters 20000 --img_size 128 --batch_size 16 --gpus 0,1 --c_dim 2 --num_workers 5

# Test
To be Added ..
```
