## Steps to run using docker using Nvidia gpus:
1. Install docker using [docker installation guide](https://docs.docker.com/engine/install/ubuntu/).
2. Install nvidia-docker2 using [nvidia-docker2 installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
### Note: Ensure to check the latest versions of nvidia cuda runtime and coroborrate with pytorch cuda requirements , this guide was uploaded on 4/9/2020.
3. Build the image using `docker build -f docker -t slimgan:gpu`.
4. Run the Container using `docker run --gpus all --shm-size 10G -it slimgan:gpu`.
5.Run the `nvidia-smi` to check if gpus work.   
6. Run thr command ``python main.py --rafd_image_dir Big_slim --num_iters 30000 --sample_step 500 --c_dim 2 --log_step 100 --model_save_step 5000 --batch_size 64 --n_critic 2 --rafd_crop_size 128 --image_size 128 --resume_iters 20000 --num_workers 5`.


## Steps to run the code on Google colab (just 3 lines of code):
0. Clone the code by running the command `!git clone https://username:password@github.com/arshagarwal/C_slim_gan.git -b arsh_spectral` . 
   **Replace the username and password with your github username and password respectively.**
1. Run the command `cd C_slim_gan` to navigate to the **C_slim_gan** directory.
2. Run the command `!bash import_dataset.sh` to import the **Big slim dataset**. 
3. Run the command `! python main.py --rafd_image_dir Big_slim --num_iters 20000 --sample_step 500 --c_dim 2 --log_step 100 --model_save_step 5000 --batch_size 64 --n_critic 2 --rafd_crop_size 128 --image_size 128 ` to train on the **Big_slim_dataset**. 
**Alternatively to train on custom dataset replace the `slim_dataset/Train_dataset` string with the path of your custom dataset.**  
  For further options such as **number of epochs, batch_size etc** refer **main.py**
### Note: After running step 3 after every sample stepth iteration generated samples will be stored in stargan/samples folder for performance evaluation.

## For Continuing training from 20000 iteration to 30000 iteration follow:
1. `python main.py --rafd_image_dir Big_slim --num_iters 30000 --sample_step 500 --c_dim 2 --log_step 100 --model_save_step 5000 --batch_size 64 --n_critic 2 --rafd_crop_size 128 --image_size 128 --resume_iters 20000`.
