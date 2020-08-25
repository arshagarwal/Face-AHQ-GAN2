## Steps to run the code on Google colab (just 3 lines of code):
0. Clone the code by running the command `!git clone https://username:password@github.com/arshagarwal/C_slim_gan.git -b arsh_spectral` . 
   **Replace the username and password with your github username and password respectively.**
1. Run the command `cd C_slim_gan` to navigate to the **C_slim_gan** directory.
2. Run the command `!bash import_dataset.sh` to import the **Big slim dataset**. 
3. Run the command `! python main.py --rafd_image_dir Big_slim --num_iters 20000 --sample_step 500 --c_dim 2 --log_step 100 --model_save_step 5000 --batch_size 64 --n_critic 2 --rafd_crop_size 128 --image_size 128 ` to train on the **Big_slim_dataset**. 
**Alternatively to train on custom dataset replace the `slim_dataset/Train_dataset` string with the path of your custom dataset.**  
  For further options such as **number of epochs, batch_size etc** refer **main.py**
## Note: After running step 3 after every sample stepth iteration generated samples will be stored in stargan/samples folder for performance evaluation.  
