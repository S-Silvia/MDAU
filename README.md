# MDAU
## Instruction
This work proposes an instance-level attribute update method called Multi-Domain Feature-Enhanced Attribute Updater (MDAU) for zero-shot image classification.
![alt text](fig2.jpg)
## Installation
python==3.6  
clip==1.0  
h5py==3.1.0  
matplotlib==3.3.4  
numpy==1.19.5  
opencv_python==4.6.0.66  
Pillow==10.3.0  
scikit_image==0.17.2
scikit_learn==1.5.0
scipy==1.5.4  
skimage==0.0  
torch==1.10.2+cu111
torchvision==0.11.3+cu111  
tqdm==4.64.0  
## File
The supporting files required to run the code are available for download at the following links  
### Pretrained parameters for Resnet101
Pretrained parameters for Resnet101 are stored in the folder "pretrained_models": https://drive.google.com/drive/folders/1lRajXvGzFOV7NnvQiGsIkjYZ8BhwBnLY?usp=drive_link 
### Checkpoints for MDAU
You can also reproduce our SoTA results directly on the test set. We have stored the trained checkpoints in the folder "out". You just need to set the parameter "only_evaluate" to True and change the path parameter "resume". 
#### out:
https://drive.google.com/drive/folders/1nnfs5UatdOP3Dh74dURz82FD_EYHGL_5?usp=drive_link
## How to run
After everything is installed and configured successfully, you can run the code on each of the three datasets by simply running the three .sh files.
