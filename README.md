# WS-DAFNet
This is the code of Weakly Supervised Dual-Attention Fusion Network for Action Unit Detection.

## Environment
The code is developed using python 3.8 and PyTorch 1.6.0 on Ubuntu 18.04. NVIDIA GPUs are needed.

## Quick start
### Installation
1. Clone this repo, and we'll call the directory that you cloned as ${ROOT}.
2. Install dependencies:
   ```
   pytorch 1.6.0
   torchvision 0.7.0
   numpy
   tqdm
   opencv-python
   scipy
   dlib
   matplotlib
   ```
3. Init result and dataset directory:
   ```
   mkdir result 
   mkdir dataset
   ```
   
   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── dataset
   ├── result
   └── source
   ```
4. Download dlib landmark detection model from [here](https://pan.baidu.com/s/1XiLL4S7Q23Tzpic09eWOIw) (passward:q8uc) and download pre-trained VGG from [here](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)
   ```
   ${ROOT}
    `-- data
        |-- vgg19-dcbb9e9d.pth
        |-- data
            |-- reflect_66.mat
            |-- shape_predictor_68_face_landmarks.dat
   ```
   
### Data preparation
1. Put [BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) dataset into the folder "dataset"
   ```
   ${ROOT}
   `-- dataset
      `-- BP4D
         |-- F001_T1_2440.jpg
         |-- F001_T1_2441.jpg
         |-- ... 
   ```
2. Align and crop the images in the dataset.
   ```
   cd data 
   python image_crop.py
   ```
3. Generate cropped image annotation file.
   ```
   cd data 
   python dataste_extract.py
   ```
### Training and Testing
1. Train on BP4D in three folds.
   ```
   python Train_WS-DAFNet.py --version='Train_DAFNet_fold1' --listtrainpath='source/BP4D_crop_new_tr1.txt' --listtestpath='source/BP4D_crop_new_ts1.txt' --weight_path='source/BP4D_crop_new_tr1_weight.txt'
   python Train_WS-DAFNet.py --version='Train_DAFNet_fold2' --listtrainpath='source/BP4D_crop_new_tr2.txt' --listtestpath='source/BP4D_crop_new_ts2.txt' --weight_path='source/BP4D_crop_new_tr2_weight.txt'
   python Train_WS-DAFNet.py --version='Train_DAFNet_fold3' --listtrainpath='source/BP4D_crop_new_tr3.txt' --listtestpath='source/BP4D_crop_new_ts3.txt' --weight_path='source/BP4D_crop_new_tr3_weight.txt'
   ```
2. Test the models saved in training phase:
   ```
   python Test_WS-DAFNet.py --version='Train_DAFNet_fold1' --listtestpath='source/BP4D_crop_new_ts1.txt' --resume=1
   ```
