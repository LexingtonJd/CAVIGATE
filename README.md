# CAVIGATE
This repository is the official PyTorch implementation of our paper CAVIGATE: Caption and Audio-Guided Video Representation Learning with Gated Attention for Partially Relevant Video Retrieval

## Table of Contents
- [Environments](#environments)
- [Datasets](#Datasets)
- [Train](#Train)
- [Evaluation](#Evaluation)
## Environments
- Python 3.8  
- PyTorch 1.9.0  
- torchvision 0.10.0  
- TensorBoard 2.6.0  
- tqdm 4.62.0  
- easydict 1.9  
- h5py 2.10.0  
- CUDA 11.1

We used Anaconda to set up a deep learning workspace that supports PyTorch. Run the following commands to create the environment and install the required packages:
```bash
conda create --name CAIVGATE python=3.8
conda activate CAVIGATE
git clone https://github.com/LexingtonJd/CAVIGATE.git # 
cd CAVIGATE
pip install -r requirements.txt
conda deactivate
 ```

## Datasets
You can download the complete feature sets for ActivityNet Captions and TV Show Retrieval (TVR) from [here](https://drive.google.com/drive/folders/11dRUeXmsWU25VMVmeuHc9nffzmZhPJEj). These feature sets were generously provided by [MS-SL](https://github.com/HuiGuanLab/ms-sl), and we gratefully acknowledge their contribution.

After preparing the dataset, extract it and set the data path in the corresponding .sh file (e.g., do_activitynet.sh).

## Train
To train CAVIGATE on Activitynet Captions:
```bash
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh
./do_activitynet.sh
 ```

To train CAVIGATE on TVR:
```bash
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh
./do_tvr.sh
 ```
## Evaluation
The model is placed in the directory $root_path/results/$model_dir after training. To evaluate it, please run the following script:
