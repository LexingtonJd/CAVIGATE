# CAVIGATE
This repository is the official PyTorch implementation of our paper CAVIGATE: Caption and Audio-Guided Video Representation Learning with Gated Attention for Partially Relevant Video Retrieval

## Table of Contents
- [Environments](#environments)
- [Datasets](#Datasets)

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
conda create --name DLDKD python=3.8 -y
conda activate DLDKD
pip install -r requirements.txt


## Datasets
You can download the complete feature sets for ActivityNet Captions and TV Show Retrieval from [here](https://drive.google.com/drive/folders/11dRUeXmsWU25VMVmeuHc9nffzmZhPJEj). These feature sets were generously provided by [MS-SL](https://github.com/HuiGuanLab/ms-sl), and we gratefully acknowledge their contribution.

