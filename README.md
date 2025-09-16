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
The model is placed in the directory $root_path/results/$model_dir after training. To evaluate it, first set the checkpoint path in the corresponding .sh file (e.g., test_a.sh).
To evaluate CAVIGATE on Activitynet Captions:
```bash
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh
./test_a.sh
 ```

To evaluate CAVIGATE on TVR:
```bash
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh
./test_t.sh
 ```
## CheckPoint
We provide the trained checkpoints on the ActivityNet Captions and TVR datasets [here](https://drive.google.com/drive/folders/1F_XLEwJMO7oorxRoSakQGRhxyGnh_u-o?usp=drive_link), with their expected performance metrics as follows:
### RoBERTa + I3D + ResNet
| Dataset              | R@1  | R@5  | R@10 | R@100 | SumR |
|----------------------|------|------|------|------|------|
| ActivityNet Captions | 9.7 | 28.9 | 41.4 | 78.7 | 158.7 |
| TVR                  | 18.1 | 40.7 | 51.7 | 87.3 | 197.7 |

### CLIP-ViT/B32
| Dataset              | R@1  | R@5  | R@10  | MedR | MnR   |
|----------------------|------|------|-------|------|-------|
| ActivityNet Captions | 15.0 | 36.7 | 49.8  | 83.0 | 184.5 |
| TVR                  | 26.4 | 51.0 | 62.4  | 91.5 | 231.2 |

