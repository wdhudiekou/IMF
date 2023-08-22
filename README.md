# IMF
 
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/wdhudiekou/IMF/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.6.0-%237732a8)](https://pytorch.org/)

### Improving Misaligned Multi-modality Image Fusion with One-stage Progressive Dense Registration [Under Review]

By Di Wang, Jinyuan Liu, Long Ma, Risheng Liu, and Xin Fan

<div align=center>
<img src="https://github.com/wdhudiekou/IMF/blob/main/Fig/network.png" width="80%">
</div>
<div align=center>
<img src="https://github.com/wdhudiekou/IMF/blob/main/Fig/MPDR.png" width="80%">
</div>

## Updates  
[2023-08-25] Our paper is available online! [[arXiv version](https://arxiv.org/pdf/2205.11876.pdf)]  


## Requirements
- CUDA 10.1
- Python 3.6 (or later)
- Pytorch 1.6.0
- Torchvision 0.7.0
- OpenCV 3.4
- Kornia 0.5.11

## Data preparation
1. You can generate misaligned infrared-visible images for training/testing by
    ```python
       cd ./data
       python generate_affine_deform_data.py
   
In 'Trainer/train_reg.py', deformable infrared images are generated in real time by default during training.

2. You can obtain self-visual saliency maps for training the fusion process of infrared and visible images by
    ```python
       cd ./data
       python get_svs_map_softmax.py
   
 ## Get start
1. You can use the pseudo infrared images [[link](https://pan.baidu.com/s/1M79RuHVe6udKhcJIA7yXgA) code: qqyj] generated by the CPSTN proposed by [UMF](https://github.com/wdhudiekou/UMF-CMGR) to train/test our C-MPDR:
    ```python
       cd ./Trainer
       python train_reg.py

       cd ./Test
       python test_reg.py
   
  Please download the [pretrained model]() (code: ) of the registration network MPDR.