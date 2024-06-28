# Real-Time Compressed Sensing for Joint Hyperspectral Image Transmission and Restoration for CubeSat (TGRS 2024)

## [[Paper Link (IEEE)]](https://ieeexplore.ieee.org/document/10474407) [[Paper Link (arXiv)]](https://arxiv.org/abs/2404.15781) [[Model Zoo (.pth, .onnx)]](https://drive.google.com/drive/folders/18UAGosITMAch5f4TwaPuyuj5xYJpmZWK?usp=sharing) 

[Chih-Chung Hsu](https://cchsu.info/), Chih-Yu Jian, Eng-Shen Tu, [Chia-Ming Lee](https://ming053l.github.io/), Guan-Lin Chen

Advanced Computer Vision LAB, National Cheng Kung University

## Overview

Coming Soon. 

## Environment

- [PyTorch >= 1.7](https://pytorch.org/)
- CUDA >= 11.2
- python==3.8.18
- pytorch==1.11.0 
- cudatoolkit=11.3 
- onnx==1.14.1
- onnxruntime==1.16.1

### Installation
```
git clone https://github.com/ming053l/RTCS.git
conda create --name rtcs python=3.8 -y
conda activate rtcs
# CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
cd RTCS
pip install -r requirements.txt
```

## How To Test

```
python test.py
```

## How To Train

Due to the consideration about patent, now we don't tend to publish training file and model at this repository. If you want to access or train RTCS with your own dataset, please contact the author.

In order to make sure reproduciblity to public, we provide a pre-trained models (ONNX version) and code, please see [this link]().

## Citations

If our work is helpful to your reaearch, please kindly cite our work. Thank!

#### BibTeX
    @misc{hsu2024realtimecompressedsensingjoint,
        title={Real-Time Compressed Sensing for Joint Hyperspectral Image Transmission and Restoration for CubeSat}, 
        author={Chih-Chung Hsu and Chih-Yu Jian and Eng-Shen Tu and Chia-Ming Lee and Guan-Lin Chen},
        year={2024},
        eprint={2404.15781},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2404.15781}, 
    }
    @ARTICLE{10474407,
        author={Hsu, Chih-Chung and Jian, Chih-Yu and Tu, Eng-Shen and Lee, Chia-Ming and Chen, Guan-Lin},
        journal={IEEE Transactions on Geoscience and Remote Sensing}, 
        title={Real-Time Compressed Sensing for Joint Hyperspectral Image Transmission and Restoration for CubeSat}, 
        year={2024},
        volume={62},
        number={},
        pages={1-16},
        keywords={Hyperspectral imaging;Real-time systems;Sensors;Decoding;Compressed sensing;Training;Task analysis;Compressed sensing;deep learning (DL);hyperspectral image (HSI);hyperspectral restoration;real-time applications},
        doi={10.1109/TGRS.2024.3378828}
    }


## Contact
If you have any question, please email zuw408421476@gmail.com to discuss with the author.
