# Real-Time Compressed Sensing for Joint Hyperspectral Image Transmission and Restoration for CubeSat (TGRS 2024)

## [[Paper Link (IEEE)]](https://ieeexplore.ieee.org/document/10474407) [[Paper Link (arXiv)]](https://arxiv.org/abs/2404.15781) [[Model Zoo (.pth, .onnx)]](https://drive.google.com/drive/folders/18UAGosITMAch5f4TwaPuyuj5xYJpmZWK?usp=sharing) 

[Chih-Chung Hsu](https://cchsu.info/), Chih-Yu Jian, Eng-Shen Tu, [Chia-Ming Lee](https://ming053l.github.io/), Guan-Lin Chen

Advanced Computer Vision LAB, National Cheng Kung University

## Overview

<img src=".\figure\main_flowchart.png" width="800"/>


<img src=".\figure\realtime.png" width="700"/>

The RTCS network features a simplified architecture that reduces the required training samples and allows for easy implementation on integer-8-based encoders, facilitating rapid compressed sensing for stripe-like HSI, which exactly matches the moderate design of miniaturized satellites on push broom scanning mechanism. In contrasts, optimization-based models that demand high-precision floating-point operations, making them difficult to deploy on edge devices. 

Our encoder employs an integer-8-compatible linear projection for stripe-like HSI data transmission, ensuring real-time compressed sensing. Furthermore, based on the novel two-streamed architecture, an efficient HSI restoration decoder is proposed for the receiver side, allowing for edge-device reconstruction without needing a sophisticated central server. This is particularly crucial as an increasing number of miniaturized satellites necessitates significant computing resources on the ground station. 

## The run-time and computational complexity comparison of the decoders of the proposed RTCS and other methods under different computing platforms
| Method      | \#Params | FLOPs   | Run-time | PSNR↑ / RMSE↓ / SAM↓ | - | - | - |
|-------------|----------|---------|--------------|----------------------|---------------|----------------|---------------------|
| (DEVICE)      |    -     |     GPU (GTX 1660) |  -                        |     -              |  CPU (i9-9900) | Jetson TX2 GPU | Denver 2 64-bit CPU |
| AAHCS       | -        | -       | -            | 32.978 / 107.917 / 4.287  | 274.15        | -              | -                   |
| SpeCA       | -        | -       | -            | 14.585 / 274.191 / 23.659 | 0.1872        | -              | -                   |
| SPACE       | -        | -       | -            | 24.304 / 479.792 / 6.483  | **274.15**    | -              | -                   |
| SpeCA       | -        | -       | -            | 16.830 / 386.398 / 17.094 | 0.257         | -              | -                   |
| DCSN        | 11.95M   | 0.866G  | 0.035        | 32.703 / 70.471 / 1.893   | 0.431         | 0.033          | 3.85                |
| DCSN-TVM    | 11.95M   | 0.866G  | 0.023        | 32.703 / 70.471 / 1.893   | 0.383         | 0.025          | 2.119               |
|-------------|----------|---------|--------------|----------------------|---------------|----------------|---------------------|
| RTCS        | 6.291M   | 0.297G  | 0.023        | 36.537 / 43.681 / 1.176   | 0.392         | 0.024          | 3.497               |
| RTCS-TVM    | 6.291M   | 0.297G  | 0.011        | 36.537 / 43.681 / 1.176   | 0.243         | 0.019          | 2.011               |



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

## Demo

<img src=".\figure\demo.png" width="700"/>
The false color representation of the visualized hyperspectral image of the (a) ground truth, and the reconstructed counterpart under completely missing (CM) bands from 50-80 using (b) the proposed RTCS, (c) SpeCA, (d) DCSN, (e) H-LSS, (f) TenTV, and (g) AAHCS

In order to show RTCS's great inference speed and performance, we provide a pre-trained models (ONNX version) and code, please see [this link](https://github.com/ming053l/RTCS/tree/main/Demo_ONNX). 

And you can also find `.pth` weights on our [Model Zoo](https://drive.google.com/drive/folders/18UAGosITMAch5f4TwaPuyuj5xYJpmZWK?usp=sharing) 

## Citations

If our work is helpful to your research, please kindly cite our work. Thank!

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
