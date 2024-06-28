# Real-Time Compressed Sensing for Joint Hyperspectral Image Transmission and Restoration for CubeSat


## Data

The following directory stores different type of data, original, noised, masked version, respectively. We provide these data and onnx model to better valid proposed RTCS's reproducibility.

Note: noised version means the original HSI image with adding AWGN noises. Masked version means the original HSI image with adding CM mask (Completely Missing).

./data/original

./data/AWGN_noisy

./data/masked_CM

## Enviroment

python==3.8.18

pytorch==1.11.0 

cudatoolkit=11.3 

onnx==1.14.1

onnxruntime==1.16.1

## Inference

To valid RTCS, please run the notebook: inference.ipynb
