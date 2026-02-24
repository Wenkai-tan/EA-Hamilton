# Elasticity‑Aware Neural Hamiltonian Fields for Mesh‑Free Hyperelastic Simulation

## Overview
This code provides the implementation for theory and experiments within "Elasticity‑Aware Neural Hamiltonian Fields for Mesh‑Free Hyperelastic Simulation".
The implementation is built upon PyTorch and kaolin libraries.

## Installation
conda create -n pienerf python=3.10
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html

## Citation

```
@software{KaolinLibrary,
      author = {Fuji Tsang, Clement and Shugrina, Maria and Lafleche, Jean Francois and Perel, Or and Loop, Charles and Takikawa, Towaki and Modi, Vismay and Zook, Alexander and Wang, Jiehan and Chen, Wenzheng and Shen, Tianchang and Gao, Jun and Jatavallabhula, Krishna Murthy and Smith, Edward and Rozantsev, Artem and Fidler, Sanja and State, Gavriel and Gorski, Jason and Xiang, Tommy and Li, Jianing and Li, Michael and Lebaredian, Rev},
      title = {Kaolin: A Pytorch Library for Accelerating 3D Deep Learning Research},
      date = {2024-11-20},
      version = {0.17.0},
      url={\url{https://github.com/NVIDIAGameWorks/kaolin}}
}
```