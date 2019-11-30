Flux.jl-DCGAN
=============

DCGAN implementation using [Flux.jl](https://fluxml.ai/Flux.jl/stable/)


## Environment
Google Cloud Platform, n1-standard-8 + 1 x NVIDIA Tesla K80  
Docker version 19.03.4, build 9013bf583a

```shell
(v1.3) pkg> st
    Status `~/.julia/environments/v1.3/Project.toml`
  [fbb218c0] BSON v0.2.4
  [31c24e10] Distributions v0.21.9
  [5789e2e9] FileIO v1.1.0
  [587475ba] Flux v0.10.0
  [f67ccb44] HDF5 v0.12.5
  [6218d12a] ImageMagick v0.7.5
  [916415d5] Images v0.18.0
  [4138dd39] JLD v0.9.1
```

## Run
It takes a few minites...
```shell
sudo docker build -t matsueushi/flux . 
sudo docker run --name flux --gpus all -it -v $PWD:/tmp -w /tmp matsueushi/flux:latest /bin/bash
julia mnist-dcgan.jl
```

## Result
30 epochs (14,000 iterations)  
![Animation](https://github.com/matsueushi/fluxjl-dcgan/blob/media/media/anim.gif)

![Loss](https://github.com/matsueushi/fluxjl-dcgan/blob/media/media/loss.png)

## References
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [GANについて概念から実装まで　～DCGANによるキルミーベイベー生成～](https://qiita.com/taku-buntu/items/0093a68bfae0b0ff879d)  
- [はじめてのGAN](https://elix-tech.github.io/ja/2017/02/06/gan.html)
- [GANによる二次元美少女画像生成](https://medium.com/@crosssceneofwindff/ganによる二次元美少女画像生成-33047bb586a0)
- [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)

Flux implementation
- [Added Condtional GAN and DCGAN tutorial #111](https://github.com/FluxML/model-zoo/pull/111)
  
Implementations of other frameworks
 - [Tensorflow tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)
 - [znxlwm/tensorflow-MNIST-GAN-DCGAN](https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN)
 - [PyTorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
 - [jacobgil/keras-dcgan](https://github.com/jacobgil/keras-dcgan)