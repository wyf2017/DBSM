<<<<<<< HEAD
﻿# Pytorch implementation of the several Deep Binocolur Stereo Matching(DBSM) Network
citation{

(1)Wang Yufeng,Wang Hongwei,Yu Guang,Yang Mingquan,Yuan Yuwei,Quan Jicheng. Stereo Matching Algorithm Based on Three-Dimensional Convolutional Neural Network[J]. Acta Optica Sinica, 2019, 39(11): 1115001

王玉锋,王宏伟,于光,杨明权,袁昱纬,全吉成. 基于三维卷积神经网络的立体匹配算法[J]. 光学学报, 2019, 39(11): 1115001

url=http://www.opticsjournal.net/Articles/Abstract?aid=OJc47fda4ba4b4ce65

(2)Wang Yufeng,Wang Hongwei,Yu Guang,Yang Mingquan,Yuan Yuwei,Quan Jicheng. Real-time stereo matching with a hierarchical refinement [J]. Acta Optica Sinica, 2020, 40(09): 0915002.

王玉锋,王宏伟,于光,杨明权,袁昱纬,全吉成. 渐进细化的实时立体匹配算法[J]. 光学学报, 2020, 40(09): 0915002.

url=http://www.opticsjournal.net/Articles/HPAbstract?manu_number=g191640

}
## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)

## Introduction

Pytorch implementation of the several Deep Binocolur Stereo Matching Network
DispNet/DispNetC, WSMCnet, MBFnet, MTLnet

## Usage

### Dependencies

- [Python-3.7](https://www.python.org/downloads/)
- [PyTorch-1.1.0](http://pytorch.org)
- [torchvision-0.3.0](http://pytorch.org)
- [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [SceneFlow dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [Middeval dataset](http://vision.middlebury.edu/stereo/)
- [Eth3d dataset](https://www.eth3d.net/datasets#low-res-two-view)

Usage of KITTI, SceneFlow, Middeval and Eth3d dataset in [stereo/dataloader/README.md](stereo/dataloader/README.md)

### Train
Reference to [demos/train_sfk_MTL.sh](demos/train_sfk_MTL.sh).

### Submission
Reference to [demos/submission_all.sh](demos/submission_all.sh) and [demos/submission.sh](demos/submission.sh).

### Pretrained Model

[WSMCnet](https://pan.baidu.com/s/1HtfUADZe8R4s2sV2cQW2qA) ( Extraction code： jz05)
[MBFnet](https://pan.baidu.com/s/1itwxOxwzgM0Rsk93sEpwIQ)


## Results on [KITTI 2015 leaderboard](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

| Method | D1-all (All) | D1-all (Noc)| Runtime (s) |Environment|
|---|---|---|---|---|
| [WSMCnet(our)](http://www.opticsjournal.net/Articles/Abstract?aid=OJc47fda4ba4b4ce65) | 2.13 % | 1.85 % | 0.39 | GTX 1070 (pytorch) |
| [PSMNet](https://arxiv.org/abs/1803.08669) | 2.32 % | 2.14 % | 0.41 | Titan Xp (pytorch)|
| [iResNet-i2](https://arxiv.org/abs/1712.01039) | 2.44 % | 2.19 % | 0.12 | Titan XP (Caffe)|
| [GC-Net](https://arxiv.org/abs/1703.04309) | 2.87 % | 2.61 % | 0.90 | Titan XP (TensorFlow)|
| [MC-CNN](https://github.com/jzbontar/mc-cnn) | 3.89 % | 3.33 % | 67 | Titan XP (Torch7)|
|---|---|---|---|---|
| [DeepPruner(fast)](http://arxiv.org/abs/1909.05845) | 2.59 % | 2.35 % | 0.06 | Titan XP (Caffe) |
| [MBFnet(our)](http://www.opticsjournal.net/Articles/HPAbstract?manu_number=g191640) | 2.96 % | 2.54 % | 0.05 | RTX 2070 (pytorch) |
| [DispNetC](http://arxiv.org/abs/1512.02134) | 4.32 % | 4.05 % | 0.06 | Titan XP (Caffe) |
| [MADnet](http://arxiv.org/abs/1810.05424) | 4.66 % | 4.27 % | 0.02 | GTX 1080Ti (tensorflow) |


## Contacts
wangyf_1991@163.com

Any discussions or concerns are welcomed!
=======
# DBSM
Deep Binocular Stereo Matching(DBSM)
>>>>>>> ced4619a84de3f8b904a4cbd0e93aa3ad1aedb0e
