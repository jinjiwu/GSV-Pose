# GSV-Pose
PyTorch implementation of the paper: Pose Estimation Method Based on Geometric Similarity Voting

![pipeline](pic/pipeline.jpg)


## Required environment

- Ubuntu 20.04
- Python 3.8 
- Pytorch 1.10.1
- CUDA 11.3

## Installing

- 使用conda创建虚拟环境
`conda create -n gsvpose python=3.8`, Install the main requirements in 'requirement.txt'.
- Install [Detectron2](https://github.com/facebookresearch/detectron2).
- 提供了自己修改版的 [GeoTransformer](https://github.com/qinzheng93/GeoTransformer.git), Download GeoTransformer's pre-trained weights in this [link](https://github.com/qinzheng93/GeoTransformer/releases)

## Data Preparation
- Download the data provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019) ([real_train](http://download.cs.stanford.edu/orion/nocs/real_train.zip), [real_test](http://download.cs.stanford.edu/orion/nocs/real_test.zip),
[ground truths](http://download.cs.stanford.edu/orion/nocs/gts.zip) and structurelized as [DualPoseNet](https://github.com/Gorilla-Lab-SCUT/DualPoseNet) like: 
```
data
├── CAMERA
│   ├── train
│   └── val
├── real
│   ├── train
│   └── test
├── gts
│   ├── val
│   └── real_test
└── obj_models
    ├── train
    ├── val
    ├── real_train
    └── real_test
```
Run the following scripts to prepare training instances:

```bash
cd provider
python training_data_prepare.py
```

## Training

Specify the dataset directory and run the following command. `SAVE_DIR` is the directory to save the trained model. 
```shell
python -m engine.train --dataset_dir ./data --dataset Real --model_save SAVE_DIR
```

More detailed configurations are in 'config/config.py'.

Our trained model can be download from this [link](https://drive.google.com/drive/folders/1GrCYZIJPPrtozOUS8MHI0Y1dbxn6Kl2-?usp=sharing). 下载之后放到`pretrain`目录下。

## Evaluation
对NOCS数据集中的val数据集进行评估，使用以下命令。
```shell
python -m evaluation.evaluate --dataset_dir YOUR_DATA_DIR --resume 1 --resume_model pretrain/gpv_pose_update.pth --model_save eval_logs
```

我们提供了可供测试的真实数据，我们的真实数据组织方式和[NOCS](https://github.com/hughw19/NOCS_CVPR2019)一致。测试命令为
```shell
python -m evaluation.evaluate --dataset_dir ./our_rgbd --resume 1 --resume_model pretrain/gpv_pose_update.pth --model_save eval_logs --our_camK True --draw_gt False
```

## Acknowledgment
Our implementation leverages the code from [GeoTransformer](https://github.com/qinzheng93/GeoTransformer), [GPV-Pose](https://github.com/lolrudy/GPV_Pose) [3dgcn](https://github.com/j1a0m0e4sNTU/3dgcn), [FS-Net](https://github.com/DC1991/FS_Net),
