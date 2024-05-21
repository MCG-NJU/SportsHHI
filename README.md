# SportsHHI

This is the official repository of the CVPR 2024 paper [**SportsHHI: A Dataset for Human-Human Interaction Detection in Sports Videos**](https://arxiv.org/abs/2404.04565).

## Download

[SportsHHI Dataset](https://huggingface.co/datasets/MCG-NJU/SportsHHI)

## Baseline Method & Evaluation

### Environment setup

```
conda create --name sports_hhi python=3.8 -y
conda activate sports_hhi
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch  # **This** command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.0
mim install mmdet
pip install einops
pip install numpy==1.23.5
```

### Training & Testing

We provide the training and testing code for our baseline method. Please first specify ```data_dir``` and ```work_dir``` in configuration files in ```configs``` folder.

For training, run ```bash train.sh $CONFIG $GPU_NUM```. ```$CONFIG``` should be a configuration file in ```configs``` folder. ```$GPU_NUM``` should be the number of gpus for training.

For testing, run ```bash test.sh $CONFIG $CHECKPOINT $GPU_NUM```. ```$CONFIG``` should be a configuration file in ```configs``` folder. ```$CHECKPOINT``` is the checkpoint to test. ```$GPU_NUM``` should be the number of gpus for training.

## Acknowledgments

Our implementation of baseline method is developed based on the [mmaction2](https://github.com/open-mmlab/mmaction2/) repository.

## Citation

If you find our code or paper useful, please cite as

```
@misc{wu2024sportshhi,
      title={SportsHHI: A Dataset for Human-Human Interaction Detection in Sports Videos}, 
      author={Tao Wu and Runyu He and Gangshan Wu and Limin Wang},
      year={2024},
      eprint={2404.04565},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
