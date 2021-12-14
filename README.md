# RepNAS: Searching for Efficient Re-parameterizing Blocks
Code accompanying the paper
> RepNAS: Searching for Efficient Re-parameterizing Blocks\

## Requirements
```
Python == 3.8.8, PyTorch == 1.8.1, torchvision == 0.9.1
```

## Datasets
ImageNet needs to be manually downloaded (preferably to a SSD) following the instructions [here](https://github.com/pytorch/examples/tree/master/imagenet).

## Pretrained models
pretrained models will be available after review.

## Architecture search
run following command to search models
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --data {}
```
where --data {} need to be changed to your own data path.

## Architecture retrain
you also can train the searched model from scratch
run following command to train models
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 retrain.py --data {}
```
## Architecture evaluation (using fused models)
run following command to validate models
```
python valid.py --data {} --pretrained {}
```
where --pretrained need to be changed to your pretrained .pt file.

