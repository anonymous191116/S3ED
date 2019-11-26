# S3EDeblur

Pytorch implementation of **ImD** and **S3-tree** for **Semantic-Guided Embedding for Severely BlurredImage Restoration**

This code is modified according to [DeblurGAN](https://github.com/KupynOrest/DeblurGAN) and [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).


## Data Prepropcessing

Getting  blurred images' feature maps and last fc features:
```
python data/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/cocotalk --images_root BLURRED_IMAGES_ROOT
```

## Initialize Training for S3-tree
Setting your own hyper-parameters in `opt` dict which is at  `pretrain_tree/train.py`.
Run S3-tree pretraining process:
```
python pretrain_tree/train.py
```

## Co-Training for S3-tree and ImD
```
python train.py --OPTINS(details are in ./option)
```
S3-tree only support generator of model type 'resnet_9blocks' for now.

## Testing
Generating  deblurred images:
```
python test.py
```
Calculating SSIM and PSNR.
```
python calculate_metrics.py
```
