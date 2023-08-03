# timm library

timm is a library containing SOTA computer vision models, layers, utilities, optimizers, schedulers, data-loaders, augmentations, and training/evaluation scripts

- `Tutorial` : [https://huggingface.co/docs/timm/index](https://huggingface.co/docs/timm/index)
- `Github` : [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)

## Why timm?
- 770 pretrained SOTA models
- Includes a variety of SOTA augmentation techniques
- Fair comparison
- Easy to run and load models


## Start with timm
```python
pip install timm
```
or for an editable install,
```python
git clone https://github.com/rwightman/pytorch-image-models
cd pytorch-image-models && pip install -e .
```

- you can easily check and load pretrained model as follow:
```python
import timm

print(timm.list_models(pretrained=True))
m = timm.create_model('resnet', pretrained=True)
```

- The timm library declares a formalized model, so it is convenient to access a specific blocks
- `i.e.` forward_features(patch embedding layer + feature extractor), forward_head(fc layer), forward...
```python
import torch
import timm

m = timm.create_model('xception41', pretrained=True)
models_forward = m.forward_features(torch.randn(2, 3, 299, 299))
```

## Training script

- All you need is simply execute the code below.
```bash
torchrun --nproc_per_node=2 --master_port=12346 train.py your_imagenet_dir --model resnet50 --cuda 0,1 --input-size 3 160 160 --test-input-size 3 224 224 --aa rand-m6-mstd0.5-inc1 --mixup .1 --cutmix 1.0 --aug-repeats 0 --remode pixel --reprob 0.0 --crop-pct 0.95 --drop-path 0 --smoothing 0.0 --bce-loss --opt lamb --weight-decay .02 --sched cosine --epochs 100 --lr 8e-3 --warmup-lr 1e-6 -b 512 -j 16 --amp --channels-last --log-wandb
```
- But, you should search optimal hyper-parameters...
- Use the `ln -s` command to access the dataset in the shared folder. `your_imagenet_dir`
```bash
ln -s /home/your_id/shared/.../imageNet
```

## Benchmark

- timm library support simple analysis such as `throughput`, `FLOPs`, `Parameter`...
- To verify your model, use benchmark code below:
```bash
CUDA_VISIBLE_DEVICES=8 python benchmark.py --bench inference --model resnet50 --amp --channels-last
```
- You can achieve "model": "resnet50", "infer_samples_per_sec": 3352.47, "infer_step_time": 76.351, "infer_batch_size": 256, "infer_img_size": 224, "param_count": 25.56


## Custom model

- To train your custom model, follow the steps below:
  1. Import the required library into your custom_models.py
     ```python
     #models.py
     from timm.models import register_model
     ```
  2. Define function that return model and decorated it with `register_model`
     ```python
     #models.py
     @register_model
     def your_model_name(pretrained=False, **kwargs):
      """Constructs your model.
      """
      model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
      return _create_your_model('imagenet', pretrained, **model_args)
     ```
  3. Import defined function in train.py
     ```python
     # train.py
     import models
     ```
  4. Use the above-mentioned training commands

# How to find hyper-parameter?
To verify the performance of the proposed methodology, it is important to find hyperparameters that can re-implement the performance of the baseline architecture.

In general, finding hyperparameters is as follows.
1. Refer to the experimental section of the paper.
   ![image](https://github.com/Team-Ryu/timm-imagenet-train/assets/90232305/bf3e7f1b-d47e-4fbe-8118-f5b4e6575267)
3. Please refer to the official GitHub address.
   - In most cases, similar questions will exist in the issue section of the official GitHub.
   - [hivit-config.py](https://github.com/zhangxiaosong18/hivit/blob/master/supervised/config.py)
   - [hivit-default config](https://github.com/zhangxiaosong18/hivit/blob/master/supervised/configs/hivit_small_224.yaml)
4. Refer to the experimental section and GitHub of the other paper that cited that paper.
   - In this case, the author used `Swin Transformer`


## Exercise
1. Find hyperparameter settings for re-implementation of paper [DeiT](https://arxiv.org/abs/2012.12877).
2. Find your own training setting that can maximize the performance of your resnet50 using the code given.


## Experiment Result

*To Do: Please fill this table.*

|          | Top-1 | Top-5 | Key factor | Reason? | Explain your setting | 
|----------|-------|-------|------------|---------|----------------------|
| Default setting  | 79.8  | 94.0  |    |         |                      |
| Your setting     | -     | -     |    |         |                      |

### Command
```bash
torchrun --nproc_per_node=2 --master_port=12345 train.py --model your_resnet_50 --cuda your_gpu_id --dataset torch/cifar100 --input-size 3 32 32 --num-classes 100 'insert your own hyperparameter' -j 4 --amp --channels-last --pin-mem --log-wandb
```
