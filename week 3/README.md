# Classification

This repo contains PyTorch-based image classification learning material.




## Tutorial

1. git clone this repo.

   ```bash
   git clone https://github.com/Team-Ryu/image-classification-baseline
   ```

2. train your model.

   ```bash
   cd image-classification-baseline
   python3 multi_train.py cifar100 -m resnet50 -c 0 --use-wandb
   ```

## Experiment Result

*To Do: Please fill this table.*

| Model    | Top-1 | Top-5 | # Params | FLOPs | Train time |
|----------|-------|-------|----------|-------|------------|
| ResNet50 | 79.8  | 94.0  | 23.7M    | 4.2G  | 1h 5m      |
| YourNet  | -     | -     | -        | -     | -          |

**Rules:**

- The total train time should be lower than 2 hours.
- The number of parameters should be lower than 30M.
- The final accuracies (top-1, top-5) should be reported not the best ones.
