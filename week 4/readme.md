# Distributed Data Parallel

With DDP, the model is replicated on every process, and every model replica will be fed with a different set of input data samples. DDP takes care of gradient communication to keep model replicas synchronized and overlaps it with the gradient computations to speed up training.

- `Tutorial` : [https://tutorials.pytorch.kr/beginner/dist_overview.html](https://tutorials.pytorch.kr/beginner/dist_overview.html)
- `Reference 1` : [https://algopoolja.tistory.com/95](https://algopoolja.tistory.com/95)
- `Reference 2` : [https://csm-kr.tistory.com/47](https://csm-kr.tistory.com/47)
- `Reference 3` : [https://tutorials.pytorch.kr/intermediate/dist_tuto.html](https://tutorials.pytorch.kr/intermediate/dist_tuto.html)

In this practice, we will use single-node with multi-gpus environment.

![ddp_tutorials](https://github.com/Team-Ryu/vision-tutorial-solution/assets/90232305/46e57e00-c2a5-44ec-82ab-30c2e525d0b3)


To run a multiprocess application, you must use the following commands.
```bash
CUDA_VISIBLE_DEVICES=GPUs_ID python -m torch.distributed.launch --nproc_per_node=NUM_GPUs your_models.py
```
- For example, run model with 2 gpus,
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 your_models.py
```
- Depending on the version, you can also use the `torchrun` command
```bash
torchrun --nproc_per_node=2 your_models.py
```


# Wandb
WandB is a central dashboard to keep track of your hyperparameters, system metrics, and predictions so you can compare models live, and share your findings.

- `Tutorial` : [https://wandb.ai/site/experiment-tracking](https://wandb.ai/site/experiment-tracking)


# ArgumentParser

커맨드 라인에 인수를 받아 간단히 프로그램을 실행할 수 있도록 하는 표준 라이브러리

### Usage

```python
#라이브러리 임포트
import argparse

#parser 정의
parser = argparse.ArgumentParser(description='해당 파일에 대한 설명')

#인수 설정
#1. 필수 인수
parser.add_argument('arg1', help='해당 argument에 대한 설명')

#2. 옵션 인수
parser.add_argument('-arg2', help='하이푼이 존재하는 경우, 옵션 인수, -arg2 = --arg2')

#3. 옵션 인수 + 약칭
parser.add_argument('-a', '--arg3', help='약칭은 보통 하나의 하이푼으로 선언')

#4. 디폴트 값 선언 + 데이터 형 지정(int, float, str 등등)
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('-t', '--text', type=str, default=None)

#5. 파서 사용 시, 반드시 선언해줘야함
args = parser.parse_args()

#호출 코드 : python this_file.py parser1 --arg2 123 -a 12345 -b 333 -t abcd

print(args.arg1) #parser1
print(args.arg2) #123
print(args.arg3) #12345
print(args.batch_size) #333
print(args.text) #abcd
```

### Grouping
서로 다른 파서그룹 설정. 서로 다른 그룹으로 명시해도 되지만 timm에서는 단순히 식별용으로 나누는 듯
```python
#라이브러리 임포트
import argparse

#parser 정의
parser = argparse.ArgumentParser(description='해당 파일에 대한 설명')

# 데이터 파서
group = parser.add_argument_group('Dataset param')
parser.add_argument('--data-dir', metavar='DIR', help='dataset path')
parser.add_argument('dataset', default='ImageNet')

# 모델 파서
group = parser.add_argument_group('Model param')
parser.add_argument('--model', default='resnet50', type=str)

# 옵티마이저 파서
group = parser.add_argument_group('Optim param')
parser.add_argument('--opt', default='AdamW', type=str)

args = parser.parse_args()


#호출 코드 python test_1.py --data-dir datapath imagenet --model model_name --opt adamw
print(args.data_dir) #datapath
print(args.dataset) #imagenet
print(args.model) #model_name
print(args.opt) #adamw
```

### 적용
```python
model = create_model(
  args.model,
  args.drop_rate,
  args.blablabla
)
```


# Exercise
A naive training code is provided. Apply DDP and Wandb to the training code provided. Check classification performance using the given model(ResNet-50) and hyper-parameters on CIFAR-100. If the model is trained normally, you will be able to achieve about 77% classification performance. You can refer to other libraries, but copying and pasting is prohibited. Please think about the principle of DDP and write your code.


- If you use the training script below with 2 gpus, you can achieve 77% top-1 accuracy. Use the provided training code and model to conduct the experiment.
```bash
torchrun --nproc_per_node=2 train.py --dataset cifar100 -b 128 --cuda 6,7 --log-wandb
```

#### 힌트
1. ddp 초기화 (local rank, world size)
2. dataset 을 ddp에 적용하려면?
3. 모델을 ddp에 적용하려면?
4. ddp를 사용하여 모델을 학습하려면?
- timm library -> train.py -> search for 'if args.distributed'!!!
