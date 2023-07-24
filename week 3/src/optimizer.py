import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from torch.optim import SGD, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR


class BinaryCrossEntropy(nn.Module):
    def __init__(self, label_smoothing=0.1, bce_target=None):
        """Binary Cross Entropy (timm)
        :arg
            label_smoothing: multi-class loss in bce.
            bce_target: remove uncertain target used with cutmix.
        """
        super(BinaryCrossEntropy, self).__init__()
        self.smoothing = label_smoothing
        self.bce_target = bce_target

    def forward(self, x, y):
        if x.shape != y.shape:
            smooth = self.smoothing / x.size(-1)
            label = 1.0 - self.smoothing + smooth
            smooth_target = torch.full_like(x, smooth)
            y = torch.scatter(smooth_target, -1, y.long().view(-1, 1), label)
        if self.bce_target:
            y = torch.gt(y, self.bce_target).long()
        return F.binary_cross_entropy_with_logits(x, y, reduction='mean')

class SoftTarget(nn.Module):
    def __init__(self, T=4.0, alpha=1):
        super(SoftTarget, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, x_s, y, x_t):
        loss_CE = F.cross_entropy(x_s, y)
        loss_KD = F.kl_div(F.log_softmax(x_s / self.T, dim=1), F.softmax(x_t / self.T, dim=1),reduction='batchmean') * self.T * self.T

        return (1. - self.alpha)*loss_CE + self.alpha*loss_KD

class NativeScalerWithGradAccum:
    def __init__(self):
        """NativeScalerWithGradAccum (timm)
        Native(pytorch) f16 scaler
        """
        self._scaler = GradScaler()

    def __call__(self, loss, optimizer, model_param, scheduler=None):
        self._scaler.scale(loss).backward()
        self._scaler.step(optimizer)
        self._scaler.update()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_optimizer_and_scheduler(model, args):
    """get optimizer and scheduler
    :arg
        model: nn.Module instance
        args: argparse instance containing optimizer and scheduler hyperparameter
    """
    parameter = model.parameters()
    total_iter = args.epoch * args.iter_per_epoch
    warmup_iter = args.warmup_epoch * args.iter_per_epoch

    if args.optimizer == 'sgd':
        optimizer = SGD(parameter, args.lr, args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(parameter, args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(parameter, args.lr, eps=args.eps, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        NotImplementedError(f"{args.optimizer} is not supported yet")

    if args.scheduler == 'cosine':
        main_scheduler = CosineAnnealingLR(optimizer, total_iter-warmup_iter, args.min_lr)
    else:
        NotImplementedError(f"{args.scheduler} is not supported yet")

    if args.warmup_epoch:
        lr_lambda = lambda e: (e * (args.lr - args.warmup_lr) / warmup_iter + args.warmup_lr) / args.lr
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], [warmup_iter])
    else:
        scheduler = main_scheduler

    return optimizer, scheduler


def get_scaler_criterion(args):
    """Get Criterion(Loss) function and scaler
    Criterion functions are divided depending on usage of mixup
    - w/ mixup - you don't need to add smoothing loss, because mixup will add smoothing loss.
    - w/o mixup - you should need to add smoothing loss
    """
    if args.criterion in ['ce', 'crossentropy']:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    elif args.criterion in ['bce', 'binarycrossentropy']:
        criterion = BinaryCrossEntropy(label_smoothing=args.smoothing, bce_target=args.bec_target)
    elif args.criterion in ['kd', 'st', 'softtarget']:
        criterion = SoftTarget(T=args.temperature, alpha=args.alpha)
    valid_criterion = nn.CrossEntropyLoss()

    if args.amp:
        scaler = NativeScalerWithGradAccum()
    else:
        scaler = None

    return criterion, valid_criterion, scaler