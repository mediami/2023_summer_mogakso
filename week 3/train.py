import time
import datetime
import torch

from src.args import get_args_parser
from src.setup import setup
from src.dataset import get_dataset, get_dataloader
from src.resnet import create_resnet
from src.optimizer import get_optimizer_and_scheduler, get_scaler_criterion
from src.metric import Metric, Accuracy, reduce_mean
from src.utils import print_metadata, save_checkpoint, get_ddp_model, resume_from_checkpoint


@torch.inference_mode()
def validate(valid_dataloader, model, criterion, args, mode='org'):
    # 1. create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')
    top1_m = Metric(reduce_every_n_step=args.print_freq, header='Top-1:')
    top5_m = Metric(reduce_every_n_step=args.print_freq, header='Top-5:')
    loss_m = Metric(reduce_every_n_step=args.print_freq, header='Loss:')

    # 2. start validate
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(valid_dataloader)
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(valid_dataloader):
        batch_size = x.size(0)
        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat = model(x)
            loss = criterion(y_hat, y)

        top1, top5 = Accuracy(y_hat, y, top_k=(1,5,))

        top1_m.update(top1, batch_size)
        top5_m.update(top5, batch_size)
        loss_m.update(loss, batch_size)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"VALID({mode}): [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m} {loss_m} {top1_m} {top5_m}")

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # 3. calculate metric
    duration = str(datetime.timedelta(seconds=batch_m.sum)).split('.')[0]
    data = str(datetime.timedelta(seconds=data_m.sum)).split('.')[0]
    f_b_o = str(datetime.timedelta(seconds=batch_m.sum - data_m.sum)).split('.')[0]
    top1 = top1_m.compute()
    top5 = top5_m.compute()
    loss = loss_m.compute()

    # 4. print metric
    space = 16
    num_metric = 6
    args.log('-'*space*num_metric)
    args.log(("{:>16}"*num_metric).format('Stage', 'Batch', 'Data', 'F+B+O', 'Top-1 Acc', 'Top-5 Acc'))
    args.log('-'*space*num_metric)
    args.log(f"{'VALID('+mode+')':>{space}}{duration:>{space}}{data:>{space}}{f_b_o:>{space}}{top1:{space}.4f}{top5:{space}.4f}")
    args.log('-'*space*num_metric)

    return loss, top1, top5


def train_one_epoch(train_dataloader, model, optimizer, criterion, args, scheduler=None, scaler=None, epoch=None):
    # 1. create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')
    loss_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Loss:')

    # 2. start validate
    model.train()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(train_dataloader)
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(train_dataloader):
        batch_size = x.size(0)
        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat = model(x)
            loss = criterion(y_hat, y)

        if args.distributed:
            loss = reduce_mean(loss, args.world_size)

        if args.amp:
            scaler(loss, optimizer, model.parameters(), scheduler)
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        loss_m.update(loss, batch_size)
        torch.cuda.synchronize()

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            thru = x.size(0) * args.world_size / batch_m.avg
            args.log(f"TRAIN({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}]  "
                     f"THR: {thru:.1f} img/s  {batch_m}  {data_m}  {loss_m}")

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # 3. calculate metric
    duration = str(datetime.timedelta(seconds=batch_m.sum)).split('.')[0]
    data = str(datetime.timedelta(seconds=data_m.sum)).split('.')[0]
    f_b_o = str(datetime.timedelta(seconds=batch_m.sum - data_m.sum)).split('.')[0]
    loss = loss_m.compute()

    # 4. print metric
    space = 16
    num_metric = 5
    args.log('-'*space*num_metric)
    args.log(("{:>16}"*num_metric).format('Stage', 'Batch', 'Data', 'F+B+O', 'Loss'))
    args.log('-'*space*num_metric)
    args.log(f"{'TRAIN('+str(epoch)+')':>{space}}{duration:>{space}}{data:>{space}}{f_b_o:>{space}}{loss:{space}.4f}")
    args.log('-'*space*num_metric)

    return loss


def run(args):
    # 0. init ddp & logger
    setup(args)

    # 1. load dataset
    train_dataset, valid_dataset = get_dataset(args)
    train_dataloader, valid_dataloader = get_dataloader(train_dataset, valid_dataset, args)

    # 2. make model
    model = create_resnet(args.model_name, args.num_classes)
    model, ddp_model = get_ddp_model(model, args)

    # 3. load optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    # 4. load criterion
    criterion, valid_criterion, scaler = get_scaler_criterion(args)

    # 5. print metadata
    print_metadata(model, train_dataset, valid_dataset, args)

    # 6. train
    start_epoch = 0
    end_epoch = args.epoch
    best_epoch = 0
    best_acc = 0
    top1_list = []
    top5_list = []

    if args.resume:
        start_epoch = resume_from_checkpoint(args.resume, model, optimizer, scaler, scheduler, args.iter_per_epoch)

    start_time = time.time()

    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(train_dataloader, ddp_model if args.distributed else model, optimizer, criterion, args, scheduler, scaler, epoch)
        val_loss, top1, top5 = validate(valid_dataloader, ddp_model if args.distributed else model, valid_criterion, args, 'org')

        if args.use_wandb:
            args.log({'train_loss':train_loss, 'val_loss':val_loss, 'top1':top1, 'top5':top5}, metric=True)

        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch
        top1_list.append(top1)
        top5_list.append(top5)

        if args.is_rank_zero and args.save_checkpoint:
            save_checkpoint(args.log_dir, model, optimizer, scaler, scheduler, epoch, is_best=best_epoch == epoch)

    print(f"Best Acc@epoch: {best_acc:.2f}@{best_epoch}")

if __name__ == '__main__':
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    run(args)