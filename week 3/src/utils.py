import os
from copy import deepcopy

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def get_ddp_model(model, args):
    model = model.to(args.device)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.distributed:
        ddp_model = DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        ddp_model = None

    return model, ddp_model


def load_state_dict_from_checkpoint(checkpoint, key_list):
    for key in key_list:
        if checkpoint.get(key, None) is not None:
            return checkpoint.get(key)

    return None

def load_checkpoint(checkpoint_path, model, optimizer=None):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optim_dict'])
        return checkpoint
    else:
        raise ValueError(f'no file exist in given checkpoint_path argument(dir={checkpoint_path}')



def resume_from_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, scheduler=None, iter_per_epoch=None):
    """resume training from checkpoint

    :arg
        checkpoint_path(str): checkpoint path
        model(nn.Module): model
        ema_model(nn.Module): ema model
        optimizer: optimizer
        scaler: pytorch native amp scaler
        scheduler: scheduler
    :return
        last epoch
    """
    obj_key_list = [(model, ['state_dict']), (optimizer, ['optimizer']), (scaler, ['scaler'])]
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        for obj, key_list in filter(lambda x: x[0] is not None, obj_key_list):
            state_dict = load_state_dict_from_checkpoint(checkpoint, key_list)

            if state_dict:
                obj.load_state_dict(state_dict)
            elif 'state_dict' in key_list:
                obj.load_state_dict(checkpoint)
            else:
                raise ValueError(f'we can not find {key_list} in given checkpoint(dir={checkpoint_path})')

        epoch = checkpoint.get('epoch', None) + 1
        if scheduler is not None:
            for _ in range((epoch-1) * iter_per_epoch):
                scheduler.step()
        print(f"Resume training from epoch {epoch}")

        return epoch

    else:
        raise ValueError(f'no file exist in given checkpoint_path argument(dir={checkpoint_path}')


def save_checkpoint(save_dir, model, optimizer, scaler, scheduler, epoch, is_best=False):
    pairs = [('state_dict',model), ('optimizer', optimizer), ('scaler', scaler), ('scheduler', scheduler)]
    checkpoint_dict = {k:v.state_dict() for k, v in pairs if v}
    checkpoint_dict['epoch'] = epoch

    torch.save(checkpoint_dict, os.path.join(save_dir, f'checkpoint_{epoch}.pth'))
    torch.save(checkpoint_dict, os.path.join(save_dir, f'checkpoint_last.pth'))
    if is_best:
        torch.save(checkpoint_dict, os.path.join(save_dir, f'checkpoint_best.pth'))
    if os.path.exists(os.path.join(save_dir, f'checkpoint_{epoch-5}.pth')):
        os.remove(os.path.join(save_dir, f'checkpoint_{epoch-5}.pth'))


def print_metadata(model, train_dataset, test_dataset, args):
    title = 'INFORMATION'
    table = [('Project Name', args.project_name), ('Project Administrator', args.who),
             ('Experiment Name', args.exp_name), ('Experiment Start Time', args.start_time),
             ('Experiment Model Name', args.model_name), ('Experiment Log Directory', args.log_dir)]
    print_tabular(title, table, args)

    title = "EXPERIMENT TARGET"
    table = [(target, str(getattr(args, target))) for target in args.exp_target]
    print_tabular(title, table, args)

    title = 'EXPERIMENT SETUP'
    table = [(target, str(getattr(args, target))) for target in [
        'train_size', 'test_size', 'random_crop_pad',
        'mean', 'std', 'hflip',
        'model_name', 'criterion', 'smoothing',
        'lr', 'epoch', 'optimizer', 'momentum', 'weight_decay', 'scheduler', 'warmup_epoch', 'batch_size'
    ]]
    print_tabular(title, table, args)

    title = 'DATA & MODEL'
    table = [('Model Parameters(M)', count_parameters(model)),
             ('Number of Train Examples', len(train_dataset)),
             ('Number of Valid Examples', len(test_dataset)),
             ('Number of Class', args.num_classes),]
    print_tabular(title, table, args)

    title = 'TERMINOLOGY'
    table = [('Batch', 'Time for 1 epoch in seconds'), ('Data', 'Time for loading data in seconds'),
             ('F+B+O', 'Time for Forward-Backward-Optimizer in seconds'), ('Top-1', 'Top-1 Accuracy'),
             ('Top-5', 'Top-5 Accuracy')]
    print_tabular(title, table, args)

    args.log("-" * 81)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_tabular(title, table, args):
    title_space = int((81 - len(title)) / 2)
    args.log("-" * 81)
    args.log(" " * title_space + title)
    args.log("-" * 81)
    for (key, value) in table:
        args.log(f"{key:<25} | {value}")