import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='pytorch-cifar-examples', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    parser.add_argument('data_dir', type=str, help='dataset dir')
    parser.add_argument('--dataset_type', type=str, default='ImageFolder', help='dataset type')

    # data
    data = parser.add_argument_group('data')
    data.add_argument('--train-size', type=int, default=(32, 32), nargs='+', help='train image size')
    data.add_argument('--test-size', type=int, default=(32, 32), nargs='+', help='test image size')
    data.add_argument('--random-crop-pad', type=int, default=0, help='pad size for ResizeRandomCrop')
    data.add_argument('--mean', type=float, default=(0.5, 0.5, 0.5), nargs='+', help='image mean')
    data.add_argument('--std', type=float, default=(0.5, 0.5, 0.5), nargs='+', help='image std')
    data.add_argument('-hf', '--hflip', type=float, default=0.5, help='random horizontal flip')

    # model
    model = parser.add_argument_group('model')
    model.add_argument('-m', '--model-name', type=str, default='resnet50', help='model name')
    model.add_argument('--in-channels', type=int, default=3, help='input channel dimension')

    # criterion
    criterion = parser.add_argument_group('criterion')
    criterion.add_argument('-c', '--criterion', type=str, default='ce', help='loss function')
    criterion.add_argument('--smoothing', type=float, default=0.1, help='label smoothing')

    # optimizer
    optimizer = parser.add_argument_group('optimizer')
    optimizer.add_argument('--lr', type=float, default=1e-3, help='learning rate(lr)')
    optimizer.add_argument('--epoch', type=int, default=100, help='epoch')
    optimizer.add_argument('--optimizer', type=str, default='adamw', help='optimizer name')
    optimizer.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    optimizer.add_argument('--weight-decay', type=float, default=1e-3, help='optimizer weight decay')
    optimizer.add_argument('--nesterov', action='store_true', default=False, help='use nesterov momentum')
    optimizer.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999], help='adam optimizer beta parameter')
    optimizer.add_argument('--eps', type=float, default=1e-6, help='optimizer eps')

    # scheduler
    scheduler = parser.add_argument_group('scheduler')
    scheduler.add_argument('--scheduler', type=str, default='cosine', help='lr scheduler')
    scheduler.add_argument('--min-lr', type=float, default=1e-6, help='lowest lr used for cosine scheduler')
    scheduler.add_argument('--warmup-lr', type=float, default=1e-4, help='warmup start lr')
    scheduler.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')

    # train time
    train_time = parser.add_argument_group('train_time')
    train_time.add_argument('-b', '--batch-size', type=int, default=256, help='batch size')
    train_time.add_argument('-j', '--num-workers', type=int, default=2, help='number of workers')
    train_time.add_argument('--pin-memory', action='store_true', default=False, help='pin memory in dataloader')
    train_time.add_argument('--amp', action='store_true', default=False, help='enable native amp(fp16) training')
    train_time.add_argument('--channels-last', action='store_true', default=False, help='change memory format to channels last')
    train_time.add_argument('--cuda', type=str, default='0,1,2,3,4,5,6,7,8', help='CUDA_VISIBLE_DEVICES options')

    # knowledge distillation
    kd = parser.add_argument_group('kd')
    kd.add_argument('-t', '--temperature', type=float, default=4.0, help='temperature for Soft Target')
    kd.add_argument('--alpha', type=float, default=0.5, help='adjust value of kd loss')
    kd.add_argument('-tm', '--teacher-name', type=str, default='resnet50', help='teacher model')

    # setup
    setup = parser.add_argument_group('setup')
    setup.add_argument('--use-wandb', action='store_true', default=False, help='track std out and log metric in wandb')
    setup.add_argument('-proj', '--project-name', type=str, default='mm_week2_assignment', help='project name used for wandb logger')
    setup.add_argument('--who', type=str, default='your-name', help='enter your name')
    setup.add_argument('-exp', '--exp-name', type=str, default=None, help='experiment name for each run')
    setup.add_argument('--exp-target', type=str, default=['model_name'], nargs='+', help='experiment target')
    setup.add_argument('-out', '--output-dir', type=str, default='log', help='where log output is saved')
    setup.add_argument('-p', '--print-freq', type=int, default=50, help='how often print metric in iter')
    setup.add_argument('--save-checkpoint', action='store_true',
                       help='if enabled, it stores checkpoint during training')
    setup.add_argument('-s', '--seed', type=int, default=None, help='fix seed')
    setup.add_argument('--resume', action='store_true', default=False, help='if true, resume from checkpoint_path')
    setup.add_argument('--checkpoint', type=str, default=None, help='load checkpoint path')
    return parser