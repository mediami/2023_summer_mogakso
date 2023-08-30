import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='analysis', add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # analysis
    analysis = parser.add_argument_group('analysis')
    analysis.add_argument('-a', '--analysis', default='all', help='methods of analysis. all or list of methods')

    # analysis-data
    parser.add_argument('--data-dir', type=str, default='./imageNet/val', help='dataset dir')
    parser.add_argument('--dataset-type', type=str, default='ImageFolder', help='dataset type')
    parser.add_argument('--img-path', type=str, default='src/data/sample_images/corgi.jpg', help='single image path')
    parser.add_argument('--img-label', type=int, default=263, help='single image label')
    parser.add_argument('--pca-data-dir', type=str, default='src/data/harryported_giffins', help='image dir for PCA')
    parser.add_argument('--n-class', type=int, default=1000, nargs='+', help='number of class')

    # train time
    train_time = parser.add_argument_group('train_time')
    train_time.add_argument('-b', '--batch-size', type=int, default=128, help='batch size')
    train_time.add_argument('-j', '--num-workers', type=int, default=4, help='number of workers')
    train_time.add_argument('--pin-memory', action='store_true', default=False, help='pin memory in dataloader')
    train_time.add_argument('--amp', action='store_true', default=False, help='enable native amp(fp16) training')
    train_time.add_argument('--channels-last', action='store_true', default=False,
                            help='change memory format to channels last')
    train_time.add_argument('--cuda', type=str, default='0', help='CUDA_VISIBLE_DEVICES options')

    # data
    data = parser.add_argument_group('data')
    data.add_argument('--test-size', type=int, default=(224, 224), nargs='+', help='test image size')
    data.add_argument('--random-crop-pad', type=int, default=0, help='pad size for ResizeRandomCrop')
    data.add_argument('--mean', type=float, default=(0.5, 0.5, 0.5), nargs='+', help='image mean')
    data.add_argument('--std', type=float, default=(0.5, 0.5, 0.5), nargs='+', help='image std')

    # model
    model = parser.add_argument_group('model')
    model.add_argument('-m', '--model-name', type=str, default='resnet50', help='model name')
    model.add_argument('--pretrained', action='store_true', default=True, help='load pretrained weight')
    model.add_argument('--in-channels', type=int, default=3, help='input channel dimension')

    # criterion
    criterion = parser.add_argument_group('criterion')
    criterion.add_argument('-c', '--criterion', type=str, default='ce', help='loss function')
    criterion.add_argument('--smoothing', type=float, default=0.1, help='label smoothing')

    # setup
    setup = parser.add_argument_group('setup')
    setup.add_argument('-out', '--output-dir', type=str, default='analysis_result', help='where log output is saved')
    setup.add_argument('-p', '--print-freq', type=int, default=50, help='how often print metric in iter')
    setup.add_argument('--save-checkpoint', action='store_true',
                       help='if enabled, it stores checkpoint during training')
    setup.add_argument('-s', '--seed', type=int, default=None, help='fix seed')
    return parser
