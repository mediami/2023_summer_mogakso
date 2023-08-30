import os

import torch
from PIL import Image
from timm import create_model

from src import *

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(args.analysis, list):
        methods = args.analysis
    elif args.analysis == 'all':
        methods = list(get_method_dict().keys())
        print(methods)
    else:
        raise AttributeError(f'{args.analysis} is not supported')
    print(f"To be performed: {methods}")

    cnn_name = 'resnet'
    tf_name = 'dino'
    model_cnn = create_model('resnet50', pretrained=True).to(device)
    model_tf = create_model('dino_vitsmall_patch8', pretrained=True).to(device)
    model_cnn.eval()
    model_tf.eval()
    data = transform(Image.open(args.img_path), args.test_size).to(device)

    # list_graph_node_names(model_cnn, verbose=False)

    common_kwargs = dict(data=data, label=args.img_label, img_size=args.test_size, data_dir=args.data_dir,
                         pca_data_dir=args.pca_data_dir, num_class=args.n_class, device=device, n_components=3,
                         ctype=['zoom_blur', 'contrast'], intensity=[1, 2])

    for m in methods:
        os.makedirs(args.output_dir + f'/{cnn_name}', exist_ok=True)
        os.makedirs(args.output_dir + f'/{tf_name}', exist_ok=True)
        method = create_method(m)

        method(model=model_cnn, model_name=cnn_name, return_nodes=RETURN_NODES[cnn_name], attn_return_nodes='',
               save=args.output_dir + f'/{cnn_name}', **common_kwargs)
        clear()
        method(model=model_tf, model_name=tf_name, return_nodes=RETURN_NODES[tf_name], patch_size=8,
               attn_return_nodes=ATTN_RETURN_NODES[tf_name], save=args.output_dir + f'/{tf_name}', **common_kwargs)
        clear()