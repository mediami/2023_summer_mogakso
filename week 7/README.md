# Analysis Framework

## Supported Analysis

Visualization: deeplift, effective_receptive_field, gradcam, self_attention_heatmap, pca

Shape and Texture: frequency

Performance: robustness, consistency

## Core Arguments

#### Related with model
- model: Torch.nn.Module = neural architecture model
- return_nodes: dict = {layer_name: output_name}  ## Layer name is target layer for generate feature map and output_name is feature map name what you want
- attn_return_nodes: dict = {layer_name: output_name} ## Similar with return_nodes, this target layer is attention map, not feature map

#### Related with data
- data_dir: str = ./imageNet ## Dataset root for performance analysis
- n_class: int = 1000
- img_path: str = src/data/sample_images/corgi.jpg  ## Single image path
- img_label: int = 256  ## Label of single image
- pca_data_dir: str = src/data/harryported_giffins  ## Image folder path for PCA

#### ETC
- n_component: int = 3 ## Number of component for PCA