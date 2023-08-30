# Model Nodes
RETURN_NODES = {
    'resnet': {'layer4': 'layer4'},
    'dino': {'blocks.11': 'block11'},
}

ATTN_RETURN_NODES = {
    'dino': {'blocks.11.attn.softmax': 'block11_attn'}
}

# Robustness analysis
CTYPE_NOISE = ["gaussian_noise", "shot_noise", "impulse_noise"]
CTYPE_BLUR = ["defocus_blur", "motion_blur", "zoom_blur", "glass_blur"]
CTYPE_WEATHER = ["snow", "frost", "fog"]
CTYPE_DIGITAL = ["brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]
CTYPE_ETC = ["speckle_noise", "gaussian_blur", "spatter", "saturate"]
CTYPE = CTYPE_NOISE + CTYPE_BLUR + CTYPE_WEATHER + CTYPE_DIGITAL + CTYPE_ETC
CTYPE_INTENSITY = [1, 2, 3, 4, 5]
