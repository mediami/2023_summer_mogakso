import copy

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model)

    def forward(self, x, **kwargs):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        return x


class Residual(nn.Module):
    def __init__(self, *fn):
        super().__init__()
        self.fn = nn.Sequential(*fn)

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def flatten(xs_list):
    return [x for xs in xs_list for x in xs]


def get_blocks(model, model_name):
    if 'vit' in model_name or 'deit' in model_name or 'dino' in model_name:
        blocks = [
            PatchEmbed(model),
            *flatten([[Residual(b.norm1, b.attn), Residual(b.norm2, b.mlp)]
                      for b in model.blocks]),
            nn.Sequential(model.norm, Lambda(lambda x: x[:, 0]), model.head),
        ]
    elif 'resnet' in model_name:
        blocks = [
            nn.Sequential(model.conv1, model.bn1, model.act1, model.maxpool),
            *model.layer1,
            *model.layer2,
            *model.layer3,
            *model.layer4,
            nn.Sequential(model.global_pool, model.fc)
        ]
    elif 'densenet' in model_name:
        blocks = [
            nn.Sequential(model.features.conv0, model.features.norm0, model.features.pool0),
            nn.Sequential(model.features.denseblock1, model.features.transition1),
            nn.Sequential(model.features.denseblock2, model.features.transition2),
            nn.Sequential(model.features.denseblock3, model.features.transition3),
            nn.Sequential(model.features.denseblock4),
            nn.Sequential(model.global_pool, model.classifier)
        ]
    else:
        raise ValueError(f'The blocks of {model_name} are not defined.')

    return blocks
