# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class ConvBlock1(nn.Module):
    def __init__(self, siz=48):
        super(ConvBlock1, self).__init__()
        self.planes = siz
        self.stem = nn.Sequential(nn.Conv2d(3, self.planes, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes, eps=1e-6, data_format='channels_first'),
                                  nn.GELU(),
                                  nn.Conv2d(self.planes, self.planes*2, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes*2, eps=1e-6, data_format='channels_first'),
                                  nn.GELU()
                                  )

    def forward(self, x):
        out = self.stem(x)
        return out



class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


convnext_settings = {
    'T': [[3, 3, 9, 3], [96, 192, 384, 768], 0.4],       # [depths, dims, dpr]
    'T_CVST': [[3, 3, 9, 3], [96, 192, 384, 768], 0.4],       # [depths, dims, dpr]
    'S': [[3, 3, 27, 3], [96, 192, 384, 768], 0.0],
    'B': [[3, 3, 27, 3], [128, 256, 512, 1024], 0.0]
}


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, strr, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        assert strr in convnext_settings.keys(), f"ConvNeXt model name should be in {list(convnext_settings.keys())}"
        depths, dims, drop_path_rate = convnext_settings[strr]
        # self.backbone = eval(backbone)(variant)

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")) if 'CVST' not in strr else ConvBlock1()

        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')


    def load_carefully(self, pretrained):
        ckpt = torch.load(pretrained)['model']
        stt = [3,3,9,3]
        # for nn in ckpt.keys():
        #     print(nn)
        with torch.no_grad():
            for i in range(4):
                for p in range(2):
                    self.downsample_layers[i][p].weight.copy_(ckpt[f'downsample_layers.{i}.{p}.weight'])
                    self.downsample_layers[i][p].bias.copy_(ckpt[f'downsample_layers.{i}.{p}.bias'])
            for j in range(4):
                for k in range(stt[j]):
                    self.stages[j][k].gamma.copy_(ckpt[f'stages.{j}.{k}.gamma'])
                    self.stages[j][k].dwconv.weight.copy_(ckpt[f'stages.{j}.{k}.dwconv.weight'])
                    self.stages[j][k].dwconv.bias.copy_(ckpt[f'stages.{j}.{k}.dwconv.bias'])
                    self.stages[j][k].norm.weight.copy_(ckpt[f'stages.{j}.{k}.norm.weight'])
                    self.stages[j][k].norm.bias.copy_(ckpt[f'stages.{j}.{k}.norm.bias'])
                    self.stages[j][k].pwconv1.weight.copy_(ckpt[f'stages.{j}.{k}.pwconv1.weight'])
                    self.stages[j][k].pwconv1.bias.copy_(ckpt[f'stages.{j}.{k}.pwconv1.bias'])
                    self.stages[j][k].pwconv2.weight.copy_(ckpt[f'stages.{j}.{k}.pwconv2.weight'])
                    self.stages[j][k].pwconv2.bias.copy_(ckpt[f'stages.{j}.{k}.pwconv2.bias'])


    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # exit()
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
        # print(x_out.size())
        # exit()
        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        # print(len(x), x[0].size())
        # exit()
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x






def convnext(backbone, pretrained):
    backbone, variant = backbone.split('-')
    model = ConvNeXt(strr=variant, drop_path_rate=0.3, layer_scale_init_value=1.0)
    try:
        model.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
        print('Loaded from checkpoint')
    except:
        ckpt = torch.load(pretrained, map_location='cpu', strict=False) #['model']
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        ckpt = {k.replace('base_model.', ''): v for k, v in ckpt.items()}
        ckpt = {k.replace('se_', 'se_module.'): v for k, v in ckpt.items()}
        model.load_state_dict.load_state_dict(ckpt)
    return model
