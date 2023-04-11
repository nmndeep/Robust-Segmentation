import torch
from torch import nn, Tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor

class LayerNorm(nn.Module):
    """Channel first layer norm
    """
    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvBlock1(nn.Module):
    def __init__(self, siz=48):
        super(ConvBlock1, self).__init__()
        self.planes = siz
        self.stem = nn.Sequential(nn.Conv2d(3, self.planes, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes, eps=1e-6),
                                  nn.GELU(),
                                  nn.Conv2d(self.planes, self.planes*2, kernel_size=3, stride=2, padding=1),
                                  LayerNorm(self.planes*2, eps=1e-6),
                                  nn.GELU()
                                  )

    def forward(self, x):
        out = self.stem(x)
        return out


    
class Block(nn.Module):
    def __init__(self, dim, dpr=0., init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4*dim, dim)
        self.gamma = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True) if init_value > 0 else None
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)   # NCHW to NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x
        
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class Stem(nn.Sequential):
    def __init__(self, c1, c2, k, s):
        super().__init__(
            nn.Conv2d(c1, c2, k, s),
            LayerNorm(c2)
        )


class Downsample(nn.Sequential):
    def __init__(self, c1, c2, k, s):
        super().__init__(
            LayerNorm(c1),
            nn.Conv2d(c1, c2, k, s)
        )


convnext_settings = {
    'T': [[3, 3, 9, 3], [96, 192, 384, 768], 0.4],       # [depths, dims, dpr]
    'T_CVST': [[3, 3, 9, 3], [96, 192, 384, 768], 0.4],       # [depths, dims, dpr]
    'S': [[3, 3, 27, 3], [96, 192, 384, 768], 0.0],
    'B': [[3, 3, 27, 3], [128, 256, 512, 1024], 0.0]
}


class ConvNeXt(nn.Module):     
    def __init__(self, model_name: str = 'T') -> None:
        super().__init__()
        
        assert model_name in convnext_settings.keys(), f"ConvNeXt model name should be in {list(convnext_settings.keys())}"
        depths, embed_dims, drop_path_rate = convnext_settings[model_name]
        self.channels = embed_dims
        self.stem = nn.ModuleList([Stem(3, embed_dims[0], 4, 4) if 'CVST' not in model_name else ConvBlock1(), *[Downsample(embed_dims[i], embed_dims[i+1], 2, 2) for i in range(3)]]) 

        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(*[
                Block(embed_dims[i], dpr[cur+j])
            for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        for i in range(4):
            self.add_module(f"norm{i}", LayerNorm(embed_dims[i]))

    def forward(self, x: Tensor):
        outs = []

        for i in range(4):
            x = self.stem[i](x)
            x = self.stages[i](x)
            norm_layer = getattr(self, f"norm{i}")
            outs.append(norm_layer(x))
        return outs


def convnext(backbone, pretrained):
    backbone, variant = backbone.split('-')
    model = ConvNeXt(variant)
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

if __name__ == '__main__':
    model = ConvNeXt('T')
    # model.load_state_dict(torch.load('C:\\Users\\sithu\\Documents\\weights\\backbones\\convnext\\convnext_tiny_1k_224_ema.pth', map_location='cpu')['model'], strict=False)
    x = torch.randn(1, 3, 224, 224)
    feats = model(x)
    for y in feats:
        print(y.shape)