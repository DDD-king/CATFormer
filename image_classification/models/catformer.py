# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def sparse_query_index(x):
    B, N, C = x.shape
    if N == 56 * 56:
        idx = list(i*56+j for i in list(range(0,56,4)) for j in list(range(0,56,4))) #1/16
    elif N == 28 * 28:
        idx = list(i*28+j for i in list(range(0,28,2)) for j in list(range(0,28,2))) #1/4
    elif N == 14 * 14:
        idx = list(i*14+j for i in list(range(0,14,1)) for j in list(range(0,14,1))) #1/1
    elif N == 7 * 7:
        idx = list(i*7+j for i in list(range(0,7,1)) for j in list(range(0,7,1))) #1/1
    return idx


class SAAC(nn.Module):
    """
    Self-Attention And Convolution
    """
    def __init__(self, dim, num_head=8, sr_ratio=1, attn_drop=0., proj_drop=0., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, kernel_size=5):
        super().__init__()
        self.num_head = num_head
        head_dim = dim // num_head
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.kernel_size = kernel_size

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm1 = norm_layer(dim)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.kernel_size > 1:
            self.norm2 = norm_layer(dim)
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)

    def forward(self, x, H, W, idx):
        B, N, C = x.shape

        q = self.q(x[:,idx,:].contiguous()).reshape(B, len(idx), self.num_head, C // self.num_head).permute(0, 2, 1, 3).contiguous()
        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).flatten(2).transpose(1, 2)
            x_ = self.norm1(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        xq = (attn @ v).transpose(1, 2).contiguous().reshape(B, len(idx), C)

        if self.kernel_size > 1:
            y = x.clone()
            y[:,idx,:] = xq.float()
            
            y = self.norm2(y)
            y = y.transpose(1, 2).contiguous().reshape(B, C, H, W)
            y = self.dwconv(y)
            xq = y.flatten(2).transpose(1, 2).contiguous()

        xq = self.proj(xq)
        xq = self.proj_drop(xq)
        return xq


class TCBlock(nn.Module):
    def __init__(self, dim, num_head=8, mlp_ratio=4., sr_ratio=1, drop_path=0., drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=5):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pos_drop = nn.Dropout(p=drop)

        self.norm1 = norm_layer(dim)
        self.attn = SAAC(dim, num_head=num_head, sr_ratio=sr_ratio, attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer, kernel_size=kernel_size)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.size()

        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        idx = sparse_query_index(x)
        
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, idx))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).contiguous().reshape(B, C, H, W)
        return x


class Stage(nn.Module):
    def __init__(self, istage=0, depths=[3, 3, 3, 3], dims=[64, 128, 320, 512], num_heads=[2, 4, 8, 16], 
                 mlp_ratio=4., sr_ratios=[4, 2, 1, 1], kernel_sizes=[5, 5, 1, 1], dp_rates=0.0, drop=0., attn_drop=0.):
        super().__init__()

        self.istage = istage
        layers = []
        cur = 0
        for i in range(4):
            if i == istage:
                for j in range(depths[i]):
                    layers.append(TCBlock(dim=dims[i], num_head=num_heads[i], mlp_ratio=mlp_ratio, sr_ratio=sr_ratios[i], drop_path=dp_rates[cur+j], 
                                          drop=drop, attn_drop=attn_drop, kernel_size=kernel_sizes[i]))
            cur += depths[i]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for i, block in enumerate(self.layers):
            x = block(x)
        return x


class Stem(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class CATFormer(nn.Module):
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
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, depths=[3, 3, 3, 3], dims=[64, 128, 320, 512], 
                 num_heads=[1, 2, 5, 8], sr_ratios=[4, 2, 1, 1], kernel_sizes=[5, 5, 1, 1], drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1., projection=1024
                 ):
        super().__init__()

        self.patch_embed1 = Stem(
                in_chans=in_chans, embed_dim=dims[0])
#        self.patch_embed1 = PatchEmbed(
#                img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=dims[0])
        self.patch_embed2 = PatchEmbed(
                img_size=img_size // 4, patch_size=2, in_chans=dims[0], embed_dim=dims[1])
        self.patch_embed3 = PatchEmbed(
                img_size=img_size // 8, patch_size=2, in_chans=dims[1], embed_dim=dims[2])
        self.patch_embed4 = PatchEmbed(
                img_size=img_size // 16, patch_size=2, in_chans=dims[2], embed_dim=dims[3])

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stage = Stage(istage=i, depths=depths, dims=dims, num_heads=num_heads, sr_ratios=sr_ratios, kernel_sizes=kernel_sizes,
                          dp_rates=dp_rates, drop=drop_rate, attn_drop=attn_drop_rate)
            self.stages.append(stage)

        self.proj = nn.Conv2d(dims[-1], projection, 1, 1)
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(projection, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed1(x)
        x = self.stages[0](x)
        x = self.patch_embed2(x)
        x = self.stages[1](x)
        x = self.patch_embed3(x)
        x = self.stages[2](x)
        x = self.patch_embed4(x)
        x = self.stages[3](x)

        x = self.proj(x) #(b c h w)
        x = self.norm(x).flatten(2, 3) #(b c h*w)
        x = self.swish(x)

        x = self.avgpool(x)  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


@register_model
def CATFormer_tiny(pretrained=False,in_22k=False, **kwargs):
    model = CATFormer(depths=[3, 3, 3, 3], dims=[64, 128, 320, 512], num_heads=[2, 4, 8, 16], sr_ratios=[4, 2, 1, 1], kernel_sizes=[5, 5, 1, 1], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def CATFormer_small(pretrained=False,in_22k=False, **kwargs):
    model = CATFormer(depths=[3, 3, 8, 4], dims=[64, 128, 320, 512], num_heads=[2, 4, 8, 16], sr_ratios=[4, 2, 1, 1], kernel_sizes=[5, 5, 1, 1], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def CATFormer_base(pretrained=False, in_22k=False, **kwargs):
    model = CATFormer(depths=[4, 5, 12, 4], dims=[64, 128, 320, 512], num_heads=[2, 4, 8, 16], sr_ratios=[4, 2, 1, 1], kernel_sizes=[5, 5, 1, 1], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def CATFormer_large(pretrained=False, in_22k=False, **kwargs):
    model = CATFormer(depths=[5, 8, 20, 7], dims=[64, 128, 320, 512], num_heads=[2, 4, 8, 16], sr_ratios=[4, 2, 1, 1], kernel_sizes=[5, 5, 1, 1], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def CATFormer_huge(pretrained=False, in_22k=False, **kwargs):
    model = CATFormer(depths=[5, 8, 20, 7], dims=[96, 192, 384, 640], num_heads=[3, 6, 12, 20], sr_ratios=[4, 2, 1, 1], kernel_sizes=[5, 5, 1, 1], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
