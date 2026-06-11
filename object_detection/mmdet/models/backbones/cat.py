# --------------------------------------------------------
# UniFormer
# Copyright (c) 2022 SenseTime X-Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Kunchang Li
# --------------------------------------------------------

from collections import OrderedDict
import math

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.cnn import (Conv2d, build_activation_layer, build_norm_layer,
                      constant_init, normal_init, trunc_normal_init)
from mmcv.runner import (_load_checkpoint, load_state_dict)
from ...utils import get_root_logger
from ..builder import BACKBONES
import random


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


def sparse_query_index(x, H, W, istage):
    B, N, C = x.shape
    idx = []
    if istage == 0:
        idx = list(i*W+j for i in list(range(0,H,4)) for j in list(range(0,W,4))) #1/16
    elif istage == 1:
        idx = list(i*W+j for i in list(range(0,H,2)) for j in list(range(0,W,2))) #1/4
    elif istage == 2:
        idx  = list(range(N))
    elif istage == 3:
        idx  = list(range(N))
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
    def __init__(self, istage, dim, num_head=8, mlp_ratio=4., sr_ratio=1, drop_path=0., drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=5):
        super().__init__()
        self.istage = istage
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

        idx = sparse_query_index(x, H, W, self.istage)

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
                    layers.append(TCBlock(istage=i, dim=dims[i], num_head=num_heads[i], mlp_ratio=mlp_ratio, sr_ratio=sr_ratios[i], drop_path=dp_rates[cur+j], 
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
#        assert H == self.img_size[0] and W == self.img_size[1], \
#            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x
    

@BACKBONES.register_module()   
class CATFormer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, depths=[3, 3, 3, 3], dims=[64, 128, 320, 512], 
                 num_heads=[1, 2, 5, 8], sr_ratios=[4, 2, 1, 1], kernel_sizes=[5, 5, 1, 1], drop_rate=0., attn_drop_rate=0., drop_path_rate=0., projection=1024, 
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 norm_after_stage=True,
                 with_cp=False,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None):
        """
        Args:
            layer (list): number of block in each layer
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            head_dim (int): dimension of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
                Default: -1 (-1 means not freezing any parameters).
            with_cp (bool, optional): Use checkpoint or not. Using checkpoint
                will save some memory while slowing down the training speed.
                Default: False.
            norm_cfg (dict): Config dict for normalization layer at
                output of backone. Defaults: dict(type='LN').
            pretrained (str, optional): model pretrained path. Default: None.
            init_cfg (dict, optional): The Config for initialization.
                Defaults to None.
        """
        super().__init__()

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        self.num_classes = num_classes
        self.num_features = self.dims = dims  # num_features for consistency with other models
        self.projection = projection
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.norm_after_stage = norm_after_stage
        self.with_cp = with_cp

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

        self.norms = nn.ModuleList()
        for i in range(4):
            if norm_after_stage:
                self.norms.append(build_norm_layer(norm_cfg, dims[i])[1])
            else:
                self.norms.append(nn.Identity())

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(CATFormer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed1.eval()
            for param in self.patch_embed1.parameters():
                param.requires_grad = False

        for i in range(0, self.frozen_stages):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if self.norm_after_stage and (i in self.out_indices):
                norm_layer = self.norms[i]
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            checkpoint = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            logger.warn(f'Load pre-trained model for '
                        f'{self.__class__.__name__} from original repo')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward_features(self, x):
        out = []
        x = self.patch_embed1(x)
        x = self.stages[0](x)
        x_out = self.norms[0](x.permute(0, 2, 3, 1))
        if 0 in self.out_indices:
            out.append(x_out.permute(0, 3, 1, 2).contiguous())
        x = self.patch_embed2(x)
        x = self.stages[1](x)
        x_out = self.norms[1](x.permute(0, 2, 3, 1))
        if 1 in self.out_indices:
            out.append(x_out.permute(0, 3, 1, 2).contiguous())
        x = self.patch_embed3(x)
        x = self.stages[2](x)
        x_out = self.norms[2](x.permute(0, 2, 3, 1))
        if 2 in self.out_indices:
            out.append(x_out.permute(0, 3, 1, 2).contiguous())
        x = self.patch_embed4(x)
        x = self.stages[3](x)
        x_out = self.norms[3](x.permute(0, 2, 3, 1))
        if 3 in self.out_indices:
            out.append(x_out.permute(0, 3, 1, 2).contiguous())
        return tuple(out)

    def forward(self, x):
        x = self.forward_features(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.proj = nn.Conv2d(dims[-1], self.projection, 1, 1)
        self.norm = nn.BatchNorm2d(self.projection)
        self.swish = MemoryEfficientSwish()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.projection, num_classes) if num_classes > 0 else nn.Identity()
