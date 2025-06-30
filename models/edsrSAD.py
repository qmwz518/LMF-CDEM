git"""
Modified from: https://github.com/thstkdgus35/EDSR-PyTorch
"""

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register

from torchvision.ops import DeformConv2d

from models.FDConv import FDConv
# qumu
class DeformableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DeformableConvBlock, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size, padding=padding)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 生成偏移量
        offset = self.offset_conv(x)
        # 可变形卷积
        out = self.deform_conv(x, offset)
        out = self.relu(out)
        return out

# 引入空间注意力 by qumu
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(feat))
        return x * attn

class ResidualDilatedBlockWithSA(nn.Module):
    def __init__(self, channels, dilation, res_scale=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out = self.sa(out)
        out += residual
        return out

# 自适应膨胀率学习
class AdaptiveDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, Bias=True,Dilation = 1): #dilation 参数只是为了参数兼容
        super().__init__()
        self.dilation = nn.Parameter(torch.tensor(1.0))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=Bias)

    def forward(self, x):
        d = self.dilation.round().int().item()
        padding = d
        return F.conv2d(x, self.conv.weight, padding=padding, dilation=d)

              

class AdaptiveDilatedBlock(nn.Module):
    def __init__(self, channels, act=nn.LeakyReLU(0.2, inplace=True),res_scale = 1):
        super().__init__()
        self.conv1 = AdaptiveDilatedConv(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = act
        self.res_scale = res_scale

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


def default_conv(in_channels, out_channels, kernel_size, Bias=True, Dilation = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), dilation=Dilation, bias=Bias)

def fd_conv(in_channels, out_channels, kernel_size, Bias=True, Dilation = 1):
    return FDConv(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), dilation=Dilation, bias=Bias)


# class MeanShift(nn.Conv2d):
#     def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

#         super(MeanShift, self).__init__(3, 3, kernel_size=1)
#         std = torch.Tensor(rgb_std)
#         self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
#         self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
#         for p in self.parameters():
#             p.requires_grad = False



class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, Bias=True, bn=False, act=nn.ReLU(), res_scale=1, spatialatt = True, Dilation = 1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, Bias=Bias,Dilation=Dilation))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.sa = SpatialAttention() if spatialatt else None


    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        if self.sa is not None:
          res = self.sa(res)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        sa = args.spatialatt
        dilation = args.dilation 
        bias = True
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        # print(f'erdsSD183 n_feats={n_feats}')#qumu
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        # self.sub_mean = MeanShift(args.rgb_range)
        # self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        # m_head = [conv(args.n_colors, n_feats, kernel_size)]
        m_head = [nn.Conv2d(args.n_colors, n_feats, kernel_size,
        padding=(kernel_size//2), dilation=dilation,bias=bias)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ,Dilation = dilation) for _ in range(n_resblocks)
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x, trend=None):
        #x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x

        if trend is not None:
            res += trend # qumud
        if not self.args.no_upsampling:
            res = self.tail(res)
        #x = self.add_mean(x)
        return res

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


@register('edsr-trend-baseline')
def make_edsr_baseline(n_resblocks=16, n_colors=1, n_feats=64, res_scale=1,
                       scale=4, no_upsampling=False, rgb_range=1,spatialatt=True,conv_type='default',dilation=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.spatialatt = spatialatt
    args.rgb_range = rgb_range
    args.n_colors = n_colors
    args.dilation = dilation
    if conv_type == 'default':
      return EDSR(args)
    elif conv_type == 'fdconv':
      return EDSR(args, conv=fd_conv)
    elif conv_type == 'adapdi':
      return EDSR(args, conv=AdaptiveDilatedConv)
    elif conv_type == 'deform':
      return EDSR(args, conv=DeformConv2d)



@register('edsr-adaptiveDi-baseline')
def make_edsr_adaptiveDi_baseline(n_resblocks=16, n_colors=1, n_feats=64, res_scale=1,
                       scale=4, no_upsampling=False, rgb_range=1,spatialatt=True):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.spatialatt = spatialatt
    args.rgb_range = rgb_range
    args.n_colors = n_colors
    return EDSR(args,conv=AdaptiveDilatedConv)

@register('edsr-trend')
def make_edsr(n_resblocks=32, n_colors = 1, n_feats=256, res_scale=0.1,
              scale=2, no_upsampling=False, rgb_range=1,spatialatt=True):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.spatialatt = spatialatt


    args.rgb_range = rgb_range
    args.n_colors = n_colors
    return EDSR(args)
