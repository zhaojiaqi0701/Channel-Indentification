import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint1
import checkpoint1
from torch.nn.modules.utils import _triple

import numpy as np
from mmengine import to_3tuple
from mmengine.runner import load_checkpoint

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.autograd import Variable
from mmengine.logging import MMLogger
# from utils.checkpoint import load_checkpoint
# from mmseg.utils import get_root_logger

# from mmseg.utils import logging

from module import Attention, PreNorm, FeedForward, CrossAttention

groups = 32


# 定义 MLP 网络结构
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


# 分块和恢复操作适配三维数据
def window_partition(x, window_size):
    B, H, W, D, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W, D):
    B = int(windows.shape[0] / (H * W * D  / window_size**3))
    x = windows.view(B, H // window_size, W // window_size, D // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, D, -1)
    return x


def get_root_logger(log_file=None, log_level='INFO'):
    logger = MMLogger.get_instance("mmseg", log_file=log_file, log_level=log_level)
    return logger




# 定义 3D 窗口注意力
class WindowAttention3D(nn.Module):
    """3D Window-based multi-head self-attention (W-MSA) with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height, width, and depth of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (H, W, D)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table for relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )  # (2*H-1) * (2*W-1) * (2*D-1), nH

        # Get pair-wise relative position index for each token inside the 3D window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_d = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d], indexing="ij"))  # 3, H, W, D
        coords_flatten = torch.flatten(coords, 1)  # 3, H*W*D
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, H*W*D, H*W*D
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # H*W*D, H*W*D, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # H*W*D, H*W*D
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: Input features with shape of (num_windows*B, N, C).
            mask: (0/-inf) mask with shape of (num_windows, H*W*D, H*W*D) or None.
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            -1
        )  # (H*W*D, H*W*D, nH)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nH, H*W*D, H*W*D)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Swin Transformer的基本块，适应三维数据
class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size  # 三维窗口尺寸
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 标准化与注意力层
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        # 全连接层
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.D = None
    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        D, H, W = self.D, self.H, self.W
        assert L == H * W  * D, f"输入特征图大小错误：L={L}, D*H*W={D * H * W}"

        shortcut = x
        x = self.norm1(x)
        # 特征图形状重构
        x = x.view(B, H, W, D, C)
        # pad feature maps to multiples of window size

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        pad_d = (self.window_size - D % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_h, 0 ,pad_w, 0, pad_d))  # 顺序为深度、高度、宽度的填充
        _,Hp, Wp ,Dp,_= x.shape
        if self.shift_size>0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size),
                                   dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # 窗口分割
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size*self.window_size*self.window_size, C)

        # 自注意力
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size,self.window_size ,self.window_size, C)

        # 窗口还原
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Dp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x = shifted_x

        if any([pad_d, pad_h, pad_w]):
            x = x[:, :H, :W ,:D, :].contiguous()

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = x[:,:H, :W ,:D, :].contiguous()

        x = x.view(B, H * W * D, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchRecover3D(nn.Module):
    """3D Patch Recover Layer for upsampling.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=dim // 2, num_groups=groups),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, H, W , D):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D*H*W, C).
            D, H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        x = x.permute(0, 2, 1)  # B, C, L
        x = x.view(B, C, H, W , D)
        x = self.up(x)  # B, C//2, D*2, H*2, W*2

        x = x.view(B, C // 2, -1).permute(0, 2, 1)

        return x


class PatchMerging3D(nn.Module):
    """3D Patch Merging Layer for downsampling.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, H, W, D):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W*D, C).
            H, W, D: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W * D, "input feature has wrong size"

        # Reshape x to (B, H, W, D, C)
        x = x.view(B, H, W, D, C)

        # Padding if necessary to ensure even dimensions
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (D % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, D % 2, 0, W % 2, 0, H % 2))

        # Splitting into eight neighboring cubes along H, W, and D
        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 D/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 D/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 D/2 C
        x3 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 D/2 C
        x4 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 D/2 C
        x5 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 D/2 C
        x6 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 D/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 D/2 C

        # Concatenate the splits along the last dimension
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)  # B H/2 W/2 D/2 8*C
        x = x.view(B, -1, 8 * C)  # B (H/2)*(W/2)*(D/2) 8*C

        # Apply normalization and reduction
        x = self.norm(x)
        x = self.reduction(x)

        return x



# class BasicLayer(nn.Module):
#     def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.,
#                  qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
#                  use_checkpoint=False,up=True):
#         super().__init__()
#         self.window_size = _triple(window_size)
#         self.shift_size = _triple(window_size // 2)
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint
#         self.up = up
#         # 构建 Swin Transformer 块
#         self.blocks = nn.ModuleList([
#             SwinTransformerBlock3D(
#                 dim=dim,
#                 num_heads=num_heads,
#                 window_size=self.window_size[0],  # 使用三维窗口大小
#                 shift_size=(0 if (i % 2 == 0) else self.shift_size[0]),
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop,
#                 attn_drop=attn_drop,
#                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                 norm_layer=norm_layer
#             ) for i in range(depth)
#         ])
#
#         self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None
#
#     def forward(self, x, D, H, W):
#         """3D 输入的前向传播."""
#         # 创建注意力掩码
#         Dp, Hp, Wp = [int(np.ceil(size / ws)) * ws for size, ws in zip((D, H, W), self.window_size)]
#         img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=x.device)
#
#         d_slices = (slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None))
#         h_slices = (slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None))
#         w_slices = (slice(0, -self.window_size[2]), slice(-self.window_size[2], -self.shift_size[2]), slice(-self.shift_size[2], None))
#         cnt = 0
#         for d in d_slices:
#             for h in h_slices:
#                 for w in w_slices:
#                     img_mask[:, d, h, w, :] = cnt
#                     cnt += 1
#
#         mask_windows = window_partition(img_mask, self.window_size[0])
#         mask_windows = mask_windows.view(-1, self.window_size[0]**3)
#         attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#         attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#
#         # 在每个块中传入 D, H, W
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint1.checkpoint(blk, x, attn_mask, D, H, W)
#             else:
#                 x = blk(x, attn_mask, D, H, W)
#
#         if self.downsample:
#             x_down = self.downsample(x, D, H, W)
#             if self.up:
#                 D, H, W = D // 2, H // 2, W // 2
#             else:
#                 D, H, W = D * 2, H * 2,W * 2
#             return x, D, H, W, x_down
#         else:
#             return x, D, H, W, x, D, H, W

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage in 3D data."""

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 up=True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2  # Shift size for SW-MSA
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.up = up

        # 构建3D Swin Transformer块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=self.window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # Patch merging/downsampling layer
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, H, W, D):
        """ Forward function for 3D input.
        Args:
            x: Input feature, tensor size (B, D*H*W, C).
            D, H, W: Spatial resolution of the input feature.
        """

        # 计算 SW-MSA 的注意力掩码

        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        Dp = int(np.ceil(D / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, Dp,  1), device=x.device)

        # 定义深度、高度和宽度方向的切片
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        d_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0

        for h in h_slices:
            for w in w_slices:
                for d in d_slices:
                    img_mask[:, h, w, d, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W, blk.D= H, W, D
            if self.use_checkpoint:
                x = checkpoint1.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W, D)
            if self.up:
                Wh, Ww,Wd  = (H + 1) // 2, (W + 1) // 2, (D + 1) // 2
            else:
                Wh, Ww,Wd  =  H * 2, W * 2,D * 2
            return x, H, W, D, x_down, Wh, Ww, Wd
        else:
            return x, H, W, D, x, H, W, D


# 3D Patch嵌入层
# class PatchEmbed3D(nn.Module):
#
#     def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         self.patch_size = to_3tuple(patch_size)  # Use a 3D tuple for patch size
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#
#         # 3D convolution for patch embedding
#         self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else None
#
#     def forward(self, x):
#         """Forward function."""
#         # padding for depth, height, and width
#         _, _, D, H, W = x.size()
#         pad_d = (self.patch_size[0] - D % self.patch_size[0]) % self.patch_size[0]
#         pad_h = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
#         pad_w = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]
#
#         # Apply padding if needed
#         if pad_d > 0 or pad_h > 0 or pad_w > 0:
#             x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
#
#         # Perform the 3D convolution
#         x = self.proj(x)  # Output shape: (B, embed_dim, D', H', W')
#
#         # Apply normalization if specified
#         if self.norm is not None:
#             D, H, W = x.shape[2:]
#             x = x.flatten(2).transpose(1, 2)  # Shape: (B, D'*H'*W', embed_dim)
#             x = self.norm(x)
#             x = x.transpose(1, 2).view(-1, self.embed_dim, D, H, W)  # Restore shape for 3D
#
#         return x


# 3D Patch 嵌入层
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size =  to_3tuple(patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

        self.embed_dim = embed_dim
    def forward(self, x):
        _, _, H, W, D = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        if D % self.patch_size[2] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[2] - D % self.patch_size[2]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww ,Wd= x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wd)

        return x
class MultiEmbed3D(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=1)
        self.bn = nn.BatchNorm3d(embed_dim)
        self.maxPool = nn.MaxPool3d(kernel_size=(patch_size[0], patch_size[1], patch_size[2]), stride=(patch_size[0], patch_size[1], patch_size[2]))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W, D = x.size()
        if D % self.patch_size[2] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[2] - D % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C D H W
        x = self.bn(x)
        x = self.maxPool(x)
        if self.norm is not None:
            Wh, Ww, Wd= x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wd)

        return x


class SwinTransformer(nn.Module):
    """3D Swin Transformer backbone.

    Args:
        (参数与原始定义相同，仅移除预训练参数加载部分)
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=1,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.5,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()
        # self.window_size = _triple(window_size)
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # Split input into non-overlapping patches (3D version)
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # Absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(pretrain_img_size)  # Adjust to 3D tuple
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(  # Ensure BasicLayer3D is used here for 3D compatibility
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging3D if (i_layer < self.num_layers - 1) else None,  # Use PatchMerging3D
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # Define output features for each stage
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # Add normalization layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

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
    def forward(self, x):
        """Forward function for 3D input."""
        x = self.patch_embed(x)

        Wh, Ww ,Wd= x.size(2), x.size(3), x.size(4)
        if self.ape:
            # Interpolate the position embedding to the input size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=( Wh, Ww,Wd), mode='trilinear', align_corners=True)
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B D*H*W C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        # print(x)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            # print('111')
            # print(layer)
            x_out, H, W, D, x, Wh, Ww, Wd  = layer(x, Wh, Ww , Wd)
            # print('222')
            # print(layer)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, D, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                outs.append(out)

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


class up_conv3d(nn.Module):
    """
    Up Convolution Block for 3D.
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv3d, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True),  # 3D upsampling
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),    # 3D convolution
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
class Decoder3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder3D, self).__init__()
        self.up = up_conv3d(in_channels, out_channels)
        # self.up = nn.ConvTranspose3d(in_channels, middle_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        print("111Before upsample - x1:", x1.shape, "x2:", x2.shape)
        x1 = self.up(x1)
        print("111After upsample x1:", x1.shape)
        print("111x2 shape:", x2.shape)
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.conv_relu(x1)
        print("111x1 shape:", x1.shape)
        return x1


class conv_block(nn.Module):
    """
    3D Convolution Block
    """
    def __init__(self, in_ch, out_ch, groups=8):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        print("x shape:", x.shape)
        return x

# 3D卷积块
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_channels=out_ch, num_groups=8),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_channels=out_ch, num_groups=8),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # return self.conv(x)
        x = self.conv(x)
        print("x shape1:", x.shape)
        return x
class SwinUp3D(nn.Module):
    def __init__(self, dim):
        super(SwinUp3D, self).__init__()
        self.up = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, D):
        B, L, C = x.shape
        assert L == H * W * D , "input feature has wrong size"

        # 对输入进行标准化和上采样
        x = self.norm(x)
        x = self.up(x)
        x = x.view(B, H, W, D, 2 * C)  # reshape为3D

        # 将通道分割为4个部分
        x0 = x[:, :, :, :, 0:C // 2]
        x1 = x[:, :, :, :, C // 2:C]
        x2 = x[:, :, :, :, C:C + C // 2]
        x3 = x[:, :, :, :, C + C // 2:C * 2]

        # 将分块后的张量在深度维度、宽度和高度方向上进行拼接
        x0 = torch.cat((x0, x1), dim=1)  # 在深度方向拼接
        x3 = torch.cat((x2, x3), dim=1)  # 在深度方向拼接
        x = torch.cat((x0, x3), dim=2)   # 在高度方向拼接

        x = x.view(B, -1, C // 2)  # 恢复形状
        return x


class SwinDecoder3D(nn.Module):

    def __init__(self,
                 embed_dim,
                 patch_size=4,
                 depths=2,
                 num_heads=6,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False):
        super(SwinDecoder3D, self).__init__()

        self.patch_norm = patch_norm

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # Stochastic depth decay rule

        # Build layers
        self.layer = BasicLayer(
            dim=embed_dim // 2,
            depth=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint)

        # 3D Upsampling
        self.up = up_conv3d(embed_dim, embed_dim // 2)

        # Convolutional layer after upsampling
        self.conv_relu = nn.Sequential(
            nn.Conv3d(embed_dim // 2, embed_dim // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        """Forward function."""
        identity = x
        B, C, H, W , D= x.shape
        x = self.up(x)  # B, C//2, 2D, 2H, 2W
        x = x.reshape(B, C // 2, H * W * D * 8)
        x = x.permute(0, 2, 1)

        x_out, H, W, D, x, Hp, Wp, Dp = self.layer(x, H * 2, W * 2, D * 2)

        x = x.permute(0, 2, 1)
        x = x.reshape(B, C // 2, H, W, D)

        # Apply 1x1x1 convolution and ReLU
        x = self.conv_relu(x)

        return x


class Swin_Decoder3D(nn.Module):
    def __init__(self, in_channels, depths, num_heads):
        super(Swin_Decoder3D, self).__init__()
        self.up = SwinDecoder3D(in_channels, depths=depths, num_heads=num_heads)

        # 3D Convolutional layers
        self.conv_relu = nn.Sequential(
            nn.Conv3d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels // 2, in_channels // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        print("222Before upsample - x1:", x1.shape, "x2:", x2.shape)
        x1 = self.up(x1)
        x2 = self.conv2(x2)
        print("222After upsample x1:", x1.shape)  # 确保 x1 被正确上采样

        # 拼接前再确认形状
        print("222Shapes before concat - x1:", x1.shape, "x2:", x2.shape)
        x1 = torch.cat((x2, x1), dim=1)  # Concatenate along channel dimension
        out = self.conv_relu(x1)
        print("222Output shape:", out.shape)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Cross_Att(nn.Module):
    def __init__(self, dim_s, dim_l):
        super().__init__()
        self.transformer_s = Transformer(dim=dim_s, depth=1, heads=3, dim_head=32, mlp_dim=128)
        self.transformer_l = Transformer(dim=dim_l, depth=1, heads=1, dim_head=64, mlp_dim=256)
        self.norm_s = nn.LayerNorm(dim_s)
        self.norm_l = nn.LayerNorm(dim_l)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear_s = nn.Linear(dim_s, dim_l)
        self.linear_l = nn.Linear(dim_l, dim_s)

    def forward(self, e, r):
        # Adapt for 3D structure with (h, w, d)
        b_e, c_e, h_e, w_e, d_e = e.shape
        e = e.reshape(b_e, c_e, -1).permute(0, 2, 1)  # Flatten 3D spatial dimensions

        b_r, c_r, h_r, w_r, d_r = r.shape
        r = r.reshape(b_r, c_r, -1).permute(0, 2, 1)  # Flatten 3D spatial dimensions

        # Process e and r using average pooling, normalization, and linear transformations
        e_t = torch.flatten(self.avgpool(self.norm_l(e).transpose(1, 2)), 1)
        r_t = torch.flatten(self.avgpool(self.norm_s(r).transpose(1, 2)), 1)
        e_t = self.linear_l(e_t).unsqueeze(1)
        r_t = self.linear_s(r_t).unsqueeze(1)

        # Concatenate with transformer processing
        r = self.transformer_s(torch.cat([e_t, r], dim=1))[:, 1:, :]
        e = self.transformer_l(torch.cat([r_t, e], dim=1))[:, 1:, :]

        # Reshape back to original 3D spatial dimensions
        e = e.permute(0, 2, 1).reshape(b_e, c_e, h_e, w_e, d_e)
        r = r.permute(0, 2, 1).reshape(b_r, c_r, h_r, w_r, d_r)

        return e, r


class UNet3D(nn.Module):
    def __init__(self, dim, n_class, in_ch=1):
        super().__init__()
        # 假设 SwinTransformer 已经适配为三维版
        self.encoder = SwinTransformer(depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], drop_path_rate=0.5,
                                       embed_dim=128)
        self.encoder2 = SwinTransformer(depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], drop_path_rate=0.2, patch_size=8,
                                        embed_dim=96)

        # 解码层
        self.layer1 = Swin_Decoder3D(8 * dim, 2, 8)
        self.layer2 = Swin_Decoder3D(4 * dim, 2, 4)
        self.layer3 = Swin_Decoder3D(2 * dim, 2, 2)
        self.layer4 = Decoder3D(dim, dim, dim // 2)
        self.layer5 = Decoder3D(dim // 2, dim // 2, dim // 4)

        # 下采样层和卷积块
        self.down1 = nn.Conv3d(in_ch, dim // 4, kernel_size=1, stride=1, padding=0)
        self.down2 = conv_block(dim // 4, dim // 2)  # ConvBlock3D 应该是 3D 卷积的版本
        self.final = nn.Conv3d(dim // 4, n_class, kernel_size=1, stride=1, padding=0)

        # 损失层
        self.loss1 = nn.Sequential(
            nn.Conv3d(dim * 8, n_class, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=(32, 32, 32), mode='trilinear', align_corners=True)
        )
        self.loss2 = nn.Sequential(
            nn.Conv3d(dim, n_class, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=True)
        )

        # 其他 3D 层
        dim_s = 96
        dim_l = 128
        self.m1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.m2 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        tb = dim_s + dim_l
        self.change1 = ConvBlock3D(tb, dim)
        self.change2 = ConvBlock3D(tb * 2, dim * 2)
        self.change3 = ConvBlock3D(tb * 4, dim * 4)
        self.change4 = ConvBlock3D(tb * 8, dim * 8)

        # Cross Attention 层
        self.cross_att_1 = Cross_Att(dim_s * 1, dim_l * 1)
        self.cross_att_2 = Cross_Att(dim_s * 2, dim_l * 2)
        self.cross_att_3 = Cross_Att(dim_s * 4, dim_l * 4)
        self.cross_att_4 = Cross_Att(dim_s * 8, dim_l * 8)

    def forward(self, x):
        # 编码阶段
        out = self.encoder(x)
        out2 = self.encoder2(x)
        e1, e2, e3, e4 = out[0], out[1], out[2], out[3]
        r1, r2, r3, r4 = out2[0], out2[1], out2[2], out2[3]

        # Cross Attention
        e1, r1 = self.cross_att_1(e1, r1)
        e2, r2 = self.cross_att_2(e2, r2)
        e3, r3 = self.cross_att_3(e3, r3)
        e4, r4 = self.cross_att_4(e4, r4)

        # 拼接和变换
        e1 = torch.cat([e1, self.m1(r1)], 1)
        e2 = torch.cat([e2, self.m1(r2)], 1)
        e3 = torch.cat([e3, self.m1(r3)], 1)
        e4 = torch.cat([e4, self.m1(r4)], 1)
        e1 = self.change1(e1)
        e2 = self.change2(e2)
        e3 = self.change3(e3)
        e4 = self.change4(e4)

        # 损失计算
        loss1 = self.loss1(e4)

        # 解码阶段
        ds1 = self.down1(x)
        ds2 = self.down2(ds1)
        d1 = self.layer1(e4, e3)
        d2 = self.layer2(d1, e2)
        d3 = self.layer3(d2, e1)
        loss2 = self.loss2(d3)
        d4 = self.layer4(d3, ds2)
        d5 = self.layer5(d4, ds1)

        # 输出
        o = self.final(d5)
        return o, loss1, loss2


# if __name__ == '__main__':
#     print('#### Test Case ###')
#     from torch.autograd import Variable
#
#     # x = Variable(torch.rand(2, 3,64,64)).cuda()
#     # model = UNet(128, 1).cuda()
#     x = Variable(torch.rand(2, 3, 64, 64))
#     model = UNet(128, 1)
#     print("Input shape:", x.shape)
#     y = model(x)
#     print('Output shape:', y[-1].shape)

if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable

    # 生成三维输入数据 (batch_size=2, channels=3, depth=64, height=64, width=64)
    x = Variable(torch.rand(2, 1, 128, 128, 128))
    model = UNet3D(128, 1)  # 确保使用的是三维版本的UNet模型
    print("Input shape:", x.shape)
    y = model(x)
    print('Output shape:', y[-1].shape)
