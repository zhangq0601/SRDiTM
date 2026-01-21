
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from abc import abstractmethod
from basic_ops import normalization
from mamba_ssm import Mamba
from .trans_mamba_block import TransMambaBlock

import pywt

def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x
def drop_path_f(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
class TimestepBlock(nn.Module):

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):

    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x


class WindowAttention(nn.Module):


    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple), B_ x H x N x C

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0).to(attn.dtype)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed(nn.Module):

    def __init__(
            self,
            in_chans,
            img_size=224,
            patch_size=4,
            embed_dim=96,
            patch_norm=False,
            ):
        super().__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if patch_norm:
            self.norm = normalization(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):

        x = self.proj(x)  # B x embed_dim x Ph x Pw
        x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):

    def __init__(self, out_chans, embed_dim=96, patch_norm=False):
        super().__init__()
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(embed_dim, out_chans, kernel_size=1, stride=1)
        if patch_norm:
            self.norm = normalization(out_chans)
        else:
            self.norm = nn.Identity()

    def forward(self, x):

        x = self.norm(self.proj(x))
        return x
 

class TransMambaBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=normalization, emb_channels=160 * 4, fft_patch_size=8, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)

        self.norm_mamba = norm_layer(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )


        self.fft_patch_size = min(fft_patch_size, min(input_resolution))
        self.adaLN_scale_msa = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.fft_patch_size * (self.fft_patch_size // 2 + 1))
        )
        self.adaLN_scale_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.fft_patch_size * (self.fft_patch_size // 2 + 1))
        )

    def forward(self, x, t):

        B, C, H, W = x.shape
        x_size = (H, W)
        x_type = x.dtype
        shortcut = x


        scale_msa = self.adaLN_scale_msa(t).reshape(B, 1, 1, 1, self.fft_patch_size, self.fft_patch_size // 2 + 1)
        x_patch = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.fft_patch_size, p2=self.fft_patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft *= scale_msa
        x = torch.fft.irfft2(x_patch_fft, s=(self.fft_patch_size, self.fft_patch_size)).to(x_type)
        x = rearrange(x, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.fft_patch_size, p2=self.fft_patch_size)


        x = self.norm1(x.permute(0, 2, 3, 1)).reshape(B, H * W, C)  # B x (H*W) x C
        attn_out, _ = self.attn(x, x, x)
        x = shortcut + self.drop_path(attn_out.reshape(B, H, W, C).permute(0, 3, 1, 2))  # 残差


        shortcut_mamba = x
        x = self.norm_mamba(x.permute(0, 2, 3, 1)).reshape(B, H * W, C)  # B x (H*W) x C
        x = self.mamba(x)
        x = shortcut_mamba + self.drop_path(x.reshape(B, H, W, C).permute(0, 3, 1, 2))  # 残差


        scale_mlp = self.adaLN_scale_mlp(t).reshape(B, 1, 1, 1, self.fft_patch_size, self.fft_patch_size // 2 + 1)
        x_patch = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.fft_patch_size, p2=self.fft_patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft *= scale_mlp
        x = torch.fft.irfft2(x_patch_fft, s=(self.fft_patch_size, self.fft_patch_size)).to(x_type)
        x = rearrange(x, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.fft_patch_size, p2=self.fft_patch_size)


        shortcut_ffn = x
        x = self.norm2(x.permute(0, 2, 3, 1)).reshape(B, H * W, C)
        x = self.mlp(x)
        x = shortcut_ffn + self.drop_path(x.reshape(B, H, W, C).permute(0, 3, 1, 2))  # 残差

        return x


class TransMambaLayer(nn.Module):

    def __init__(self, in_chans, embed_dim, num_heads, window_size, depth=2,
                 img_size=64, patch_size=1, mlp_ratio=2.0, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=normalization, patch_norm=False,
                 swin_attn_type='AdaLN', time_embed_dim=640, **kwargs):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim

        self.blocks = nn.ModuleList([
            TransMambaBlock(
                dim=embed_dim,
                input_resolution=(img_size // patch_size, img_size // patch_size),
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                emb_channels=time_embed_dim,
                **kwargs
            ) for i in range(depth)
        ])

        self.fusion = nn.Conv2d(embed_dim * depth, embed_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t):

        features = []
        for block in self.blocks:
            x = block(x, t)
            features.append(x)

        concat_feat = torch.cat(features, dim=1)  # B x (C*depth) x H x W
        fused_feat = self.fusion(concat_feat)
        x = x + fused_feat
        return x

class BasicLayer(nn.Module):

        def __init__(
                self,
                in_chans,
                embed_dim,
                num_heads,
                window_size,
                depth=2,
                img_size=224,
                patch_size=4,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=normalization,
                use_checkpoint=False,
                patch_norm=True,
                time_embed_dim=160 * 4,
                patch_emb=True,
                mamba_d_state=16,
                mamba_d_conv=4,
                mamba_expand=2.,
                **kwargs,
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.depth = depth
            self.use_checkpoint = use_checkpoint
            self.patch_emb = patch_emb

            if self.patch_emb:
                self.patch_embed = PatchEmbed(
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    img_size=img_size,
                    patch_size=patch_size,
                    patch_norm=patch_norm,
                )
                self.patch_unembed = PatchUnEmbed(
                    out_chans=in_chans,
                    embed_dim=embed_dim,
                    patch_norm=patch_norm,
                )

            input_resolution = self.patch_embed.patches_resolution if self.patch_emb else (img_size, img_size)
            self.input_resolution = input_resolution

            self.blocks = nn.ModuleList([
                TransMambaBlock(
                    dim=embed_dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    emb_channels=time_embed_dim,
                    mamba_d_state=mamba_d_state,
                    mamba_d_conv=mamba_d_conv,
                    mamba_expand=mamba_expand,
                    **kwargs,
                ) for i in range(depth)
            ])

            self.layer_fusion = nn.Sequential(
                nn.Conv2d(embed_dim * depth, embed_dim, 1, 1),
                norm_layer(embed_dim),
                nn.GELU()
            )

        def forward(self, x, t):
            if self.patch_emb:
                x = self.patch_embed(x)

            layer_feats = []
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x, t)
                else:
                    x = blk(x, t)
                layer_feats.append(x)

            fused_feat = torch.cat(layer_feats, dim=1)
            fused_feat = self.layer_fusion(fused_feat)
            x = x + fused_feat

            if self.patch_emb:
                x = self.patch_unembed(x)
            return x


