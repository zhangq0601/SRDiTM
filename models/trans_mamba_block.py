import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange
from timm.models.layers import DropPath
from .trans_mamba import normalization, WindowAttention, window_partition, window_reverse, to_2tuple


class MambaBlock(nn.Module):

    def __init__(self, dim, d_state=16, d_conv=4, expand=2., drop=0.):
        super().__init__()
        self.dim = dim
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=drop,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.mamba(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class TransMambaBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=normalization, emb_channels=640, fft_patch_size=8,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm_mamba = norm_layer(dim)
        self.mamba = MambaBlock(dim=dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, 1),
            act_layer(),
            nn.Dropout(drop),
            nn.Conv2d(mlp_hidden_dim, dim, 1, 1),
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

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, 1, H, W))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size).permute(0, 2, 3, 1).contiguous()
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, t):
        B, C, Ph, Pw = x.shape
        x_size = (Ph, Pw)
        x_type = x.dtype
        shortcut = x

        x = self.norm1(x)
        scale_msa = self.adaLN_scale_msa(t).reshape(t.shape[0], 1, 1, 1, self.fft_patch_size,
                                                    self.fft_patch_size // 2 + 1)
        x_patch = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.fft_patch_size, p2=self.fft_patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft *= scale_msa
        x = torch.fft.irfft2(x_patch_fft, s=(self.fft_patch_size, self.fft_patch_size)).to(x_type)
        x = rearrange(x, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.fft_patch_size, p2=self.fft_patch_size)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask.to(x.dtype))
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device, x.dtype))
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Ph, Pw)
        if self.shift_size > 0:
            x_attn = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x_attn = shifted_x

        x_mamba = self.norm_mamba(shortcut)
        x_mamba = self.mamba(x_mamba)

        x = shortcut + self.drop_path(x_attn + x_mamba)
        shortcut = x

        x = self.norm2(x)
        scale_mlp = self.adaLN_scale_mlp(t).reshape(t.shape[0], 1, 1, 1, self.fft_patch_size,
                                                    self.fft_patch_size // 2 + 1)
        x_patch = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.fft_patch_size, p2=self.fft_patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft *= scale_mlp
        x = torch.fft.irfft2(x_patch_fft, s=(self.fft_patch_size, self.fft_patch_size)).to(x_type)
        x = rearrange(x, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.fft_patch_size, p2=self.fft_patch_size)
        x = shortcut + self.drop_path(self.mlp(x))

        return x


class TransMambaLayer(nn.Module):

    def __init__(self, in_chans, embed_dim, num_heads, window_size, depth=6,
                 img_size=224, patch_size=1, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=normalization, use_checkpoint=False,
                 patch_norm=True, time_embed_dim=640, **kwargs):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            normalization(embed_dim) if patch_norm else nn.Identity()
        )
        self.patch_unembed = nn.Sequential(
            normalization(embed_dim) if patch_norm else nn.Identity(),
            nn.Conv2d(embed_dim, in_chans, kernel_size=1, stride=1)
        )

        self.blocks = nn.ModuleList([
            TransMambaBlock(
                dim=embed_dim,
                input_resolution=(img_size // patch_size, img_size // patch_size),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                emb_channels=time_embed_dim,
                **kwargs
            ) for i in range(depth)
        ])

        self.layer_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * depth, embed_dim, kernel_size=1, stride=1),
            norm_layer(embed_dim),
            nn.GELU()
        )

    def forward(self, x, t):
        x = self.patch_embed(x)
        layer_feats = []
        for block in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(block, x, t)
            else:
                x = block(x, t)
            layer_feats.append(x)

        fused_feat = torch.cat(layer_feats, dim=1)
        fused_feat = self.layer_fusion(fused_feat)
        x = x + fused_feat  #
        x = self.patch_unembed(x)
        return x