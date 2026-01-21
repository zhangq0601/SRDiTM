
import torch
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from basic_ops import (
    linear,
    conv_nd,
    avg_pool_nd,
    normalization,
    timestep_embedding,
)
from trans_mamba import BasicLayer, TimestepBlock

try:
    import xformers
    import xformers.ops as xop
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class DiTMSRModel(nn.Module):

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        swin_depth=2,
        swin_embed_dim=96,
        window_size=8,
        mlp_ratio=2.0,
        patch_norm=False,
        cond_lq=True,
        cond_mask=False,
        lq_size=256,
        lq_channels=None,
        swin_attn_type='AdaLN',
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2.,
        **kwargs,
    ):
        super().__init__()

        if num_heads == -1:
            assert swin_embed_dim % num_head_channels == 0 and num_head_channels > 0

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq
        self.cond_mask = cond_mask

        time_embed_dim = model_channels * 4         # 640 = 160 * 4 
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if cond_lq and lq_size == image_size: 
            self.feature_extractor = nn.Identity()
            base_chn = 4 if cond_mask else 3    # lq_channel
            base_chn = lq_channels if lq_channels else base_chn
        else:
            feature_extractor = []
            feature_chn = 4 if cond_mask else 3
            base_chn = 16
                
            for ii in range(int(math.log(lq_size / image_size) / math.log(2))):
                feature_extractor.append(nn.Conv2d(feature_chn, base_chn, 3, 1, 1))
                feature_extractor.append(nn.SiLU())
                feature_extractor.append(Downsample(base_chn, True, out_channels=base_chn*2))
                base_chn *= 2
                feature_chn = base_chn
            self.feature_extractor = nn.Sequential(*feature_extractor)

        ch = input_ch = int(channel_mult[0] * model_channels)   # 160 * 1
        in_channels += base_chn     # 6 = 3+3
        self.unet_in_channels = in_channels #   save for calculate flops     
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = image_size     # 64
        for level, mult in enumerate(channel_mult):
            layers = []
            if ds in attention_resolutions:
                layers.append(
                    BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                            swin_attn_type=swin_attn_type,
                            time_embed_dim=time_embed_dim,
                            mamba_d_state=mamba_d_state,
                            mamba_d_conv=mamba_d_conv,
                            mamba_expand=mamba_expand,
                            **kwargs,
                        )
                )
            self.input_blocks.append(TimestepEmbedSequential(*layers))
            input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = int(channel_mult[level + 1] * model_channels)
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds //= 2

        self.middle_block = TimestepEmbedSequential(
            BasicLayer(
                    in_chans=ch,
                    embed_dim=swin_embed_dim,
                    num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                    window_size=window_size,
                    depth=swin_depth,
                    img_size=ds,
                    patch_size=1,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=dropout,
                    attn_drop=0.,
                    drop_path=0.,
                    use_checkpoint=False,
                    norm_layer=normalization,
                    patch_norm=patch_norm,
                    swin_attn_type=swin_attn_type,
                    time_embed_dim=time_embed_dim,
                    **kwargs,
                     ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            ich = input_block_chans.pop()
            layers = [
                conv_nd(2, ch + ich, int(model_channels * mult), 3, stride=1, padding=1)
            ]
            ch = int(model_channels * mult)
            if ds in attention_resolutions:
                layers.append(
                    BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                            swin_attn_type=swin_attn_type,
                            time_embed_dim=time_embed_dim,
                            **kwargs,
                                )
                )
            self.output_blocks.append(TimestepEmbedSequential(*layers))
            
            ich = input_block_chans.pop()
            layers = [
                conv_nd(2, ch + ich, int(model_channels * mult), 3, stride=1, padding=1)
            ]
            ch = int(model_channels * mult)
            if ds in attention_resolutions:
                layers.append(
                    BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=1,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.,
                            drop_path=0.,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                            swin_attn_type=swin_attn_type,
                            time_embed_dim=time_embed_dim,
                            **kwargs,
                                )
                )

            if level > 0 :
                out_ch = int(channel_mult[level - 1] * model_channels)

                layers.append(Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                ds *= 2
                ch = out_ch
            self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, lq=None, mask=None):

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)

        if lq is not None:
            assert self.cond_lq
            if mask is not None:
                assert self.cond_mask
                lq = th.cat([lq, mask], dim=1)
            lq = self.feature_extractor(lq.type(self.dtype))
            x = th.cat([x, lq], dim=1)

        h = x.type(self.dtype)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        return out

