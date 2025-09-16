# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from ldm.util import instantiate_from_config
from ldm.modules.attention import LinearAttention


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    while in_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, out_channels=None,
                 use_interpolate=True, scale_factor=2.0, mode='nearest'):
        super().__init__()
        self.with_conv = with_conv
        self.use_interpolate = use_interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        if out_channels is None:
            out_channels = in_channels
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        if not use_interpolate:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        if self.use_interpolate:
            x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        else:
            x = self.up(x)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, pooling='avg'):
        super().__init__()
        self.with_conv = with_conv
        self.pooling = pooling
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="reflect", value=0)
            x = self.conv(x)
        else:
            if self.pooling == 'avg':
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            elif self.pooling == 'max':
                x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            else:
                raise Exception(f'Unknown pooling {self.pooling}')
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512,
                 padding_mode='zeros', use_freq_attn=False, image_size=None, use_ch_attn=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.use_freq_attn = use_freq_attn
        self.use_ch_attn = use_ch_attn

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     padding_mode=padding_mode)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)

        if use_ch_attn:
            self.ch_attn = ChAttn(out_channels)

        if use_freq_attn:
            assert image_size is not None
            self.freq_attn = FreqGuidedAttn(out_channels, image_size)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     padding_mode=padding_mode)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     padding_mode=padding_mode)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x

        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.use_ch_attn:
            h = self.ch_attn(h)

        if self.use_freq_attn:
            h = self.freq_attn(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class FreqGuidedAttn(nn.Module):
    def __init__(self, in_channels, image_size, order=2, init_cutoff=0.1):
        super().__init__()
        self.h = self.w = image_size
        self.n = order
        self.max_freq = 0.5 * (2 ** 0.5)

        # learnable parameter α for cutoff
        init_val = torch.logit(torch.tensor(init_cutoff), eps=1e-6)
        self.alpha = nn.Parameter(init_val * torch.ones(in_channels))

        # build half-spectrum frequency distance matrix D(u,v)
        u = torch.fft.fftfreq(self.h)[:, None]             # (H,1)
        v = torch.fft.rfftfreq(self.w)[None, :]            # (1,W//2+1)
        D = torch.sqrt(u**2 + v**2)                        # (H,W//2+1)
        self.register_buffer('D_half', D)

        # spatial attention convs (for LP and HP branches)
        self.sa_conv_lp = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sa_conv_hp = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.gamma_hp = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.gamma_lp = nn.Parameter(torch.ones(1, in_channels, 1, 1))

    def forward(self, x):
        # x: (B,C,H,W)
        B, C, H, W = x.shape
        assert H == self.h and W == self.w

        # 1) cutoff
        D0 = torch.sigmoid(self.alpha) * self.max_freq         # in (0, max_freq)
        D = self.D_half.to(x)[None, None]  # (1,1,H,W/2+1)
        D = D.expand(1, C, -1, -1)  # (1,C,H,W/2+1)
        D0 = D0.view(1,C,1,1).expand(B, C, *D.shape[-2:])  # (B,C,H,W/2+1)

        # 2) build low-pass and high-pass half-spectrum filters
        H_lp = 1 / (1 + (D / D0) ** (2 * self.n))  # (B,C,H,W/2+1)
        H_hp = 1 - H_lp

        # 3) rfft2, filtering, irfft2
        X = torch.fft.rfft2(x, norm='ortho')  # (B,C,H,W/2+1)
        x_lp = torch.fft.irfft2(X * H_lp, s=(H, W), norm='ortho')
        x_hp = torch.fft.irfft2(X * H_hp, s=(H, W), norm='ortho')

        # 4) spatial attention fusion per branch
        # LP branch spatial attention
        lp_avg = x_lp.mean(dim=1, keepdim=True)
        lp_max, _ = x_lp.max(dim=1, keepdim=True)
        attn_lp = torch.sigmoid(self.sa_conv_lp(torch.cat([lp_avg, lp_max], dim=1)))
        x_lp_att = x_lp * attn_lp

        hp_avg = x_hp.mean(dim=1, keepdim=True)
        hp_max, _ = x_hp.max(dim=1, keepdim=True)
        attn_hp = torch.sigmoid(self.sa_conv_hp(torch.cat([hp_avg, hp_max], dim=1)))
        x_hp_att = x_hp * attn_hp

        return x + x_hp_att * self.gamma_hp + x_lp_att * self.gamma_lp

class ChAttn(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # [b, c, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.gap(x).view(b, c)  # [b, c]
        y = self.fc(y).view(b, c, 1, 1)  # [b, c, 1, 1]
        return x * y


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels, return_qkv=False, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.return_qkv = return_qkv

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 padding_mode=padding_mode)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 padding_mode=padding_mode)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 padding_mode=padding_mode)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        padding_mode=padding_mode)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        if self.return_qkv:
            return x+h_, [k.permute(0,2,1),v.permute(0,2,1)]
        else:
            return x+h_


def make_attn(in_channels, attn_type="vanilla", return_qkv=False, padding_mode='zeros'):
    assert attn_type in ["vanilla", "linear", "mutihead", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels, return_qkv, padding_mode='zeros')
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Encoder(nn.Module):
    def __init__(self, *, ch_e, out_ch, ch_mult=(1,2,4,8), num_res_blocks_e, attn_resolutions,
                 dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels, z_size=32,
                 double_z=False, use_linear_attn=False, attn_type="vanilla", ckpt_path=None,
                 skip_connection=True, use_freq_attn=False, use_ch_attn=False, **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch_e
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks_e = num_res_blocks_e
        self.resolution = resolution
        self.in_channels = in_channels
        self.skip_connection = skip_connection
        self.use_freq_attn = use_freq_attn
        self.use_ch_attn = use_ch_attn

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch_e * in_ch_mult[i_level]
            block_out = ch_e * ch_mult[i_level]
            for i_block in range(self.num_res_blocks_e[i_level]):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         use_freq_attn=use_freq_attn,
                                         image_size=curr_res,
                                         use_ch_attn=use_ch_attn))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type,
                                          return_qkv=False if i_level < self.num_resolutions else True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       use_freq_attn=use_freq_attn,
                                       image_size=curr_res,
                                       use_ch_attn=use_ch_attn)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, return_qkv=True)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       use_freq_attn=use_freq_attn,
                                       image_size=curr_res,
                                       use_ch_attn=use_ch_attn)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        h = self.conv_in(x)
        hs = []
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks_e[i_level]):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                hs.append(h)
                h = self.down[i_level].downsample(h)

        # middle
        hs.append(h)
        h = self.mid.block_1(hs[-1], temb)
        h, qkv = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.skip_connection:
            return h, hs, qkv
        else:
            return h, None, qkv


class Decoder(nn.Module):
    def __init__(self, *, ch_d, out_ch, ch_mult=(1,2,4,8), num_res_blocks_d, attn_resolutions,
                 dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels,
                 tanh_out=False, use_linear_attn=False, attn_type="vanilla", skip_connection=True,
                 use_freq_attn=False, use_ch_attn=False, use_res_z=False, **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch_d
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks_d = num_res_blocks_d
        self.resolution = resolution
        self.in_channels = in_channels
        self.tanh_out = tanh_out
        self.skip_connection = skip_connection
        self.use_freq_attn = use_freq_attn
        self.use_ch_attn = use_ch_attn
        self.use_res_z = use_res_z

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch_d*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))
        self.mult_scale_conv = nn.ModuleList()
        self.mult_scale_up = nn.ModuleList()

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels * 2 if use_res_z else z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       use_freq_attn=use_freq_attn,
                                       image_size=curr_res,
                                       use_ch_attn=use_ch_attn)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       use_freq_attn=use_freq_attn,
                                       image_size=curr_res,
                                       use_ch_attn=use_ch_attn)

        if skip_connection:
            skip_ch = [ch_d * i for i in ch_mult]
        else:
            skip_ch = [0 for _ in ch_mult]
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch_d*ch_mult[i_level]
            for i_block in range(self.num_res_blocks_d[i_level]):
                block.append(ResnetBlock(in_channels=block_in + skip_ch[i_level] if i_block == 0 else block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         use_freq_attn=use_freq_attn,
                                         image_size=curr_res,
                                         use_ch_attn=use_ch_attn))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                self.mult_scale_conv.append(nn.Conv2d(block_in, ch_d * ch_mult[0], 1, 1, 0))
                self.mult_scale_up.append(Upsample(ch_d * ch_mult[0], resamp_with_conv,
                                                   scale_factor=2 ** i_level, mode='bilinear'))
                up.upsample = Upsample(block_in, resamp_with_conv, out_channels=block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        # if skip_connection:
        #     self.conv_skip = nn.Conv2d(block_in + skip_ch[0], block_in, kernel_size=3, stride=1, padding=1)
        self.fuse = nn.Conv2d(ch_d * ch_mult[0] * len(ch_mult), block_in, 1, 1, 0)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, res_z=None, hs=None):
        #assert z.shape[1:] == self.z_shape[1:]
        if hs is not None and self.skip_connection:
            skip = list(hs)
        if self.use_res_z:
            assert res_z is not None
        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(torch.cat((z, res_z), 1) if self.use_res_z else z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        mult_scale = []

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks_d[i_level]):
                # if hs is not None and self.skip_connection and i_block==0:
                #     h = torch.cat((h, skip.pop()), dim=1)
                if hs is not None and self.skip_connection and i_block == 0:
                    skip_feat = skip.pop()  # [B, C_skip, Hs, Ws]
                    h = torch.cat((h, skip_feat), dim=1)
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                mult_scale.append(h)
                h = self.up[i_level].upsample(h)

        for i, (conv, up) in enumerate(zip(self.mult_scale_conv, self.mult_scale_up)):
            mult_scale[i] = up(conv(mult_scale[i]))
        mult_scale.append(h)
        h = self.fuse(torch.cat(mult_scale, dim=1))
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
