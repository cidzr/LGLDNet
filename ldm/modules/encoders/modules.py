import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import partial
import clip
from einops import rearrange, repeat
import kornia

from ldm.modules.diffusionmodules.model import Encoder as AEEncoder
from ldm.modules.diffusionmodules.model import Decoder as AEDeoder
from ldm.modules.diffusionmodules.model import make_attn, Upsample, Downsample, ResnetBlock, nonlinearity, Normalize
from ldm.modules.encoders.vq import SoftVectorQuantizer, VectorQuantizer
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class LabelEncoder(nn.Module):
    def __init__(self, *, ch_e, out_ch, num_res_blocks_e, in_channels, z_channels, z_size, resolution,
                 ch_mult=(1,2,4), double_z=False, use_quantize=False, **ignore_kwargs):
        super().__init__()
        self.use_quantize = use_quantize
        self.layers = nn.ModuleList()
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       ch_e,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       padding_mode='reflect')
        in_channels = ch_e
        for i, ch_i in enumerate(ch_mult):
            out_channels = ch_i * ch_e
            for i_block in range(num_res_blocks_e[i]):
                self.layers.append(ResnetBlock(in_channels=in_channels if i_block == 0 else out_channels,
                                               out_channels=out_channels,
                                               dropout=0.,
                                               padding_mode='reflect'))
            in_channels = out_channels
            if i != len(ch_mult) - 1:
                self.layers.append(Downsample(in_channels, with_conv=True))

        self.mid = nn.Sequential(ResnetBlock(in_channels=in_channels,
                                             out_channels=in_channels,
                                             dropout=0.,
                                             padding_mode='reflect'),
                                 make_attn(in_channels, attn_type='vanilla', return_qkv=False, padding_mode='reflect'),
                                 ResnetBlock(in_channels=in_channels,
                                             out_channels=in_channels,
                                             dropout=0.,
                                             padding_mode='reflect'))

        self.norm_out = Normalize(in_channels)
        self.final = nn.Conv2d(in_channels, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1,
                               padding=1, padding_mode='reflect', bias=False)

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.mid(x)
        x = self.norm_out(x)
        x = nonlinearity(x)
        x = self.final(x)
        if not self.use_quantize:
            x = F.layer_norm(x, normalized_shape=x.shape[1:])
        return x


class VAEInterface(AbstractEncoder):
    def __init__(self, ddconfig, stage, VQconfig=None, kl_weight=None, ckpt_path=None,
                 ignore_keys=[], quant_type='hard', *args, **kwargs):
        super().__init__(**kwargs)
        self.quant = VQconfig is not None
        self.kl_weight = kl_weight
        self.stage = stage
        if stage == "first_stage":
            self.encoder = LabelEncoder(use_quantize=self.quant or self.kl_weight, **ddconfig)
            self.decoder = AEDeoder(**ddconfig)
            if self.quant:
                assert kl_weight is None
                self.quant_conv = nn.Conv2d(ddconfig['z_channels'], VQconfig['e_dim'], kernel_size=1)
                self.post_quant_conv = nn.Conv2d(VQconfig['e_dim'], ddconfig['z_channels'], kernel_size=1)
                if quant_type == 'hard':
                    self.quantize = VectorQuantizer(**VQconfig)
                elif quant_type == 'soft':
                    self.quantize = SoftVectorQuantizer(**VQconfig)
                else:
                    raise NotImplementedError
            if self.kl_weight is not None:
                assert VQconfig is None
                self.quant_conv = nn.Conv2d(ddconfig['z_channels'] * 2, ddconfig['z_channels'] * 2, kernel_size=1)
                self.post_quant_conv = nn.Conv2d(ddconfig['z_channels'], ddconfig['z_channels'], kernel_size=1)
            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        elif stage == "cond_stage":
            scale_factor = 2 if ddconfig["double_z"] else 1
            self.cond_model = AEEncoder(**ddconfig)
            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        model_sd = self.state_dict()

        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"\033[31mDeleting key {k} from state_dict.\033[0m")
                    del sd[k]

        new_sd = {}
        for k, v in sd.items():
            if k not in model_sd:
                print(f"\033[31mUnexpected key {k} in checkpoint — skipping.\033[0m")
                continue

            model_v = model_sd[k]
            if v.shape == model_v.shape:
                new_sd[k] = v
            else:
                if len(v.shape) != len(model_v.shape):
                    print(f"\033[31mShape mismatch (dims) for {k}: {v.shape} vs {model_v.shape} — skipping.\033[0m")
                    continue

                slices = tuple(slice(0, min(v.shape[i], model_v.shape[i])) for i in range(len(v.shape)))
                print(f"\033[31mPartial loading {k}: ckpt {v.shape} → model {model_v.shape}\033[0m")
                new_param = model_v.clone()
                new_param[slices] = v[slices]
                new_sd[k] = new_param

        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        print(f"Loaded from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys.")
        if missing:
            print("Missing keys (partially or not loaded):", missing)
        if unexpected:
            print("Unexpected keys (ignored):", unexpected)

    def encode(self, x):
        h = self.encoder(x)
        if self.quant:
            h = self.quant_conv(h)
            return h
        elif self.kl_weight is not None:
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            return posterior
        else:
            return h

    def decode(self, x, res_z=None, hs=None):
        if self.quant or self.kl_weight:
            if self.quant:
                x, emb_loss, info = self.quantize(x)
            x = self.post_quant_conv(x)
        dec = self.decoder(x, res_z, hs)
        return dec

    # ImageEncoder
    def forward(self, x):
        z, hs, cond = self.cond_model(x)
        if self.quant:
            z = self.quant_conv(z)
        return hs, z, z, cond
