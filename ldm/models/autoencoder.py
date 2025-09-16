import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from contextlib import contextmanager
from omegaconf import OmegaConf
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import visdom

from ldm.modules.encoders.vq import SoftVectorQuantizer, VectorQuantizer

from ldm.modules.diffusionmodules.model import Decoder
from ldm.modules.encoders.modules import LabelEncoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.losses.losses import loss_masks
from ldm.modules.ema import LitEma

from ldm.util import instantiate_from_config, prepare_latent_to_log
from ldm.metric import ROCMetric, SigmoidMetric, SamplewiseSigmoidMetric, PD_FA

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 VQconfig=None,
                 quant_type="hard",
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 input_key="gray_srg",
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 mode=None,
                 use_quantize=False,
                 l_bce_weight=1.0,
                 l_dice_weight=1.0,
                 l_q_weight=1.0,
                 sens_test=False,
                 ):
        super().__init__()
        self.quant_type = quant_type
        self.image_key = image_key
        self.input_key = input_key
        self.use_quantize = use_quantize
        # self.encoder = Encoder(**ddconfig)
        self.encoder = LabelEncoder(use_quantize=use_quantize, **ddconfig)
        self.decoder = Decoder(**ddconfig)
        if use_quantize:
            assert VQconfig is not None
            if quant_type == 'hard':
                self.quantize = VectorQuantizer(**VQconfig)
            elif quant_type == 'soft':
                self.quantize = SoftVectorQuantizer(**VQconfig)
            else:
                raise NotImplementedError
            self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], VQconfig["e_dim"], 1)
            self.post_quant_conv = torch.nn.Conv2d(VQconfig["e_dim"], ddconfig["z_channels"], 1)
        if monitor is not None:
            self.monitor = monitor
            if mode is not None:
                self.mode = mode
            self.batch_resize_range = batch_resize_range
            if self.batch_resize_range is not None:
                print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

            self.use_ema = use_ema
            if self.use_ema:
                self.model_ema = LitEma(self)
                print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            self.scheduler_config = scheduler_config
            self.l_bce_weight = l_bce_weight
            self.l_dice_weight = l_dice_weight
            self.l_q_weight = l_q_weight
            self.sens_test = sens_test

            self.iou = SigmoidMetric(score_thresh=0)
            self.niou = SamplewiseSigmoidMetric(1, score_thresh=0)
            self.ROC = ROCMetric(1, 10)
            self.PD_FA = PD_FA(1, 10)

            if self.sens_test:
                self.iou_noise = SigmoidMetric(score_thresh=0)
                self.PD_FA_noise = PD_FA(1, 10)

            self.vis_img = None
            self.vis_monitor = None


    def on_validation_start(self):
        if self.vis_img is None:
            self.vis_img = visdom.Visdom(env="Pred and Latent Visualization")
        if self.vis_monitor is None:
            self.vis_monitor = visdom.Visdom(env="Metrics and Monitors")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        # h, _, _ = self.encoder(x)
        h_prequant = self.encoder(x)
        if self.use_quantize:
            h = self.quant_conv(h_prequant)
            quant, emb_loss, info = self.quantize(h)
            return h_prequant, quant, emb_loss, info
        return h_prequant

    def encode_to_prequant(self, x):
        # out, _, _ = self.encoder(x)
        h = self.encoder(x)
        if self.use_quantize:
            hs = self.quant_conv(h)
        return h, hs

    def decode(self, quant):
        # quant = F.tanh(quant)
        if self.use_quantize:
            quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input, return_pred_indices=False, batch_idx=None, is_test=False, add_noise=0):
        if self.use_quantize:
            h_prequant, quant, diff, info = self.encode(input)
        else:
            quant = self.encode(input)
            h_prequant = quant
            diff = torch.tensor(0, dtype=torch.float, requires_grad=False).to(self.device)
        if add_noise > 0:
            noise = torch.randn(quant.shape, device=self.device) * add_noise
            quant = quant + noise
        # if self.training:
        #     quant = randn_inj(quant)
        dec = self.decode(quant)
        if batch_idx == 0 and not is_test:
            z = prepare_latent_to_log(quant.detach().cpu())
            z_prequant = prepare_latent_to_log(h_prequant.detach().cpu())
            z_max, z_min = z.max(), z.min()
            z = (z - z_min) / (z_max - z_min)
            self.vis_img.image(
                np.uint8(255 * z),
                opts=dict(title='z_quant',
                          caption=f'z.max={z_max}, z.min={z_min}',),
                win='z_quant'
            )
            self.vis_img.image(
                np.uint8(255 * z_prequant),
                opts=dict(title='z_prequant'),
                win='z_prequant'
            )
            self.vis_img.image(
                np.uint8(255 * torch.sigmoid(prepare_latent_to_log(dec.detach().cpu()))),
                opts=dict(title='reconstruction_sigma' ),
                win='reconstruction_sigma'
            )


        if return_pred_indices and self.use_quantize:
            return dec, diff, quant, info[2]
        return dec, diff, quant

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size + 16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx):
        # x = self.get_input(batch, self.image_key)
        x = self.get_input(batch, self.input_key)
        xrec, qloss, quant = self(x, batch_idx=batch_idx)
        label = self.get_input(batch, self.image_key)
        loss_bce, loss_dice = loss_masks(xrec, label, activated=False)
        log = {"train/loss_bce": loss_bce.detach().item(),
               "train/loss_dice": loss_dice.detach().mean(),
               "train/loss_codebook": qloss.detach().mean()}
        self.log_dict(log, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        loss = (loss_bce * self.l_bce_weight + loss_dice * self.l_dice_weight
                + qloss * self.l_q_weight)
        return loss

    def training_epoch_end(self, outputs=None):
        metrics = self.trainer.callback_metrics
        epoch = self.current_epoch
        self.log_to_visdom(metrics.get('train/loss_bce'), epoch,
                           win_name="win_loss_bce", title="Training BCE Loss", ylabel="loss_bce")
        self.log_to_visdom(metrics.get('train/loss_dice'), epoch,
                           win_name="win_loss_dice", title="Training Dice Loss", ylabel="loss_dice")
        self.log_to_visdom(metrics.get('train/loss_codebook'), epoch,
                           win_name="win_loss_codebook", title="Training Codebook Loss", ylabel="loss_codebook")
        self.log_to_visdom(metrics.get('train/loss_smooth'), epoch,
                           win_name="win_loss_smooth", title="Training Smooth Loss", ylabel="loss_smooth")

    def on_validation_epoch_start(self):
        self.iou.reset()
        self.niou.reset()
        self.ROC.reset()
        self.PD_FA.reset()

    def validation_step(self, batch, batch_idx):
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict_ema

    def _validation_step(self, batch, batch_idx, suffix=""):
        # x = self.get_input(batch, self.image_key)
        x = self.get_input(batch, self.input_key)
        xrec, qloss, _ = self(x, batch_idx=batch_idx)
        label = self.get_input(batch, self.image_key)
        loss_bce, loss_dice = loss_masks(xrec, label, activated=False)
        log = {f"val{suffix}/bce_loss": loss_bce.detach().mean(),
               f"val{suffix}/dice_loss": loss_dice.detach().mean(),
               f"val{suffix}/loss_codebook": qloss.detach().mean()}

        self.iou.update(xrec, label)
        self.niou.update(xrec, label)
        self.ROC.update(xrec, label)
        self.PD_FA.update(xrec, label)

        self.log_dict(log)
        return self.log_dict

    def validation_epoch_end(self, outputs=None):
        epoch = self.current_epoch

        _, iou = self.iou.get()
        _, niou = self.niou.get()
        _, _, _, _, F1_scores = self.ROC.get()
        FAs, PDs = self.PD_FA.get()
        fa = FAs[5]
        pd = PDs[5]
        f1 = F1_scores[5]
        self.log("val/IOU", iou, prog_bar=False, on_epoch=True)
        self.log("val/nIOU", niou, prog_bar=False, on_epoch=True)
        self.log("val/PD", pd, prog_bar=False, on_epoch=True)
        self.log("val/FA", fa, prog_bar=False, on_epoch=True)
        self.log("val/F1_score", f1, prog_bar=False, on_epoch=True)
        self.log_to_visdom(iou, epoch, win_name="win_iou", title="Validation iou", ylabel="iou")
        print("\033[31miou: {:.5g}, niou: {:.5g}, PD: {:.5g}, FA: {:.5g}, F1_score: {:.5g}\033[0m".format(
            iou, niou, pd, fa, f1))

    def on_test_epoch_start(self):
        self.iou.reset()
        self.niou.reset()
        self.ROC.reset()
        self.PD_FA.reset()

        if self.sens_test:
            self.iou_noise.reset()
            self.PD_FA_noise.reset()

    def test_step(self, batch, batch_idx):
        with self.ema_scope():
            # x = self.get_input(batch, self.image_key)
            x = self.get_input(batch, self.input_key)
            xrec, qloss, _ = self(x, batch_idx=batch_idx, is_test=True)
            label = self.get_input(batch, self.image_key)
            loss_bce, loss_dice = loss_masks(xrec, label, activated=False)
            log = {f"test_ema/bce_loss": loss_bce.detach().mean(),
                   f"test_ema/dice_loss": loss_dice.detach().mean(),
                   f"test_ema/loss_codebook": qloss.detach().mean()}

            self.iou.update(xrec, label)
            self.niou.update(xrec, label)
            self.ROC.update(xrec, label)
            self.PD_FA.update(xrec, label)

            if self.sens_test:
                x_noise_rec, _, _ = self(x, batch_idx=batch_idx, is_test=True, add_noise=0.5)
                self.iou_noise.update(x_noise_rec, label)
                self.PD_FA_noise.update(x_noise_rec, label)

            self.log_dict(log)
            return self.log_dict

    def test_epoch_end(self, outputs=None):
        _, iou = self.iou.get()
        _, niou = self.niou.get()
        _, _, _, _, F1_scores = self.ROC.get()
        FAs, PDs = self.PD_FA.get()
        fa = FAs[5]
        pd = PDs[5]
        f1 = F1_scores[5]
        self.log("val/IOU", iou, prog_bar=False, on_epoch=True)
        self.log("val/nIOU", niou, prog_bar=False, on_epoch=True)
        self.log("val/PD", pd, prog_bar=False, on_epoch=True)
        self.log("val/FA", fa, prog_bar=False, on_epoch=True)
        self.log("val/F1_score", f1, prog_bar=False, on_epoch=True)
        print("\033[31miou: {:.5g}, niou: {:.5g}, PD: {:.5g}, FA: {:.5g}, F1_score: {:.5g}\033[0m".format(
            iou, niou, pd, fa, f1))
        if self.sens_test:
            _, iou_noise = self.iou_noise.get()
            FAs_noise, _ = self.PD_FA_noise.get()
            fa_noise = FAs_noise[5]
            print("\033[32miou_noise: {:.5g}, FA_noise: {:.5g}\033[0m".format(iou_noise, fa_noise))

    def log_to_visdom(self, value, epoch, win_name, title, xlabel="Epoch", ylabel="Value"):
        if value is not None:
            value = value.item() if torch.is_tensor(value) else value
            win = getattr(self, win_name, None)
            win = self.vis_monitor.line(
                X=np.array([epoch]),
                Y=np.array([value]),
                win=win_name,
                update='append' if win is not None else None,
                opts=dict(title=title, xlabel=xlabel, ylabel=ylabel)
            )
            setattr(self, win_name, win)

    def configure_optimizers(self):
        lr = self.learning_rate
        print("lr", lr)
        opt_list = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if self.use_quantize:
            opt_list += (list(self.quantize.parameters()) +
                         list(self.quant_conv.parameters()) +
                         list(self.post_quant_conv.parameters()))
        opt_ae = torch.optim.AdamW(opt_list, lr=lr)

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }
            ]
            return [opt_ae], scheduler
        return [opt_ae], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=True, **kwargs):
        log = dict()
        x = self.get_input(batch, self.input_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _, quant = self(x, return_pred_indices=False)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["GT"] = self.get_input(batch, self.image_key)
        log["reconstructions"] = xrec
        log["latent"] = prepare_latent_to_log(quant)
        # if torch.sum(ind == 1).item() > torch.sum(ind == 0).item():
        #     ind = 1 - ind
        # ind = ind * 2 - 1
        # log["latent_ind"] = prepare_latent_to_log(ind.view(4, 1, self.z_size, self.z_size))
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation" or self.image_key == "gray_seg"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class Prior():
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        self.logvar = math.log(var)


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 input_key="gray_srg",
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 mode=None,
                 l_bce_weight=1.0,
                 l_dice_weight=1.0,
                 l_kl_weight=1.0,
                 sens_test=False,
                 env_name=None,
                 prior=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.input_key = input_key
        # self.encoder = Encoder(**ddconfig)
        self.encoder = LabelEncoder(use_quantize=True, **ddconfig)
        self.decoder = Decoder(**ddconfig)

        assert ddconfig["double_z"]
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scheduler_config = scheduler_config
        self.l_bce_weight = l_bce_weight
        self.l_dice_weight = l_dice_weight
        self.l_kl_weight = l_kl_weight

        self.iou = SigmoidMetric(score_thresh=0)
        self.niou = SamplewiseSigmoidMetric(1, score_thresh=0)
        self.ROC = ROCMetric(1, 10)
        self.PD_FA = PD_FA(1, 10)

        self.iou_sample = SigmoidMetric(score_thresh=0)
        self.niou_sample = SamplewiseSigmoidMetric(1, score_thresh=0)
        self.ROC_sample = ROCMetric(1, 10)
        self.PD_FA_sample = PD_FA(1, 10)

        self.vis_img = None
        self.vis_monitor = None
        self.env_name = env_name

        if monitor is not None:
            self.monitor = monitor
        if mode is not None:
            self.mode = mode
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if prior is not None:
            self.prior = Prior(prior['mean'], prior['var'])
        else:
            self.prior = None



    def on_validation_start(self):
        if self.vis_img is None:
            self.vis_img = visdom.Visdom(env=f"Pred and Latent Visualization{self.env_name}")
        if self.vis_monitor is None:
            self.vis_monitor = visdom.Visdom(env=f"Metrics and Monitors{self.env_name}")


    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")


    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)


    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior


    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec


    def forward(self, input, sample_posterior=True, batch_idx=None, is_test=False):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        if batch_idx == 0 and not is_test:
            z = prepare_latent_to_log(z.detach().cpu())
            z_max, z_min = z.max(), z.min()
            z = (z - z_min) / (z_max - z_min)
            self.vis_img.image(
                np.uint8(255 * z),
                opts=dict(title='z',
                          caption=f'z.max={z_max}, z.min={z_min}', ),
                win='z'
            )
            self.vis_img.image(
                np.uint8(255 * torch.sigmoid(prepare_latent_to_log(dec.detach().cpu()))),
                opts=dict(title='reconstruction_sigma'),
                win='reconstruction_sigma'
            )
        return dec, posterior


    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x


    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.input_key)
        reconstructions, posterior = self(inputs, batch_idx=batch_idx)

        loss_kl = posterior.kl(other=self.prior)
        loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]

        label = self.get_input(batch, self.image_key)
        loss_bce, loss_dice = loss_masks(reconstructions, label, activated=False)
        log = {"train/loss_bce": loss_bce.detach().item(),
               "train/loss_dice": loss_dice.detach().mean(),
               "train/loss_kl": loss_kl.detach().mean()}
        self.log_dict(log, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        loss = loss_bce * self.l_bce_weight + loss_dice * self.l_dice_weight + loss_kl * self.l_kl_weight
        return loss


    def training_epoch_end(self, outputs=None):
        metrics = self.trainer.callback_metrics
        epoch = self.current_epoch
        self.log_to_visdom(metrics.get('train/loss_bce'), epoch,
                           win_name="win_loss_bce", title="Training BCE Loss", ylabel="loss_bce")
        self.log_to_visdom(metrics.get('train/loss_dice'), epoch,
                           win_name="win_loss_dice", title="Training Dice Loss", ylabel="loss_dice")
        self.log_to_visdom(metrics.get('train/loss_kl'), epoch,
                           win_name="win_loss_kl", title="Training KL Loss", ylabel="loss_kl")


    def on_validation_epoch_start(self):
        self.iou.reset()
        self.niou.reset()
        self.ROC.reset()
        self.PD_FA.reset()


    def validation_step(self, batch, batch_idx):
        with self.ema_scope():
            inputs = self.get_input(batch, self.image_key)
            reconstructions, posterior = self(inputs, batch_idx=batch_idx)

            loss_kl = posterior.kl(other=self.prior)
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]

            label = self.get_input(batch, self.image_key)
            loss_bce, loss_dice = loss_masks(reconstructions, label, activated=False)
            log = {"val_ema/loss_bce": loss_bce.detach().item(),
                   "val_ema/loss_dice": loss_dice.detach().mean(),
                   "val_ema/loss_kl": loss_kl.detach().mean()}

            self.iou.update(reconstructions, label)
            self.niou.update(reconstructions, label)
            self.ROC.update(reconstructions, label)
            self.PD_FA.update(reconstructions, label)

            self.log_dict(log)
        return self.log_dict


    def validation_epoch_end(self, outputs=None):
        epoch = self.current_epoch

        _, iou = self.iou.get()
        _, niou = self.niou.get()
        _, _, _, _, F1_scores = self.ROC.get()
        FAs, PDs = self.PD_FA.get()
        fa = FAs[5]
        pd = PDs[5]
        f1 = F1_scores[5]
        self.log("val/IOU", iou, prog_bar=False, on_epoch=True)
        self.log("val/nIOU", niou, prog_bar=False, on_epoch=True)
        self.log("val/PD", pd, prog_bar=False, on_epoch=True)
        self.log("val/FA", fa, prog_bar=False, on_epoch=True)
        self.log("val/F1_score", f1, prog_bar=False, on_epoch=True)
        self.log_to_visdom(iou, epoch, win_name="win_iou", title="Validation iou", ylabel="iou")
        print("\033[31miou: {:.5g}, niou: {:.5g}, PD: {:.5g}, FA: {:.5g}, F1_score: {:.5g}\033[0m".format(
            iou, niou, pd, fa, f1))


    def on_test_epoch_start(self):
        self.iou.reset()
        self.niou.reset()
        self.ROC.reset()
        self.PD_FA.reset()

        self.iou_sample.reset()
        self.niou_sample.reset()
        self.ROC_sample.reset()
        self.PD_FA_sample.reset()


    def test_step(self, batch, batch_idx):
        with self.ema_scope():
            inputs = self.get_input(batch, self.image_key)
            reconstructions, posterior = self(inputs, sample_posterior=False, is_test=True)
            reconstructions_sample, _ = self(inputs, is_test=True)

            loss_kl = posterior.kl()
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]

            label = self.get_input(batch, self.image_key)
            loss_bce, loss_dice = loss_masks(reconstructions, label, activated=False)
            log = {"test_ema/loss_bce": loss_bce.detach().item(),
                   "test_ema/loss_dice": loss_dice.detach().mean(),
                   "test_ema/loss_kl": loss_kl.detach().mean()}

            self.iou.update(reconstructions, label)
            self.niou.update(reconstructions, label)
            self.ROC.update(reconstructions, label)
            self.PD_FA.update(reconstructions, label)

            self.iou_sample.update(reconstructions_sample, label)
            self.niou_sample.update(reconstructions_sample, label)
            self.ROC_sample.update(reconstructions_sample, label)
            self.PD_FA_sample.update(reconstructions_sample, label)

            self.log_dict(log)
            return self.log_dict


    def test_epoch_end(self, outputs=None):
        _, iou = self.iou.get()
        _, niou = self.niou.get()
        _, _, _, _, F1_scores = self.ROC.get()
        FAs, PDs = self.PD_FA.get()
        fa = FAs[5]
        pd = PDs[5]
        f1 = F1_scores[5]
        self.log("val/IOU", iou, prog_bar=False, on_epoch=True)
        self.log("val/nIOU", niou, prog_bar=False, on_epoch=True)
        self.log("val/PD", pd, prog_bar=False, on_epoch=True)
        self.log("val/FA", fa, prog_bar=False, on_epoch=True)
        self.log("val/F1_score", f1, prog_bar=False, on_epoch=True)
        print("\033[31miou: {:.5g}, niou: {:.5g}, PD: {:.5g}, FA: {:.5g}, F1_score: {:.5g}\033[0m".format(
            iou, niou, pd, fa, f1))


        _, iou_sample = self.iou_sample.get()
        _, niou_sample = self.niou_sample.get()
        _, _, _, _, F1_scores_sample = self.ROC_sample.get()
        FAs_sample, PDs_sample = self.PD_FA_sample.get()
        fa_sample = FAs_sample[5]
        pd_sample = PDs_sample[5]
        f1_sample = F1_scores_sample[5]
        print("\033[31miou_sample: {:.5g}, niou_sample: {:.5g}, PD_sample: {:.5g},"
              "FA_sample: {:.5g}, F1_score_sample: {:.5g}\033[0m".format(
            iou_sample, niou_sample, pd_sample, fa_sample, f1_sample))


    def log_to_visdom(self, value, epoch, win_name, title, xlabel="Epoch", ylabel="Value"):
        if value is not None:
            value = value.item() if torch.is_tensor(value) else value
            win = getattr(self, win_name, None)
            win = self.vis_monitor.line(
                X=np.array([epoch]),
                Y=np.array([value]),
                win=win_name,
                update='append' if win is not None else None,
                opts=dict(title=title, xlabel=xlabel, ylabel=ylabel)
            )
            setattr(self, win_name, win)

    def configure_optimizers(self):
        lr = self.learning_rate
        print("lr", lr)
        opt_list = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt_list += (list(self.quant_conv.parameters()) +
                     list(self.post_quant_conv.parameters()))
        opt_ae = torch.optim.AdamW(opt_list, lr=lr)

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }
            ]
            return [opt_ae], scheduler
        return [opt_ae], []


    def get_last_layer(self):
        return self.decoder.conv_out.weight


    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.input_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, posterior = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        if self.input_key != self.image_key:
            log["GT"] = self.get_input(batch, self.image_key)
        log["reconstructions"] = xrec
        log["latent_sample"] = prepare_latent_to_log(posterior.sample())
        log["latent_mode"] = prepare_latent_to_log(posterior.mode())
        log["samples"] = self.decode(torch.rand_like(posterior.sample()))
        # if torch.sum(ind == 1).item() > torch.sum(ind == 0).item():
        #     ind = 1 - ind
        # ind = ind * 2 - 1
        # log["latent_ind"] = prepare_latent_to_log(ind.view(4, 1, self.z_size, self.z_size))
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log


    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

