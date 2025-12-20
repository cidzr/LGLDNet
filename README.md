# Rebuttal
## Additional ablation study for "directly predicting the latent posterior" (anotated as "post-pred")
| No. | post-pred | KLDA | FAE | IRSTD-mIoU | IRSTD-Pd  | IRSTD-Fa  | IRSTD-F1  | NUDT-mIoU |  NUDT-Pd  | NUDT-Fa  |  NUDT-F1  |
|-----|:---------:|:----:|:---:|:----------:|:---------:|:---------:|:---------:|:---------:|:---------:|:--------:|:---------:|
| 1   |     ✘     |  ✘   |  ✘  |   62.55    |   90.41   |   16.09   |   76.96   |   91.98   |   98.94   |   6.30   |   95.85   |
| 2   |     ✘     |  ✘   |  ✔  |   63.10    |   90.48   | **11.60** |   77.38   |   93.89   |   98.94   |   3.54   |   96.85   |
| 3   |     ✔     |  ✘   |  ✘  |   63.79    |   90.48   |   13.21   |   77.90   |   93.94   | **99.15** |   3.10   |   96.87   |
| 4   |     ✔     |  ✔   |  ✘  |   66.18    |   92.25   |   12.53   |   79.65   |   92.25   | **99.15** |   5.45   |   95.97   |
| 5   |     ✔     |  ✘   |  ✔  |   65.21    |   90.48   |   28.62   |   78.94   |   94.64   |   98.94   | **2.78** |   97.24   |
| 6   |     ✔     |  ✔   |  ✔  | **67.17**  | **93.54** |   16.55   | **80.36** | **95.20** | **99.15** |   3.29   | **97.54** |

The supplemented ablation table demonstrates that posterior prediction, KLDA and FAE each contribute independently and complementarily. Posterior prediction alone improves mIoU, Fa and F1 over the baseline (No. 3 vs. 1), showing the intrinsic benefit of predicting the latent posterior. The full configuration (post-pred + KLDA + FAE, No. 6) yields the best overall performance, demonstrating that the three components are necessary and jointly optimal rather than redundant.

## Quantitative analysis of FLOPs, inference speed, and parameter quantity of diffusion-based IRSTD methods
| Method                                | DCFRNet(w/o diffusion model) |    DCFRNet(w diffusion model)    |          IRSTD-Diff           |                    LDLGNet(Ours)                     |
|----------------------------------------|:----------------------------:|:--------------------------------:|:-----------------------------:|:----------------------------------------------------:|
| FLOPs                                  |            178 G             |                -                 |               -               |                         22 G                         |
| FLOPs for 1 diffusion time step (Unet) |              -               |              172 G               |            67.87 G            |                        5.22 G                        |
| Parameters                             |           66.14 M            |          66.14 + 115 M           |            30.01 M            |                       48.21 M                        |
| Runtime (per image)                    |           44.52 ms           | 9681.83 ms (200 ddim time steps) | 9014.56 (200 ddim time steps) | 83.57 ms (single step)<br/>993.20 ms (20 ddim steps) |

Quantitatively, LDLGNet substantially reduces computational cost compared to pixel-space diffusion. Total FLOPs of our model are 22 G vs 178 G for DCFRNet. Per-step Unet FLOPs drop from 172 G to 5.22 G (≈33× reduction). Importantly, runtime is diffusion-step dependent: pixel-space methods require ≈9 s for 200 DDIM steps, whereas our model runs in 83.6 ms (single step) and ≈993 ms for 20 steps, showing practical, order-of-magnitude runtime benefits when using the latent diffusion. In addition, the difference in detection metrics between our method using 20 step DDIM sampling and single step sampling is very small, which allows our method to avoid time-consuming denoising processes.

# LGLDNet
The official repository for "A Label-Guided Latent Diffusion Network for Infrared Small Target Detection" (under review)
## Overall Framework
![outline](assets/framework.png)
## Contributions
- We propose LGLDNet, the first Latent Diffusion Model applied to IRSTD, where the diffusion model is trained to predict the latent posterior of ground truth, making the decoder exploit richer deep latent features beyond mere skip connections.
- We introduce KL distribution alignment (KLDA), which aligns the predicted distribution to the latent posterior of ground truth, reducing the difficulty of learning the target distribution and providing the decoder with purer samples.
- We design frequency-aware adaptive enhancement (FAE) that automatically decomposes features into high- and low-frequency bands and selectively enhances cues for small target while suppressing background clutter.
- Extensive experiments on IRSTD-1k and NUDT-SIRST demonstrate that IRSTD-LDM outperforms state-of-the-art IRSTD methods.
## Recommended environment
Create an environment by the following methods:
```bash
conda env create -f environment.yaml
```
## Usage — `main.py`

`main.py` accepts a configuration argument `--base` (path to a YAML config) and a boolean `--train` flag. Below are minimal example commands.

Use `python -m visdom.server` for visualization before start training.

### 1) Train Mask-VAE (pretrained Mask-VAE for [IRSTD-1k](https://drive.google.com/file/d/18alU2uTodp9Sgf-7fIuW6XQDrOqpafIE/view?usp=drive_link) and [NUDT-SIRST](https://drive.google.com/file/d/1v9PWkcjv7WsEaHhxIIJRehxUP6thae_X/view?usp=drive_link))

This runs training with the Mask-VAE configuration.

```bash
python main.py --base configs/autoencoder/autoencoder_kl_32x32x4.yaml --train True
```

### 2) Test Mask-VAE

Run with the same base config but set `--train False`.

```bash
python main.py --base configs/autoencoder/autoencoder_kl_32x32x4.yaml --train False
```

### 3) Train Diffusion model

This uses the diffusion configuration for full LGLDNet training.

```bash
python main.py --base configs/latent-diffusion/diffusion.yaml --train True
```

### 4) Test Diffusion model

```bash
python main.py --base configs/latent-diffusion/diffusion.yaml --train False
```
## Results and Trained Models
#### Qualitative Results

![outline](assets/Qualitative.png)

#### Quantative Results 
| Dataset   | mIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6)) |F1 (x10(-2))||
|-----------|:--------------:|:-----:|:-----:|:-----:|:-----:|
| IRSTD-1k  |     67.17      |  93.54 |16.55|80.36|[checkpoint](https://drive.google.com/file/d/1L_nJMiJrXdO9fGTC8idFE6RDXZuE5CKZ/view?usp=drive_link)|
| NUDT-SIRST |     95.20      |  99.15 | 3.29 |97.54|[checkpoint](https://drive.google.com/file/d/17cXBas_BB17l_v4RpJft-Gx8v6TH7SLz/view?usp=drive_link)|

This code is highly borrowed from [latent-diffusion](https://github.com/CompVis/latent-diffusion.git). Thanks to CompVis.
