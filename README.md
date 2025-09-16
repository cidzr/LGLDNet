# LGLDNet
The official repository for "A LABEL-GUIDED LATENT DIFFUSION NETWORK FOR INFRARED SMALL TARGET DETECTION"
# Contributions
\item We propose IRSTD-LDM, the first Latent diffusion model applied to IRSTD, where the diffusion model is trained to predict the latent posterior of ground truth, making the decoder exploit richer deep latent features beyond mere skip connections.
\item We introduce KLDA, which aligns the predicted distribution to the latent posterior of ground truth, reducing the difficulty of learning the target distribution and providing the decoder with purer samples.
\item We design FAE that automatically decomposes features into high- and low-frequency bands and selectively enhances cues for small target while suppressing background clutter.
\item Extensive experiments on IRSTD-1k and NUDT-SIRST demonstrate that IRSTD-LDM outperforms state-of-the-art IRSTD methods.
