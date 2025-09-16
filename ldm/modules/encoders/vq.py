import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta, metric='l2'):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.metric = metric

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.metric == 'cosine':
            # normalize z_flat and codebook
            z_n = F.normalize(z_flat, p=2, dim=1)  # (N,D)
            e_n = F.normalize(self.embedding.weight, p=2, dim=1)  # (K,D)
            # cosine sim -> to a "distance" by negation
            # we need a matrix same shape as d: (N, K)
            # here use d = - sim so argmin(d) == argmax(sim)
            sim = torch.matmul(z_n, e_n.t())  # (N,K)
            d = -sim
        elif self.metric == 'l2':
            # (z - e)^2 = z^2 + e^2 - 2 z·e
            d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
                torch.matmul(z_flat, self.embedding.weight.t())
        else:
            raise NotImplementedError

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        #.........\end

        # with:
        # .........\start
        #min_encoding_indices = torch.argmin(d, dim=1)
        #z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class SoftVectorQuantizer(nn.Module):
# see https://github.com/Hhhhhhao/continuous_tokenizer/blob/main/modelling/quantizers/softvq.py
    def __init__(
            self,
            n_e,
            e_dim,
            entropy_loss_ratio=0.01,
            tau=0.07,
            num_codebooks=1,
            l2_norm=False,
            show_usage=False,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.n_e = n_e
        self.e_dim = e_dim
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.tau = tau

        # Single embedding layer for all codebooks
        self.embedding = nn.Parameter(torch.randn(num_codebooks, n_e, e_dim))
        self.embedding.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        if self.l2_norm:
            self.embedding.data = F.normalize(self.embedding.data, p=2, dim=-1)

        if self.show_usage:
            self.register_buffer("codebook_used", torch.zeros(num_codebooks, 65536))

    def forward(self, z):
        # Handle different input shapes
        dim_z = z.dim()
        if dim_z == 4:
            b, c, h, w = z.shape
            z = torch.einsum('b c h w -> b h w c', z).contiguous()
            z = z.view(z.size(0), -1, z.size(-1))

        batch_size, seq_length, _ = z.shape

        # Ensure sequence length is divisible by number of codebooks
        assert seq_length % self.num_codebooks == 0, \
            f"Sequence length ({seq_length}) must be divisible by number of codebooks ({self.num_codebooks})"

        segment_length = seq_length // self.num_codebooks
        z_segments = z.view(batch_size, self.num_codebooks, segment_length, self.e_dim)

        # Apply L2 norm if needed
        embedding = F.normalize(self.embedding, p=2, dim=-1) if self.l2_norm else self.embedding
        if self.l2_norm:
            z_segments = F.normalize(z_segments, p=2, dim=-1)

        z_flat = z_segments.permute(1, 0, 2, 3).contiguous().view(self.num_codebooks, -1, self.e_dim)

        logits = torch.einsum('nbe, nke -> nbk', z_flat, embedding.detach())

        # Calculate probabilities
        probs = F.softmax(logits / self.tau, dim=-1)

        # Quantize
        z_q = torch.einsum('nbk, nke -> nbe', probs, embedding)

        # Reshape back
        z_q = z_q.view(self.num_codebooks, batch_size, segment_length, self.e_dim).permute(1, 0, 2, 3).contiguous()

        # Calculate cosine similarity
        with torch.no_grad():
            zq_z_cos = F.cosine_similarity(
                z_segments.view(-1, self.e_dim),
                z_q.view(-1, self.e_dim),
                dim=-1
            ).mean()

        # Get indices for usage tracking
        indices = torch.argmax(probs, dim=-1)  # (batch*segment_length, num_codebooks)

        # Track codebook usage
        if self.show_usage:
            for k in range(self.num_codebooks):
                cur_len = indices.size(0)
                self.codebook_used[k, :-cur_len].copy_(self.codebook_used[k, cur_len:].clone())
                self.codebook_used[k, -cur_len:].copy_(indices[:, k])

        vq_loss = commit_loss = 0.0
        entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(logits.view(-1, self.n_e))

        # Calculate codebook usage
        codebook_usage = torch.tensor([
            len(torch.unique(self.codebook_used[k])) / self.n_e
            for k in range(self.num_codebooks)
        ]).mean() if self.show_usage else 0

        z_q = z_q.view(batch_size, -1, self.e_dim)

        # Reshape back to match original input shape
        if dim_z == 4:
            z_q = rearrange(z_q, 'b (h w) c -> b h w c', h=h, w=w).permute(0,3,1,2)

        # Calculate average probabilities
        avg_probs = torch.mean(torch.mean(probs, dim=-1))
        max_probs = torch.mean(torch.max(probs, dim=-1)[0])

        return z_q, entropy_loss, (
            None,  # perplexity
            None,  # min_encodings
            indices.view(batch_size, self.num_codebooks, segment_length),
            avg_probs,
            max_probs,
            z_q.detach(),
            z.detach(),
            codebook_usage,
            zq_z_cos
        )


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss


def residual_quantize(z, quantizer: VectorQuantizer, L: int):
    loss = 0.0
    infos = []
    z_l = torch.zeros_like(z)
    r = z
    for l in range(1, L + 1):
        r_post_quant, emb_loss, info = quantizer(r)
        z_l = r_post_quant + z_l
        r = r - z_l
        loss += emb_loss
        infos.append(info)
    return z_l, loss, infos
