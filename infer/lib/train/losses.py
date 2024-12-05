import torch
import torch.nn.functional as F


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l

def normalize_embeddings(embeddings):
    """
    Normalizes embeddings to unit vectors.

    Args:
        embeddings (torch.Tensor): Embeddings to normalize. Shape: [batch_size, embedding_dim]

    Returns:
        torch.Tensor: Normalized embeddings.
    """
    return F.normalize(embeddings, p=2, dim=1)

def speaker_embedding_loss(generated_embeddings, real_embeddings, margin=0.0):
    """
    Computes the Cosine Embedding Loss between generated and real speaker embeddings.

    Args:
        generated_embeddings (torch.Tensor): Embeddings from generated audio. Shape: [batch_size, embedding_dim]
        real_embeddings (torch.Tensor): Embeddings from real audio. Shape: [batch_size, embedding_dim]
        margin (float): Margin for cosine embedding loss.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Infer device from the generated embeddings
    device = generated_embeddings.device

    # Create target tensor on the same device
    target = torch.ones(generated_embeddings.size(0)).to(device)  # Labels: 1 for similar pairs

    # Compute Cosine Embedding Loss
    loss = F.cosine_embedding_loss(generated_embeddings, real_embeddings, target, margin=margin)
    return loss

# Phoneme Consistency Loss
def phoneme_consistency_loss(real_phoneme_features, generated_phoneme_features):
    """
    Computes the L1 loss between phoneme features of real and generated audio.

    Args:
        real_phoneme_features (torch.Tensor): Phoneme-level acoustic features of real audio.
        generated_phoneme_features (torch.Tensor): Phoneme-level acoustic features of generated audio.

    Returns:
        torch.Tensor: Phoneme consistency loss.
    """
    loss = F.l1_loss(generated_phoneme_features, real_phoneme_features)
    return loss