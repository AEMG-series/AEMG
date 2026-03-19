import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling_nst import NSTBackbone, get_aemg_base_config, get_aemg_large_config

def trunc_normal_(tensor, std=0.02):
    if hasattr(torch.nn.init, 'trunc_normal_'):
        torch.nn.init.trunc_normal_(tensor, std=std)
    else:
        torch.nn.init.normal_(tensor, mean=0, std=std)


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def sg(x):
    return x.detach()


class VQAEMG(nn.Module):
    def __init__(self, encoder_config, decoder_config, n_embed=8192, codebook_embed_dim=256,
                 decoder_out_dim=800, **kwargs):
        super().__init__()
        self.encoder = NSTBackbone(**encoder_config)
        self.decoder = NSTBackbone(**decoder_config)
        self.n_embed = n_embed
        self.codebook_embed_dim = codebook_embed_dim
        self.decoder_out_dim = decoder_out_dim
        self.embedding = nn.Embedding(n_embed, codebook_embed_dim)
        nn.init.uniform_(self.embedding.weight, -1/n_embed, 1/n_embed)
        self.encode_proj = nn.Sequential(
            nn.Linear(encoder_config["embed_dim"], encoder_config["embed_dim"]),
            nn.Tanh(),
            nn.Linear(encoder_config["embed_dim"], codebook_embed_dim)
        )
        self.decode_proj = nn.Sequential(
            nn.Linear(decoder_config["embed_dim"], decoder_config["embed_dim"]),
            nn.Tanh(),
            nn.Linear(decoder_config["embed_dim"], decoder_out_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in [self.encode_proj, self.decode_proj]:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x, in_chan_matrix=None, in_time_matrix=None):
        h = self.encoder(x, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix,
                         return_qrs_tokens=True)
        z = self.encode_proj(h)
        z_norm = l2norm(z)
        d = z_norm.pow(2).sum(dim=-1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * torch.einsum('bnd,kd->bnk', z_norm, self.embedding.weight)
        indices = torch.argmin(d, dim=-1)
        z_q = self.embedding(indices)
        return z_q, indices, z_norm

    def decode(self, z_q, in_chan_matrix=None, in_time_matrix=None):
        h = self.decoder(z_q, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix,
                         return_qrs_tokens=True)
        return self.decode_proj(h)

    def get_codebook_indices(self, x, in_chan_matrix=None, in_time_matrix=None):
        _, indices, _ = self.encode(x, in_chan_matrix, in_time_matrix)
        return indices

    def forward(self, x, in_chan_matrix=None, in_time_matrix=None):
        z_q, indices, z = self.encode(x, in_chan_matrix, in_time_matrix)
        x_rec = self.decode(z_q, in_chan_matrix, in_time_matrix)
        L_rec = F.mse_loss(x_rec, x)
        v_selected = self.embedding(indices)
        L_vocab = F.mse_loss(sg(z), l2norm(v_selected))
        L_commit = F.mse_loss(z, sg(l2norm(v_selected)))
        loss = L_rec + L_vocab + 0.25 * L_commit
        log = {
            "train/rec_loss": L_rec.detach(),
            "train/vocab_loss": L_vocab.detach(),
            "train/commit_loss": L_commit.detach(),
            "train/total_loss": loss.detach()
        }
        return loss, log


def vqaemg_base(n_embed=8192, code_dim=256):
    enc_cfg = get_aemg_base_config()
    dec_cfg = get_aemg_base_config().copy()
    dec_cfg["depth"] = 2
    dec_cfg["token_dim"] = code_dim
    dec_cfg["embed_dim"] = code_dim
    return VQAEMG(enc_cfg, dec_cfg, n_embed=n_embed, codebook_embed_dim=code_dim)


def vqaemg_large(n_embed=8192, code_dim=256):
    enc_cfg = get_aemg_large_config()
    dec_cfg = get_aemg_large_config().copy()
    dec_cfg["depth"] = 2
    dec_cfg["token_dim"] = code_dim
    dec_cfg["embed_dim"] = code_dim
    return VQAEMG(enc_cfg, dec_cfg, n_embed=n_embed, codebook_embed_dim=code_dim)
