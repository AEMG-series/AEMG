import torch
import torch.nn as nn
from modeling_nst import NSTBackbone, get_aemg_base_config, get_aemg_large_config


def random_masking(x, mask_ratio=0.5):
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask.to(torch.bool)


class AEMGPretrain(nn.Module):
    def __init__(self, vocab_size=8192, config=None):
        super().__init__()
        config = config or get_aemg_base_config()
        self.backbone = NSTBackbone(**config)
        self.lm_head = nn.Linear(config["embed_dim"], vocab_size)

    def forward(self, x, mask_bool_matrix=None, in_chan_matrix=None, in_time_matrix=None,
                return_all_tokens=False):
        h = self.backbone(x, mask_bool_matrix, in_chan_matrix, in_time_matrix,
                          return_qrs_tokens=True, return_all_tokens=False)
        logits = self.lm_head(h)
        if return_all_tokens:
            return logits
        if mask_bool_matrix is not None and mask_bool_matrix.any():
            return logits[mask_bool_matrix]
        return logits


def aemg_pretrain_base(vocab_size=8192):
    return AEMGPretrain(vocab_size=vocab_size, config=get_aemg_base_config())


def aemg_pretrain_large(vocab_size=8192):
    return AEMGPretrain(vocab_size=vocab_size, config=get_aemg_large_config())
