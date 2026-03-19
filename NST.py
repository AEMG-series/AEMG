import torch
import torch.nn as nn


def rearrange_qkv(t, heads):
    b, n, hd = t.shape
    d = hd // heads
    return t.view(b, n, heads, d).transpose(1, 2)


def repeat_cls(cls_token, b):
    return cls_token.unsqueeze(0).expand(b, -1)


def pack_cls_x(cls_tokens, x):
    return torch.cat([cls_tokens.unsqueeze(1), x], dim=1)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, heads=8, head_dim=64, dim_head=None, dropout=0.0):
        super().__init__()
        head_dim = dim_head if dim_head is not None else head_dim
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == embed_dim)
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.norm = nn.LayerNorm(embed_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embed_dim), nn.Dropout(dropout))
            if project_out else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange_qkv(t, self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(embed_dim=embed_dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(embed_dim, mlp_dim, dropout=dropout),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, token_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(token_dim, embed_dim)

    def forward(self, x):
        return self.proj(x)


class SpatialEmbedding(nn.Module):
    def __init__(self, num_embeddings=17, embed_dim=256):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, x, in_chan_matrix):
        spatial_embeddings = self.embed(in_chan_matrix.clamp(0, 16))
        return x + spatial_embeddings


class TemporalEmbedding(nn.Module):
    def __init__(self, num_embeddings=17, embed_dim=256):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, x, in_time_matrix):
        temporal_embeddings = self.embed(in_time_matrix.clamp(0, 16))
        return x + temporal_embeddings


class NSTBackbone(nn.Module):
    def __init__(self, seq_len=256, token_dim=800, embed_dim=256, depth=6, heads=8,
                 dim_head=64, mlp_dim=1024, dropout=0.1, emb_dropout=0.1,
                 code_dim=32, Encoder=True):
        super().__init__()
        self.token_embed = TokenEmbedding(token_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))
        self.spa_embed = SpatialEmbedding(17, embed_dim)
        self.tem_embed = TemporalEmbedding(17, embed_dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            embed_dim=embed_dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, dropout=dropout
        )
        self.norm_layer = nn.LayerNorm(embed_dim)

    def forward_feature(self, x, mask_bool_matrix=None, in_chan_matrix=None, in_time_matrix=None):
        x = self.token_embed(x)
        b, seq_len, embed_dim = x.shape
        if mask_bool_matrix is None:
            mask_bool_matrix = torch.zeros((b, seq_len), dtype=torch.bool, device=x.device)
        mask_tokens = self.mask_token.expand(b, seq_len, -1)
        w = mask_bool_matrix.unsqueeze(-1).type_as(mask_tokens)
        x = x * (1 - w) + mask_tokens * w
        if in_chan_matrix is not None:
            x = self.spa_embed(x, in_chan_matrix)
        if in_time_matrix is not None:
            x = self.tem_embed(x, in_time_matrix)
        cls_tokens = repeat_cls(self.cls_token, b)
        x = pack_cls_x(cls_tokens, x)
        x += self.pos_embed[:, :x.shape[1]]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm_layer(x)
        return x

    def forward(self, x, mask_bool_matrix=None, in_chan_matrix=None, in_time_matrix=None,
                return_qrs_tokens=False, return_all_tokens=False):
        zero_mask = (x == 0).all(dim=2)
        x = self.forward_feature(x, mask_bool_matrix, in_chan_matrix, in_time_matrix)
        if return_all_tokens:
            return x
        x = x[:, 1:, :]
        x[zero_mask] = 0
        if return_qrs_tokens:
            return x
        return x.mean(1)


def get_aemg_base_config():
    return dict(
        seq_len=256, token_dim=800, embed_dim=256, depth=6, heads=8,
        dim_head=64, mlp_dim=1024, dropout=0.1, emb_dropout=0.1
    )


def get_aemg_large_config():
    return dict(
        seq_len=256, token_dim=800, embed_dim=512, depth=12, heads=16,
        dim_head=64, mlp_dim=2048, dropout=0.1, emb_dropout=0.1
    )
