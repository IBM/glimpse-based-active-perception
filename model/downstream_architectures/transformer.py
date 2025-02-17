import torch
import torch.nn as nn

from utils import standardize_glimpse_locs
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., start_with_layer_norm=True):
        super().__init__()
        self.start_with_layer_norm = start_with_layer_norm
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        if self.start_with_layer_norm:
            x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., start_with_layer_norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head,
                          dropout = dropout, start_with_layer_norm=start_with_layer_norm),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class TransformerDownstreamArchitecture(nn.Module):
    def __init__(
            self,
            glimpse_size: int,
            glimpse_emd_dim: int,
            heads: int,
            head_dim: int,
            mlp_dim: int,
            readout_sizes: list,
            timesteps,
            glimpse_locs_emb_dim=None,
            depth=1,
            n_glimpse_scales=1,
            context_norm_glimpse_locs=False,
            context_norm_glimpses=False,
            use_pos_emb=False,
            use_cls_token=True,
            num_channels_in_glimpse=1,
            glimpse_embedder=None, 
            glimpse_locs_embedder=None, 
            dropout=0.,
    ):
        super().__init__()

        self.use_cls_token = use_cls_token
        self.use_pos_emb = use_pos_emb
        self.context_norm_glimpses = nn.LayerNorm(normalized_shape=[timesteps]) if context_norm_glimpses else nn.Identity()
        self.context_norm_glimpse_locs = nn.LayerNorm(normalized_shape=[timesteps]) if context_norm_glimpse_locs else nn.Identity()

        self.glimpse_locs_emb_dim = glimpse_locs_emb_dim
        self.num_glimpses = timesteps + int(self.use_cls_token)
        if glimpse_embedder is None:
            self.to_glimpse_embedding = nn.Sequential(
                nn.Flatten(start_dim=2),
                nn.LayerNorm([n_glimpse_scales * (num_channels_in_glimpse * glimpse_size ** 2)]),
                nn.Linear(n_glimpse_scales * (num_channels_in_glimpse * glimpse_size ** 2), glimpse_emd_dim),
                nn.LayerNorm([glimpse_emd_dim]),
            )
        else:
            self.to_glimpse_embedding = glimpse_embedder

        if self.use_cls_token:
            self.glimpse_cls_token = nn.Parameter(torch.randn(1, 1, glimpse_emd_dim))
            if glimpse_locs_emb_dim is not None:
                self.glimpse_loc_cls_token = nn.Parameter(torch.randn(1, 1, glimpse_locs_emb_dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_glimpses, glimpse_emd_dim))
        if glimpse_locs_emb_dim is not None:
            if glimpse_locs_embedder is None:
                self.glimpse_pos_embedder = nn.Linear(2, glimpse_locs_emb_dim)
            else:
                self.glimpse_pos_embedder = glimpse_locs_embedder
        else:
            self.glimpse_pos_embedder = None

        if self.glimpse_locs_emb_dim is not None:
            transfomer_inp_dim = glimpse_emd_dim + glimpse_locs_emb_dim
        else:
            transfomer_inp_dim = glimpse_emd_dim

        self.transformer = Transformer(
            transfomer_inp_dim,
            depth, heads, head_dim, mlp_dim, start_with_layer_norm=True,
            dropout=dropout)
        layers = []
        for i, size in enumerate(readout_sizes):
            if i == 0:
                layers.append(
                    nn.Linear(
                        transfomer_inp_dim * (1 if self.use_cls_token else self.num_glimpses),
                        size
                    )
                )
            else:
                layers.append(nn.Linear(readout_sizes[i - 1], size))
            if i < len(readout_sizes) - 1:
                layers.append(nn.SiLU())

        self.readout = nn.Sequential(*layers)
        self.n_glimpse_scales = n_glimpse_scales
        self.num_channels_in_glimpse = num_channels_in_glimpse
        if self.num_channels_in_glimpse > 3 and glimpse_embedder is None:
            assert self.n_glimpse_scales == 1

    def forward(self, glimpses, glimpse_locs=None, **kwargs):
        """
        :param glimpses: expected shape [B, T, S, C, H, W]
        :param glimpse_locs: expected shape [B, T, 2]
        :return:
        """

        if self.num_channels_in_glimpse <= 3:
            glimpses = glimpses[:, :, :self.n_glimpse_scales]

        if glimpse_locs is not None and self.context_norm_glimpse_locs:
            glimpse_locs = standardize_glimpse_locs(glimpse_locs)

        glimpses_embedding = self.to_glimpse_embedding(glimpses)
        glimpses_embedding = self.context_norm_glimpses(glimpses_embedding.permute(0, 2, 1)).permute(0, 2, 1)
        if self.use_cls_token:
            glimpses_embedding = torch.cat(
                [
                    glimpses_embedding, self.glimpse_cls_token.repeat(glimpses_embedding.shape[0], 1, 1)
                ], dim=1)
        if self.use_pos_emb:
            glimpses_embedding += self.pos_embedding

        if glimpse_locs is not None and self.glimpse_pos_embedder is not None:
            glimpse_locs_embedding = self.glimpse_pos_embedder(glimpse_locs)
            glimpse_locs_embedding = self.context_norm_glimpse_locs(glimpse_locs_embedding.permute(0, 2, 1)).permute(0, 2, 1)
            if self.use_cls_token:
                glimpse_locs_embedding = torch.cat(
                    [
                        glimpse_locs_embedding,
                        self.glimpse_loc_cls_token.repeat(glimpse_locs_embedding.shape[0], 1, 1)
                    ], dim=1)
            glimpses_embedding = torch.cat([glimpses_embedding, glimpse_locs_embedding], dim=-1)

        out = self.transformer(glimpses_embedding)
        if self.use_cls_token:
            out = out[:, -1]
        out = self.readout(out.flatten(1)).squeeze()
        return out


