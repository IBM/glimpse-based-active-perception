from torch import nn
import torch
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots.nan_to_num(-torch.inf)).nan_to_num(0.)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v.nan_to_num())
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class RelationalCrossAttention(nn.Module):
    def __init__(
            self, 
            dim, heads=8, dim_head=64, dropout=0.0, layer_idx=0, dim_symbols=None,
            symbols_context_norm_t=None,
        ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        if dim_symbols is None:
            self.norm_symbols = self.norm
            dim_symbols = dim
        else:
            self.norm_symbols = nn.LayerNorm(dim_symbols)
            if symbols_context_norm_t is not None:
                self.norm_symbols = nn.Sequential([
                    lambda x: x.permute(0, 2, 1),
                    nn.LayerNorm(symbols_context_norm_t),
                    lambda x: x.permute(0, 2, 1),
                ])

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        torch.manual_seed(3 + layer_idx)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        torch.manual_seed(3 + layer_idx)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)

        self.to_v = nn.Linear(dim_symbols, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_symbols),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, g, x, mask=None):
        g = self.norm(g)
        x = self.norm_symbols(x)

        q = self.to_q(g)
        k = self.to_k(g)

        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots.nan_to_num(-torch.inf)).nan_to_num()
        attn = self.dropout(attn)

        out = torch.matmul(attn, v.nan_to_num())
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class AbstractorCore(nn.Module):
    def __init__(
            self,
            dim, depth, heads, dim_head, mlp_dim, dropout=0.0, glimpse_loc_dim=None,
            self_attn_skip_conn=True,
    ):
        super().__init__()
        
        self.self_attn_skip_conn = self_attn_skip_conn
        if glimpse_loc_dim is None:
            glimpse_loc_dim = dim
        self.layers = nn.ModuleList([])

        self.norm = nn.LayerNorm(dim if glimpse_loc_dim is None else glimpse_loc_dim)
        for d_idx in range(depth):
            modules = [
                PreNorm(glimpse_loc_dim, SelfAttention(glimpse_loc_dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                RelationalCrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, layer_idx=d_idx, dim_symbols=glimpse_loc_dim),
                PreNorm(glimpse_loc_dim, FeedForward(glimpse_loc_dim, mlp_dim, dropout=dropout))
            ]
            self.layers.append(nn.ModuleList(modules))

    def forward(self, g, x, mask=None):
        for layer in self.layers:
            self_attn, relational_attn, ff = layer

            x = relational_attn(g, x, mask=mask) + x
            x = ff(x) + x
            if self.self_attn_skip_conn:
                x = self_attn(x, mask=mask) + x
            else:
                x = self_attn(x, mask=mask) 
            x = ff(x) + x
        return x


class Abstractor(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, 
                 dim_head=64, dropout=0.0, glimpse_loc_dim=None,
                 agg_op='mean', self_attn_skip_conn=True,
                 num_classes=2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.abstractor_core = AbstractorCore(
            dim, depth, heads, dim_head, mlp_dim, dropout, glimpse_loc_dim=glimpse_loc_dim,
            self_attn_skip_conn=self_attn_skip_conn)

        if glimpse_loc_dim is None:
            glimpse_loc_dim = dim

        self.mlp_head = nn.Linear(glimpse_loc_dim, num_classes)
        self.agg_op = agg_op

    def forward(self, g, x, mask=None):
        g = self.dropout(g)
        x = self.dropout(x)

        out = self.abstractor_core(g, x, mask=mask)

        if self.agg_op == 'mean':
            out = out.mean(dim=1)  
        elif self.agg_op == 'max':
            out = out.max(dim=1).values
        else:
            raise NotImplementedError

        return self.mlp_head(out)


class AbstractorDownstreamArchitecture(nn.Module):
    def __init__(
            self, 
            glimpse_dim, depth, heads, mlp_dim, timesteps,
            context_norm_glimpses=False,
            context_norm_glimpse_locs=False,
            glimpse_loc_dim=None,
            glimpse_embedder=None,
            glimpse_locs_embedder=None,
            agg_op='mean',
            self_attn_skip_conn=True,
            dropout=0.,
            num_classes=2,
    ):
        """
        The most important parameters (the rest can be left to their default values)
        ----------
        glimpse_dim: dimensionality of each glimpse, if 'glimpse_embedder' is None, it should be the dimensionality of
            flattened raw glimpse (e.g., C * G * G), otherwise it correspond to the out dim of 'glimpse_embedder'
        timesteps: number of glimpses in the glimpsing process
        context_norm_glimpses: whether to use temporal context normalization for glimpses
        context_norm_glimpse_locs: whether to use temporal context normalization for glimpse locations
        glimpse_loc_dim: dimensionality of each glimpse location. If 'glimpse_locs_embedder' is None, it is just 2,
            otherwise it should correspond to the out dim of 'glimpse_locs_embedder'
        glimpse_embedder: an embedder for glimpses
        glimpse_locs_embedder: an embedder for glimpse locations
        num_classes: number of output classes
        """
        super(AbstractorDownstreamArchitecture, self).__init__()
        self.glimpse_dim = glimpse_dim
 
        if glimpse_embedder is not None:
            self.glimpse_embedder = glimpse_embedder
        else:
            self.glimpse_embedder = None

        if glimpse_locs_embedder is not None:
            self.glimpse_locs_embedder = glimpse_locs_embedder
        else:
            self.glimpse_locs_embedder = None
        
        self.context_norm_glimpses = nn.LayerNorm(normalized_shape=[timesteps]) if context_norm_glimpses else nn.Identity()
        self.context_norm_glimpse_locs = nn.LayerNorm(normalized_shape=[timesteps]) if context_norm_glimpse_locs else nn.Identity()

        self.abstractor = Abstractor(
            glimpse_dim, depth, heads, mlp_dim, glimpse_loc_dim=glimpse_loc_dim,
            agg_op=agg_op, self_attn_skip_conn=self_attn_skip_conn,
            dropout=dropout,
            num_classes=num_classes)

    def forward(self, glimpses=None, glimpse_locs=None, **kwargs):
        """
        :param glimpses: [B, T, S, C, H, W]
        :param glimpse_locs: [B, T, 2]
        :return:
        """
        if glimpses is not None:
            if self.glimpse_embedder is not None:
                glimpses = self.glimpse_embedder(glimpses)
            else:
                glimpses = glimpses.flatten(2)
            glimpses = self.context_norm_glimpses(glimpses.permute(0, 2, 1)).permute(0, 2, 1)

        if glimpse_locs is not None:
            if self.glimpse_locs_embedder is not None:
                glimpse_locs = self.glimpse_locs_embedder(glimpse_locs)
            glimpse_locs = self.context_norm_glimpse_locs(glimpse_locs.permute(0, 2, 1)).permute(0, 2, 1)

        score = self.abstractor(glimpses, glimpse_locs).squeeze()
        return score

