import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Batch

def get_single_data(graph_list):
    return Batch.from_data_list(graph_list).to(device)

EPS = 1e-8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class MultiHeadAttention(torch.nn.Module):
#     def __init__(self, input_dim, n_heads, ouput_dim=None):
#
#         super(MultiHeadAttention, self).__init__()
#         self.d_k = self.d_v = input_dim // n_heads
#         self.n_heads = n_heads
#         if ouput_dim == None:
#             self.ouput_dim = input_dim
#         else:
#             self.ouput_dim = ouput_dim
#         self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
#         self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
#         self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
#         self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)
#
#     def forward(self, X):
#         ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
#         Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
#         K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
#         V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)
#
#         scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
#         # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
#         attn = torch.nn.Softmax(dim=-1)(scores)
#         context = torch.matmul(attn, V)
#         # context: [len_q, n_heads * d_v]
#         context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
#         output = self.fc(context)
#         return output
class AttentionPooling(torch.nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # 使用全连接层计算注意力权重
        attention_weights = self.softmax(x)
        # 使用注意力权重对输入进行加权平均
        pooled_output = torch.sum(x * attention_weights, dim=1)
        return pooled_output

# class EncoderLayer(torch.nn.Module):
#     def __init__(self, input_dim, n_heads):
#         super(EncoderLayer, self).__init__()
#         self.attn = MultiHeadAttention(input_dim, n_heads)
#         self.AN1 = torch.nn.LayerNorm(input_dim)
#
#         self.l1 = torch.nn.Linear(input_dim, input_dim)
#         self.AN2 = torch.nn.LayerNorm(input_dim)
#
#     def forward(self, X):
#         output = self.attn(X)
#         X = self.AN1(output + X)
#
#         output = self.l1(X)
#         X = self.AN2(output + X)
#
#         return X
def proj(x, c):
    norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=EPS)
    maxnorm = (1.0 - 1e-5) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

def expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = torch.clamp(torch.norm(u, dim=-1, keepdim=True), min=EPS)
    gamma = torch.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return proj(gamma, c)

def logmap0(y, c):
    sqrt_c = c ** 0.5
    y_norm = torch.clamp(torch.norm(y, dim=-1, keepdim=True), min=EPS)
    scaled = torch.clamp(sqrt_c * y_norm, max=1 - 1e-5)
    return y / y_norm * (1.0 / sqrt_c) * torch.atanh(scaled)

def mobius_add(x, y, c):
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c * c * x2 * y2
    return proj(num / torch.clamp(den, min=EPS), c)
def masked_mean(x, mask):
    mask = mask.unsqueeze(-1).to(x.dtype)   # [B, L, 1]
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

def masked_max(x, mask):
    x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
    out = x.max(dim=1).values
    out[out < -1e8] = 0.0
    return out

class PairCrossInteraction(nn.Module):
    """
    先做 drug1 <-> drug2 节点级 cross-attention，
    再把交互后的节点表示池化成 drug 表示，
    最后构造 128 维 pair_feat
    """
    def __init__(self, node_dim, drug_dim=64, nhead=4, dropout=0.1):
        super().__init__()
        self.attn_h2t = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.attn_t2h = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.h_update = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.t_update = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )

        # 节点级 -> drug级
        self.drug_proj = nn.Sequential(
            nn.Linear(node_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, drug_dim),
        )

        # 显式双线性交互
        self.bilinear = nn.Bilinear(drug_dim, drug_dim, drug_dim)

        # 保持最终输出 128 维，不动 head
        self.pair_proj = nn.Sequential(
            nn.Linear(drug_dim * 5, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )
        self.out_norm = nn.LayerNorm(128)

    def forward(self, h_nodes, h_mask, t_nodes, t_mask):
        # h -> t
        h2t = self.attn_h2t(
            h_nodes, t_nodes, t_nodes,
            key_padding_mask=~t_mask,
            need_weights=False
        )[0]

        # t -> h
        t2h = self.attn_t2h(
            t_nodes, h_nodes, h_nodes,
            key_padding_mask=~h_mask,
            need_weights=False
        )[0]

        h_ctx = h_nodes + self.h_update(torch.cat([h_nodes, h2t], dim=-1))
        t_ctx = t_nodes + self.t_update(torch.cat([t_nodes, t2h], dim=-1))

        h_pool = torch.cat([
            masked_mean(h_ctx, h_mask),
            masked_max(h_ctx, h_mask)
        ], dim=-1)
        t_pool = torch.cat([
            masked_mean(t_ctx, t_mask),
            masked_max(t_ctx, t_mask)
        ], dim=-1)

        h_feat = self.drug_proj(h_pool)   # [B, 64]
        t_feat = self.drug_proj(t_pool)   # [B, 64]

        bi_feat = self.bilinear(h_feat, t_feat)

        pair_raw = torch.cat([
            h_feat,
            t_feat,
            (h_feat - t_feat).abs(),
            h_feat * t_feat,
            bi_feat
        ], dim=-1)   # [B, 5*64]

        pair_feat = self.pair_proj(pair_raw)   # [B, 128]
        pair_feat = self.out_norm(pair_feat)
        return pair_feat
class HyperbolicGINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, init_c=1.0, dropout=0.1, train_eps=True):
        super().__init__()
        self.raw_c = nn.Parameter(torch.tensor(math.log(math.expm1(init_c))), requires_grad=True)
        self.train_eps = train_eps
        if train_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.zeros(1))

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.tan_ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        self.res_proj = None
        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)

    def curvature(self):
        return F.softplus(self.raw_c) + 1e-4

    def forward(self, x_h, edge_index):
        c = self.curvature()

        # hyperbolic -> tangent
        x_t = logmap0(x_h, c)  # [N, D]

        row, col = edge_index
        agg = torch.zeros_like(x_t)
        # GIN 风格：用 sum aggregation，不除度
        agg.index_add_(0, row, x_t[col])

        self_term = (1.0 + self.eps) * x_t
        out_t = self.mlp(self_term + agg)
        out_t = self.tan_ln(out_t)
        out_t = self.dropout(out_t)

        res_t = x_t if self.res_proj is None else self.res_proj(x_t)
        out_t = out_t + res_t

        # tangent -> hyperbolic
        out_h = expmap0(out_t, c)
        return proj(out_h, c)

class HyperbolicGIN_DDI(nn.Module):
    """
    One-layer HyperbolicGIN encoder.
    输入: PyG molecular graph, 包含 data.x, data.edge_index, data.batch
    输出: drug_feat, [N_drug, drug_dim]
    """
    def __init__(self, in_features=55, drug_dim=64, dropout=0.2, c=1.0):
        super().__init__()
        self.in_features = in_features
        self.drug_dim = drug_dim

        hidden = 128

        # atom feature projection: 55 -> 128
        self.input_proj = nn.Linear(in_features, hidden)
        self.hyp1 = HyperbolicGINLayer(
            hidden,
            hidden,
            init_c=c,
            dropout=dropout
        )

        self.post_ln = nn.LayerNorm(hidden)
        self.post_drop = nn.Dropout(dropout)

        self.drug_proj = nn.Sequential(
            nn.Linear(hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, drug_dim),
        )

    def encode_one_drug(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. 原子特征投影
        x = self.input_proj(x)

        # 2. 映射到 Poincaré ball
        c0 = self.hyp1.curvature()
        x_h = expmap0(x, c0)

        # 3. 一层 HyperbolicGIN
        h1 = self.hyp1(x_h, edge_index)

        # 4. 映射回切空间，作为节点表示
        t1 = logmap0(h1, c0)

        # 5. 后处理
        x_t = self.post_ln(t1)
        x_t = F.gelu(x_t)
        x_t = self.post_drop(x_t)

        return x_t, batch

    def forward(self, data):
        x_t, batch = self.encode_one_drug(data)

        # 节点级 -> 图级 / 单药级
        drug_pool = torch.cat([
            global_mean_pool(x_t, batch),
            global_max_pool(x_t, batch)
        ], dim=-1)

        drug_feat = self.drug_proj(drug_pool)  # [N_drug, drug_dim]
        return drug_feat

class GIN_DDI(nn.Module):
    """
    普通欧氏GIN，用于双曲GIN消融实验。

    与当前HyperbolicGIN_DDI保持一致：
    1. 相同的输入维度；
    2. 相同的隐藏维度；
    3. 相同的GIN层数；
    4. 相同的Mean+Max图级读出；
    5. 相同的药物输出维度。

    主要区别：
        GIN_DDI不使用双曲映射和曲率参数。
    """
    def __init__(
            self,
            in_features=55,
            drug_dim=64,
            dropout=0.2
    ):
        super().__init__()

        self.in_features = in_features
        self.drug_dim = drug_dim

        hidden = 128

        # 原子特征投影：55 -> 128
        self.input_proj = nn.Linear(
            in_features,
            hidden
        )

        # 普通欧氏GIN
        gin_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden)
        )

        self.gin1 = GINConv(
            nn=gin_mlp,
            train_eps=True
        )

        self.gin_ln = nn.LayerNorm(hidden)
        self.gin_drop = nn.Dropout(dropout)

        self.post_ln = nn.LayerNorm(hidden)
        self.post_drop = nn.Dropout(dropout)

        # 与当前HyperbolicGIN_DDI保持相同的Mean+Max读出
        self.drug_proj = nn.Sequential(
            nn.Linear(hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, drug_dim),
        )

    def encode_one_drug(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # 原子特征投影
        x = self.input_proj(x)

        # 保存残差
        residual = x

        # 普通欧氏GIN消息传递
        atom_feat = self.gin1(
            x,
            edge_index
        )

        atom_feat = self.gin_ln(atom_feat)
        atom_feat = self.gin_drop(atom_feat)

        # 与双曲GIN保持相同的残差结构
        atom_feat = atom_feat + residual

        atom_feat = self.post_ln(atom_feat)
        atom_feat = F.gelu(atom_feat)
        atom_feat = self.post_drop(atom_feat)

        return atom_feat, batch

    def forward(self, data):
        atom_feat, batch = self.encode_one_drug(data)

        # 与当前双曲GIN保持相同的Mean+Max读出
        drug_pool = torch.cat([
            global_mean_pool(atom_feat, batch),
            global_max_pool(atom_feat, batch)
        ], dim=-1)

        drug_feat = self.drug_proj(
            drug_pool
        )

        return drug_feat

def get_data(h_list, t_list):
    h_loader = Batch.from_data_list(h_list).to(device)
    t_loader = Batch.from_data_list(t_list).to(device)
    data = (h_loader,t_loader)
    return data