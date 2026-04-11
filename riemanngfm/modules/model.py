import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from riemanngfm.modules.layers import EuclideanEncoder, ManifoldEncoder
from riemanngfm.modules.basics import HyperbolicStructureLearner, SphericalStructureLearner, StarStructureLearner
from riemanngfm.manifolds import Lorentz, Sphere, ProductSpace
import inspect


class GeoGFM(nn.Module):
    def __init__(
        self,
        n_layers,
        in_dim,
        hidden_dim,
        embed_dim,
        bias,
        activation,
        dropout,
        curvature_mode="learnable",
        kappa=1.0,
        kappa_sign_h=-1,
        kappa_sign_s=1,
    ):
        super(GeoGFM, self).__init__()

        self.curvature_mode = curvature_mode
        self.kappa = float(kappa)
        self.kappa_sign_h = int(kappa_sign_h)
        self.kappa_sign_s = int(kappa_sign_s)

        learnable = self.curvature_mode == "learnable"
        k_abs = abs(self.kappa)

        # Lorentz/Sphere 都用正的 k（曲率幅值），由流形类型决定几何符号
        self.manifold_H = Lorentz(k=k_abs, learnable=learnable)
        self.manifold_S = Sphere(k=k_abs, learnable=learnable)

        self.product = ProductSpace(*[(self.manifold_H, embed_dim),
                                      (self.manifold_S, embed_dim)])
        self.init_block = InitBlock(self.manifold_H, self.manifold_S,
                                    in_dim, hidden_dim, embed_dim, bias,
                                    activation, dropout)
        self.blocks = nn.ModuleList([])
        for i in range(n_layers):
            self.blocks.append(StructuralBlock(self.manifold_H, self.manifold_S,
                                               embed_dim, hidden_dim, embed_dim, dropout))
        self.proj = nn.Sequential(nn.Linear(2 * embed_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, embed_dim))

    @staticmethod
    def _build_manifold(manifold_cls, sign, mode, kappa):
        learnable = (mode == "learnable")
        k_abs = abs(float(kappa))  # 关键：k只用正值
        # sign 暂不用于 k，避免负 k 导致 NaN
        return manifold_cls(k=k_abs, learnable=learnable)

    def forward(self, data):
        """

        :param data: Dataset for a graph contains batched sub-trees.
        :return: z: node product representations
        """
        x = data.x.clone()
        
        x_E, x_H, x_S = self.init_block(x, data.edge_index, data.tokens(data.n_id))  # [N, Hidden]
        for i, block in enumerate(self.blocks):
            x_E, x_H, x_S = block((x_E, x_H, x_S), data)
        return x_E, x_H, x_S

    def loss(self, x_tuple):
        """

        :param x_tuple: (x_E, x_H, x_S)
        :return:
        """

        x_E, x_H, x_S = x_tuple

        H_E = self.manifold_H.proju(x_H, x_E)
        S_E = self.manifold_S.proju(x_S, x_E)

        H_E = self.manifold_H.transp0back(x_H, H_E)
        S_E = self.manifold_S.transp0back(x_S, S_E)

        log0_H = self.manifold_H.logmap0(x_H)
        log0_S = self.manifold_S.logmap0(x_S)
        H_E = self.proj(torch.cat([log0_H, H_E], dim=-1))
        S_E = self.proj(torch.cat([log0_S, S_E], dim=-1))
        loss = self.cal_cl_loss(H_E, S_E)

        return loss

    def cal_cl_loss(self, x1, x2):
        EPS = 1e-6

        # 防 NaN/Inf 扩散
        x1 = torch.nan_to_num(x1, nan=0.0, posinf=1e4, neginf=-1e4)
        x2 = torch.nan_to_num(x2, nan=0.0, posinf=1e4, neginf=-1e4)

        norm1 = x1.norm(dim=-1).clamp_min(EPS)
        norm2 = x2.norm(dim=-1).clamp_min(EPS)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / (
            torch.einsum('i,j->ij', norm1, norm2) + EPS
        )
        sim_matrix = sim_matrix.clamp(-1.0, 1.0)          # cosine 范围保护
        sim_matrix = torch.exp((sim_matrix / 0.2).clamp(-20, 20))  # 防 exp 溢出

        pos_sim = sim_matrix.diag().clamp_min(EPS)
        denom1 = sim_matrix.sum(dim=-2).clamp_min(EPS)
        denom2 = sim_matrix.sum(dim=-1).clamp_min(EPS)

        loss_1 = -(pos_sim / denom1).clamp_min(EPS).log().mean()
        loss_2 = -(pos_sim / denom2).clamp_min(EPS).log().mean()
        return (loss_1 + loss_2) / 2.


class InitBlock(nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, bias, activation, dropout):
        super(InitBlock, self).__init__()
        self.Euc_init = EuclideanEncoder(in_dim, hidden_dim, out_dim, bias, activation, dropout)
        self.Hyp_init = ManifoldEncoder(manifold_H, in_dim, hidden_dim, out_dim, bias, None, 0.)
        self.Sph_init = ManifoldEncoder(manifold_S, in_dim, hidden_dim, out_dim, bias, None, 0.)

    def forward(self, x, edge_index, tokens):
        """

        :param tokens: input tokens
        :param x: raw features
        :param edge_index: edges
        :return: (E, H, S) Manifold initial representations
        """
        # E = self.Euc_init(tokens, edge_index)
        E = self.Euc_init(tokens, edge_index)  
        H = self.Hyp_init(tokens, edge_index)
        S = self.Sph_init(tokens, edge_index)
        return E, H, S


# class StructuralBlock(nn.Module):
#     def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout):
#         super(StructuralBlock, self).__init__()
#         self.manifold_H = manifold_H
#         self.manifold_S = manifold_S
#         self.Hyp_learner = HyperbolicStructureLearner(self.manifold_H, self.manifold_S, in_dim, hidden_dim, out_dim, dropout)
#         self.Sph_learner = SphericalStructureLearner(self.manifold_H, self.manifold_S, in_dim, hidden_dim, out_dim, dropout)
#         self.proj = self.proj = nn.Sequential(nn.Linear(3 * out_dim, hidden_dim),
#                                   nn.ReLU(),
#                                   nn.Linear(hidden_dim, out_dim))

#     def forward(self, x_tuple, data):
#         """

#         :param x_tuple: (x_E, x_H, x_S)
#         :param data: Dataset for a graph contains batched sub-graphs and sub-trees
#         :return: x_tuple: (x_H, x_S)
#         """
#         x_E, x_H, x_S = x_tuple
#         x_H = self.Hyp_learner(x_H, x_S, data.batch_tree)
#         x_S = self.Sph_learner(x_H, x_S, data)

#         H_E = self.manifold_H.proju(x_H, x_E)
#         S_E = self.manifold_S.proju(x_S, x_E)

#         H_E = self.manifold_H.transp0back(x_H, H_E)
#         S_E = self.manifold_S.transp0back(x_S, S_E)

#         E = torch.cat([x_E, H_E, S_E], dim=-1)
#         x_E = self.proj(E)
#         return x_E, x_H, x_S

class StructuralBlock(nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout):
        super(StructuralBlock, self).__init__()
        self.manifold_H = manifold_H
        self.manifold_S = manifold_S
        
        # 🆕 NEW: 同时初始化两种树状结构学习器
        # 默认的通用树学习器 (处理BFS树)
        self.Hyp_learner = HyperbolicStructureLearner(self.manifold_H, self.manifold_S, in_dim, hidden_dim, out_dim, dropout)
        # 针对性的星型图学习器
        self.Star_learner = StarStructureLearner(self.manifold_H, self.manifold_S, in_dim, hidden_dim, out_dim, dropout)
        
        # 球面学习器保持不变
        self.Sph_learner = SphericalStructureLearner(self.manifold_H, self.manifold_S, in_dim, hidden_dim, out_dim, dropout)
        
        self.proj = nn.Sequential(nn.Linear(3 * out_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, out_dim))

    def forward(self, x_tuple, data):
        """
        (模块化修正版：根据数据是否存在 batch_star 来动态选择学习器)
        """
        x_E, x_H, x_S = x_tuple

        # --- 🆕 NEW: 动态选择树状结构学习器 ---
        if hasattr(data, 'batch_star') and data.batch_star is not None:
            # 如果是 github 或 software 数据集，它们会有 batch_star
            # 我们优先使用针对性的 Star_learner
            # print("INFO: 检测到星型图结构，使用 StarStructureLearner。")
            x_H = self.Star_learner(x_H, x_S, data.batch_star)
        else:
            # 对于其他数据集，使用默认的通用 Hyp_learner
            x_H = self.Hyp_learner(x_H, x_S, data.batch_tree)
        
        # 球面学习器保持不变
        x_S = self.Sph_learner(x_H, x_S, data)

        # --- (后续的投影和拼接逻辑保持不变) ---
        H_E = self.manifold_H.proju(x_H, x_E)
        S_E = self.manifold_S.proju(x_S, x_E)
        H_E = self.manifold_H.transp0back(x_H, H_E)
        S_E = self.manifold_S.transp0back(x_S, S_E)
        E = torch.cat([x_E, H_E, S_E], dim=-1)
        x_E = self.proj(E)
        
        return x_E, x_H, x_S
