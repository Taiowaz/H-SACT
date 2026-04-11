import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from riemanngfm.modules.layers import EuclideanEncoder, ManifoldEncoder
from riemanngfm.modules.basics import (
    HyperbolicStructureLearner,
    SphericalStructureLearner,
    StarStructureLearner,
)
from riemanngfm.manifolds import Lorentz, Sphere, ProductSpace


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
        # ===== 曲率消融参数（新增，默认兼容原逻辑）=====
        learnable_curvature=False,
        k_h_init=1.0,
        k_s_init=1.0,
        curvature_lr_scale=0.1,
        curvature_clip_min=1e-3,
        curvature_clip_max=1e3,
        **kwargs,
    ):
        super(GeoGFM, self).__init__()

        # manifold（先按原方式初始化，曲率稍后同步）
        self.manifold_H = Lorentz()
        self.manifold_S = Sphere()

        # ===== 曲率配置 =====
        self.learnable_curvature = bool(learnable_curvature)
        self.curvature_lr_scale = float(curvature_lr_scale)
        self.curvature_clip_min = float(curvature_clip_min)
        self.curvature_clip_max = float(curvature_clip_max)

        # 用“幅值参数”做稳定化：k_h = -softplus(raw_h), k_s = +softplus(raw_s)
        # 你传入的 k_h_init/k_s_init 作为幅值初始化（通常都传正值 1.0）
        k_h_mag_init = max(float(abs(k_h_init)), self.curvature_clip_min)
        k_s_mag_init = max(float(abs(k_s_init)), self.curvature_clip_min)

        # 反 softplus 近似初始化，避免初值偏差太大
        # raw = log(exp(x)-1)
        def _inv_softplus(x: float) -> float:
            x = max(x, 1e-12)
            return float(np.log(np.exp(x) - 1.0 + 1e-12))

        if self.learnable_curvature:
            self.k_h_raw = nn.Parameter(
                torch.tensor(_inv_softplus(k_h_mag_init), dtype=torch.float32)
            )
            self.k_s_raw = nn.Parameter(
                torch.tensor(_inv_softplus(k_s_mag_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "k_h_raw",
                torch.tensor(_inv_softplus(k_h_mag_init), dtype=torch.float32),
            )
            self.register_buffer(
                "k_s_raw",
                torch.tensor(_inv_softplus(k_s_mag_init), dtype=torch.float32),
            )

        # 初始化时先同步一次曲率
        self._sync_manifold_curvature()

        self.product = ProductSpace(
            *[(self.manifold_H, embed_dim), (self.manifold_S, embed_dim)]
        )
        self.init_block = InitBlock(
            self.manifold_H,
            self.manifold_S,
            in_dim,
            hidden_dim,
            embed_dim,
            bias,
            activation,
            dropout,
        )
        self.blocks = nn.ModuleList([])
        for i in range(n_layers):
            self.blocks.append(
                StructuralBlock(
                    self.manifold_H,
                    self.manifold_S,
                    embed_dim,
                    hidden_dim,
                    embed_dim,
                    dropout,
                )
            )
        self.proj = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    # ===== 曲率工具函数 =====
    def _current_curvature(self):
        # magnitude in [clip_min, clip_max]
        k_h_mag = F.softplus(self.k_h_raw).clamp(
            self.curvature_clip_min, self.curvature_clip_max
        )
        k_s_mag = F.softplus(self.k_s_raw).clamp(
            self.curvature_clip_min, self.curvature_clip_max
        )

        # 按审稿实验定义：hyperbolic 负，spherical 正
        k_h = -k_h_mag
        k_s = k_s_mag
        return k_h, k_s

    def _assign_manifold_attr(self, manifold, name, value):
        """
        安全写入 manifold 曲率字段。
        - 若字段是 nn.Parameter / Tensor：原地 copy_，不替换对象
        - 若字段是普通 Python 数值：直接 setattr
        """
        if not hasattr(manifold, name):
            return False

        cur = getattr(manifold, name)

        # 情况1: Parameter（你当前报错就是这个）
        if isinstance(cur, torch.nn.Parameter):
            with torch.no_grad():
                v = value.detach().to(device=cur.device, dtype=cur.dtype)
                if cur.data.shape != v.shape:
                    v = v.reshape(cur.data.shape)
                cur.data.copy_(v)
            return True

        # 情况2: Tensor buffer
        if torch.is_tensor(cur):
            with torch.no_grad():
                v = value.detach().to(device=cur.device, dtype=cur.dtype)
                if cur.shape != v.shape:
                    v = v.reshape(cur.shape)
                cur.copy_(v)
            return True

        # 情况3: 普通标量
        setattr(manifold, name, float(value.detach().cpu().item()))
        return True

    def _sync_manifold_curvature(self):
        """
        把当前曲率写回 manifold 对象。
        为兼容不同 manifold 实现，尝试常见字段名：k / c / curvature。
        """
        k_h, k_s = self._current_curvature()

        # 超曲率
        ok_h = (
            self._assign_manifold_attr(self.manifold_H, "k", k_h)
            or self._assign_manifold_attr(self.manifold_H, "c", k_h)
            or self._assign_manifold_attr(self.manifold_H, "curvature", k_h)
        )
        # 球曲率
        ok_s = (
            self._assign_manifold_attr(self.manifold_S, "k", k_s)
            or self._assign_manifold_attr(self.manifold_S, "c", k_s)
            or self._assign_manifold_attr(self.manifold_S, "curvature", k_s)
        )

        # 如果你的 manifold 类字段名不同，这里会静默继续；你可按实际类名补字段
        if (not ok_h) or (not ok_s):
            pass

    def get_curvature_dict(self):
        """便于日志记录：当前曲率数值。"""
        k_h, k_s = self._current_curvature()
        return {
            "k_h": float(k_h.detach().cpu().item()),
            "k_s": float(k_s.detach().cpu().item()),
            "learnable_curvature": self.learnable_curvature,
            "curvature_lr_scale": self.curvature_lr_scale,
        }

    def curvature_parameters(self):
        """便于外部 optimizer 做单独 lr 分组。"""
        if self.learnable_curvature:
            return [self.k_h_raw, self.k_s_raw]
        return []

    def forward(self, data):
        """
        :param data: Dataset for a graph contains batched sub-trees.
        :return: z: node product representations
        """
        # 每次前向前同步，确保用到当前曲率
        self._sync_manifold_curvature()

        x = data.x.clone()
        x_E, x_H, x_S = self.init_block(
            x, data.edge_index, data.tokens(data.n_id)
        )  # [N, Hidden]
        for i, block in enumerate(self.blocks):
            x_E, x_H, x_S = block((x_E, x_H, x_S), data)
        return x_E, x_H, x_S

    def loss(self, x_tuple):
        """
        :param x_tuple: (x_E, x_H, x_S)
        :return:
        """
        # loss 中也同步一次，防止外部先调 forward 后曲率被改动
        self._sync_manifold_curvature()

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
        norm1 = x1.norm(dim=-1)
        norm2 = x2.norm(dim=-1)
        sim_matrix = torch.einsum("ik,jk->ij", x1, x2) / (
            torch.einsum("i,j->ij", norm1, norm2) + EPS
        )
        sim_matrix = torch.exp(sim_matrix / 0.2)
        pos_sim = sim_matrix.diag()
        loss_1 = pos_sim / (sim_matrix.sum(dim=-2) + EPS)
        loss_2 = pos_sim / (sim_matrix.sum(dim=-1) + EPS)

        loss_1 = -torch.log(loss_1).mean()
        loss_2 = -torch.log(loss_2).mean()
        loss = (loss_1 + loss_2) / 2.0
        return loss


class InitBlock(nn.Module):
    def __init__(
        self,
        manifold_H,
        manifold_S,
        in_dim,
        hidden_dim,
        out_dim,
        bias,
        activation,
        dropout,
    ):
        super(InitBlock, self).__init__()
        self.Euc_init = EuclideanEncoder(
            in_dim, hidden_dim, out_dim, bias, activation, dropout
        )
        self.Hyp_init = ManifoldEncoder(
            manifold_H, in_dim, hidden_dim, out_dim, bias, None, 0.0
        )
        self.Sph_init = ManifoldEncoder(
            manifold_S, in_dim, hidden_dim, out_dim, bias, None, 0.0
        )

    def forward(self, x, edge_index, tokens):
        """

        :param tokens: input tokens
        :param x: raw features
        :param edge_index: edges
        :return: (E, H, S) Manifold initial representations
        """
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

        # 默认通用树学习器 (BFS树)
        self.Hyp_learner = HyperbolicStructureLearner(
            self.manifold_H, self.manifold_S, in_dim, hidden_dim, out_dim, dropout
        )
        # 星型图学习器
        self.Star_learner = StarStructureLearner(
            self.manifold_H, self.manifold_S, in_dim, hidden_dim, out_dim, dropout
        )
        # 球面学习器
        self.Sph_learner = SphericalStructureLearner(
            self.manifold_H, self.manifold_S, in_dim, hidden_dim, out_dim, dropout
        )

        self.proj = nn.Sequential(
            nn.Linear(3 * out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x_tuple, data):
        """
        根据数据是否存在 batch_star 动态选择学习器
        """
        x_E, x_H, x_S = x_tuple

        if hasattr(data, "batch_star") and data.batch_star is not None:
            x_H = self.Star_learner(x_H, x_S, data.batch_star)
        else:
            x_H = self.Hyp_learner(x_H, x_S, data.batch_tree)

        x_S = self.Sph_learner(x_H, x_S, data)

        H_E = self.manifold_H.proju(x_H, x_E)
        S_E = self.manifold_S.proju(x_S, x_E)
        H_E = self.manifold_H.transp0back(x_H, H_E)
        S_E = self.manifold_S.transp0back(x_S, S_E)
        E = torch.cat([x_E, H_E, S_E], dim=-1)
        x_E = self.proj(E)

        return x_E, x_H, x_S
