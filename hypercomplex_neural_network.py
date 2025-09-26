import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TensorialAlgebraicAttention(nn.Module):
    def __init__(self, dem):
        super().__init__()
        self.dem = dem
        self.T = nn.Parameter(torch.randn(dem, dem, dem) * 0.01)

    def forward(self, Q: torch.Tensor, K: torch.Tensor):
        # Q: [n, d], K: [n, d]
        # хотим получить [n, n, d] через тензориальное умножение

        # Эйнштейново умножение: q_i * k_j * T_{ijk} → r_k
        # объединённое свёртывание по батчу:
        # Q[n, i], K[m, j], T[i,j,k] → A[n, m, k]
        A = torch.einsum('ni,mj,ijk->nmk', Q, K, self.T)
        return A


class HypercomplexSelfAttention(nn.Module):
    def __init__(self, dem, seq_len):
        super().__init__()
        self.dem = dem
        self.seq_len = seq_len

        self.algebraic_attention = TensorialAlgebraicAttention(dem)
        self.Wq = nn.Linear(dem, dem)
        self.Wk = nn.Linear(dem, dem)
        
        # Тензор весов: [n, dem, dem]
        self.V = nn.Parameter(torch.randn(seq_len, dem, dem) * 0.01)

    def forward(self, X):
        # X: [n, dem]
        Q = self.Wq(X)  # [n, dem]
        K = self.Wk(X)  # [n, dem]

        # Внимание: [n, n, dem]
        attn = self.algebraic_attention(Q, K)

        # Масштаб и softmax
        attn = attn / math.sqrt(self.dem)
        attn_weights = F.softmax(attn, dim=1)  # по оси m

        # Тензориальное умножение с обучаемым тензором V: [n, n, d] × [n, d, d] → [n, d]
        output = torch.einsum('nmk,mkd->nd', attn_weights, self.V)

        return output
