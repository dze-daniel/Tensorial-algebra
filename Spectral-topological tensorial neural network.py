import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralTopologicalAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        v_dim = d
        self.d = d  # размерность эмбеддингов
        self.v_dim = v_dim  # выходная размерность value

        # Обучаемые матрицы проекций
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wq = nn.Linear(d, d, bias=False)

        # Обучаемый тензор алгебры (d, d, d)
        self.T = nn.Parameter(torch.randn(d, d, d) * 0.01)

        # Обучаемый тензор значений (d, 6, d, v)
        self.Wv = nn.Parameter(torch.randn(d, 6, d, v_dim) * 0.01)

    def spectral_signature(self, K, Q):
        """
        K, Q: (B, N, d)
        Возвращает спектрально-топологическую подпись: (B, N, N, 6, d)
        """
        B, N, d = K.shape
        T = self.T  # (d, d, d)

        # Тензорное взаимодействие: (B, N, N, d, d, d)
        C = torch.einsum('bni,bmj,ijk->bnmijk', Q, K, T)  # [B,N,N,d,d,d]
        C_flat = C.reshape(B * N * N, d, d, d).permute(0, 3, 1, 2)  # [BNN, d, d, d] → [BNN, d, d, d] → [BNN*d, d, d]
        C_perm = C_flat.reshape(-1, d, d)  # (BNN*d, d, d)

        # Спектрально-топологическая подпись: (BNN, d, 6)
        det = torch.linalg.det(C_perm).reshape(-1, d)  # (BNN, d)
        trace = C_perm.diagonal(dim1=1, dim2=2).sum(-1).reshape(-1, d)  # (BNN, d)
        svd = torch.linalg.svdvals(C_perm).reshape(-1, d, d)
        sigma_min = svd[:, :, -1]  # (BNN, d)
        sigma_max = svd[:, :, 0]   # (BNN, d)
        rank_approx = torch.log(svd + 1e-6).sum(-1)  # (BNN, d)
        eigvals = torch.linalg.eigvals(C_perm).reshape(-1, d, d)
        rho = eigvals.abs().amax(-1)  # (BNN, d)

        signature = torch.stack([det, trace, rank_approx, sigma_min, sigma_max, rho], dim=1)  # (BNN, 6, d)
        signature = signature.permute(0, 2, 1).reshape(B, N, N, 6, d)  # (B, N, N, 6, d)
        return signature

    def forward(self, X):
        """
        X: (B, N, d) — входной тензор
        Возвращает: (B, N, v_dim) — выход значения attention
        """
        B, N, d = X.shape
        K = torch.einsum('bnd,dk->bnk', X, self.Wk)  # (B, N, d)
        Q = torch.einsum('bnd,dq->bnq', X, self.Wq)  # (B, N, d)

        # Вычисление спектрально-топологической подписи: (B, N, N, 6, d)
        Sign = self.spectral_signature(K, Q)  # [B,N,N,6,d]

        # Построение value: V[b, m, s, k, v] = X × Wv
        V = torch.einsum('bmd,dskv->bmskv', X, self.Wv)  # (B, M=N, 6, d, v)

        # Attention свёртка: Sign[b,n,m,s,k], V[b,m,s,k,v] → Out[b,n,v]
        Out = torch.einsum('bnmsk,bmskv->bnv', Sign, V)  # (B, N, v_dim)

        return Out
