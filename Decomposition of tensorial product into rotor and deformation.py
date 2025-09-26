import torch
import torch.nn as nn

class TensorialDecomposition(nn.Module):
    """
    Тензориальная нейросеть, вычисляющая деформацию и ротор тензориального произведения двух векторов
    по обучаемому тензору алгебры T ∈ ℝ^{n × n × n}.
    """

    def __init__(self, n, init_std=0.01):
        """
        :param n: размерность входных векторов (a и b)
        :param init_std: стандартное отклонение для инициализации T
        """
        super().__init__()
        self.n = n
        self.T = nn.Parameter(torch.randn(n, n, n) * init_std)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        """
        :param a: вектор размера (n,)
        :param b: вектор размера (n,)
        :return: кортеж (def, rot) — деформация и ротор, оба размерности (n,)
        """
        assert a.shape == b.shape == (self.n,), "Input vectors must have shape (n,)"

        # Внешнее произведение
        outer = torch.einsum('i,j->ij', a, b)

        # Симметричная и антисимметричная части
        S = (outer + outer.T) / 2  # деформация
        A = (outer - outer.T) / 2  # ротор

        # Свёртки с обучаемым тензором T
        deformation = torch.einsum('ij,ijk->k', S, self.T)
        rotation = torch.einsum('ij,ijk->k', A, self.T)

        return deformation, rotation
