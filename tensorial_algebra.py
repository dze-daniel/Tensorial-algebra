import torch
import torch.nn as nn
import numpy as np
import math
from functools import reduce
from operator import mul

class TensorialAlgebra(nn.Module):
    def __init__(self, n: int, algebra_tensor=None):
        """
        n: размерность выравненного вектора
        algebra_tensor: (n, n, n) — тензор алгебры (T_{ijk})
        """
        super().__init__()
        self.n = n

        if algebra_tensor is None:
            T = torch.zeros((n, n, n), dtype=torch.float32)
        elif isinstance(algebra_tensor, np.ndarray):
            T = torch.tensor(algebra_tensor, dtype=torch.float32)
        elif isinstance(algebra_tensor, torch.Tensor):
            T = algebra_tensor.float()
        else:
            raise TypeError("algebra_tensor must be None, np.ndarray, or torch.Tensor")

        # Проверка формы тензора алгебры
        if T.shape != (n, n, n):
            raise ValueError(f"Algebra tensor must have shape ({n}, {n}, {n}), but got {T.shape}")

        self.T = nn.Parameter(T)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Выполняет тензориальное умножение двух тензоров a и b
        с приведением их к размерности (n,) и последующим восстановлением исходной формы
        """
        if a.shape != b.shape:
            raise ValueError(f"Input tensors must have the same shape, but got {a.shape} and {b.shape}")

        flat_dim = reduce(mul, a.shape, 1)
        if flat_dim != self.n:
            raise ValueError(f"Total number of elements in input tensors must be {self.n}, but got {flat_dim}")

        # Сохраняем исходную форму
        original_shape = a.shape

        # Выравнивание
        a_flat = a.reshape(self.n)
        b_flat = b.reshape(self.n)

        # Эйнштейнова свёртка: a_i * b_j * T_{ijk}
        result_flat = torch.einsum('i,j,ijk->k', a_flat, b_flat, self.T)

        # Возвращаем результат к исходной форме
        result = result_flat.reshape(original_shape)
        return result
