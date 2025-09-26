import torch
import torch.nn as nn
import torch.linalg as LA

class SpectralTopologicalSignature(nn.Module):
    def __init__(self, n: int, eps: float = 1e-6):
        """
        n: размерность векторов a и b (и размерность алгебры)
        eps: малое число для стабилизации логарифмов
        """
        super().__init__()
        self.n = n
        self.eps = eps
        self.T = nn.Parameter(torch.randn(n, n, n) * 0.01)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a, b: тензоры формы (batch_size, n)
        return: подпись формы (batch_size, n, 6)
        """
        assert a.shape == b.shape
        assert a.shape[-1] == self.n
        B = a.shape[0]

        # Тензорное умножение: C[batch, i, j, k] = a[batch, i] * b[batch, j] * T[i, j, k]
        C = torch.einsum('bi,bj,ijk->bijk', a, b, self.T)  # (B, n, n, n)
        C_perm = C.permute(0, 3, 1, 2)  # (B, k, i, j)

        # Определитель по каждой матрице C_k: (B, n)
        det = LA.det(C_perm)  # (B, n)

        # След: сумма диагоналей по каждой матрице C_k
        trace = C_perm.diagonal(dim1=2, dim2=3).sum(dim=-1)  # (B, n)

        # Сингулярные значения: (B, n, n)
        svdvals = LA.svdvals(C_perm)  # (B, n, n)
        sigma_min = svdvals[:, :, -1]  # (B, n)
        sigma_max = svdvals[:, :, 0]   # (B, n)

        # Аппроксимация ранга: сумма логарифмов сингулярных значений
        approx_rank = torch.log(svdvals + self.eps).sum(dim=-1)  # (B, n)

        # Спектральный радиус: abs(eigvals).max(dim)
        eigvals = LA.eigvals(C_perm)  # (B, n, n), комплексные
        rho = eigvals.abs().max(dim=-1).values  # (B, n)

        # Объединение признаков: (B, n, 6)
        signature = torch.stack([
            det,
            trace,
            approx_rank,
            sigma_min,
            sigma_max,
            rho
        ], dim=-1)

        return signature



# Пример использования
if __name__ == "__main__":
    n = 5  # размерность алгебры
    B = 2  # размер батча

    # Инициализация модели
    model = SpectralTopologicalSignature(n)

    # Примеры входных векторов (batch_size, n)
    a = torch.randn(B, n, requires_grad=True)
    b = torch.randn(B, n, requires_grad=True)

    # Вычисление спектрально-топологической подписи
    signature = model(a, b)  # shape: (B, n, 6)

    # Вывод формы и значений
    print("Signature shape:", signature.shape)  # (2, 5, 6)
    print("Signature[0]:\n", signature[0])       # Подпись для первого примера
    print("Signature[1]:\n", signature[1])       # Подпись для второго примера

    loss = signature.sum()
    loss.backward()

    print("Grad по a:\n", a.grad)  # покажет ∂loss/∂a
    print("Grad по T:\n", model.T.grad)  # покажет ∂loss/∂T
