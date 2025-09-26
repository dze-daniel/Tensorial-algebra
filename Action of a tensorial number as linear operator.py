import torch

class TensorialAlgebra(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        # Параметр t — тензор структуры алгебры
        self.T = torch.nn.Parameter(torch.randn(n, n, n, dtype=torch.float64))

    def forward(self, x, y):
        """
        Умножение двух элементов алгебры: x * y
        Эйнштейнова свёртка: i,j,ijk->k
        """
        return torch.einsum("i,j,ijk->k", x, y, self.T)
    
    def number_as_operator(self, A, side="left"):
        """
        Действие тензориального числа A в алгебре T как линейного оператора.

        side = "left":  матрица M^L(A) = einsum("i,ijk->jk", A, T)
        side = "right": матрица M^R(A) = einsum("j,ijk->ik", A, T)

        Возвращает матрицу n×n.
        """
        if side == "left":
            return torch.einsum("i,ijk->jk", A, self.T)
        elif side == "right":
            return torch.einsum("j,ijk->ik", A, self.T)
        else:
            raise ValueError("side must be 'left' or 'right'")

    def search_unit(self, side="left", w=None):
        """
        Поиск левой или правой единицы.
        w — произвольный вектор для случая бесконечного множества решений.
        Возвращает:
            None — если решения нет,
            tensor — если решение единственное или вычислено по w.
        """
        n = self.n
        # Формируем матрицу A в зависимости от стороны
        if side == "left":
            A = self.T.permute(1, 2, 0).reshape(n*n, n)
        elif side == "right":
            A = self.T.permute(0, 2, 1).reshape(n*n, n)
        else:
            raise ValueError("side must be 'left' or 'right'")

        # b — vec(E_n)
        b = torch.eye(n, dtype=torch.float64).reshape(-1, 1)

        # Ранги
        rank_A = torch.linalg.matrix_rank(A)
        rank_Ab = torch.linalg.matrix_rank(torch.cat([A, b], dim=1))

        # Случай 1: Нет решения
        if rank_A != rank_Ab:
            return None

        # Случай 2: Единственное решение
        if rank_A == n:
            return torch.linalg.pinv(A) @ b

        # Случай 3: Бесконечно много решений
        A_plus = torch.linalg.pinv(A)
        proj = torch.eye(n, dtype=torch.float64) - A_plus @ A
        if w is None:
            raise ValueError("Need vector w for infinite solution case")
        return (A_plus @ b + proj @ w.reshape(-1, 1))


if __name__ == "__main__":
    n = 4
    alg = TensorialAlgebra(n)

    # Пример вектора A
    A = torch.randn(n, dtype=torch.float64)

    # Левый оператор
    ML = alg.number_as_operator(A, side="left")
    print("Левое действие:\n", ML)

    # Правый оператор
    MR = alg.number_as_operator(A, side="right")
    print("Правое действие:\n", MR)
