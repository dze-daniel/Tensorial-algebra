import torch
import torch.nn as nn
import string


class HighestRankTensorialAlgebra(nn.Module):
    def __init__(self, shape):
        """
        shape: tuple, e.g. (2, 3, 4, 5, 6)
        """
        super().__init__()
        self.shape = shape
        self.rank = len(shape)

        if self.rank > 17:
            raise ValueError(
                "Congratulations! You've reached 18+ dimensional tensors, which exceeds the PyTorch einsum symbol limit and ASCII alphabet capacity. "
                "This is not a mathematical limitation — you can multiply 18+ tensors, and you will, once PyTorch supports unicode einsum. 😁 "
                "Congratulations on reaching this level of deep mathematical tensor algebra!"
            )

        # Generate unique triple symbols for each axis
        self.symbols = list(string.ascii_letters)
        self.symbol_groups = [self.symbols[i*3:(i+1)*3] for i in range(self.rank)]

        # Register algebra tensors as learnable parameters
        self.algebras = nn.ParameterList([
            nn.Parameter(torch.randn(n, n, n)) for n in shape
        ])

        # Prepare einsum expression
        self._einsum_expr = self._build_einsum_expr()

    def _build_einsum_expr(self):
        idx1 = ''.join([g[0] for g in self.symbol_groups])  # e.g. adgjm
        idx2 = ''.join([g[1] for g in self.symbol_groups])  # e.g. behkn
        algebra = [ ''.join(g) for g in self.symbol_groups]  # ['abc', 'def', ...]
        out = ''.join([g[2] for g in self.symbol_groups])    # e.g. cfilo

        return f"{idx1}, {idx2}, " + ', '.join(algebra) + f" -> {out}"

    def forward(self, T, U):
        """
        T and U should have shape equal to self.shape
        """
        return torch.einsum(self._einsum_expr, T, U, *self.algebras)


# -------------------------------
# 🧪 Пример использования
# -------------------------------

if __name__ == "__main__":
    shape = (2, 3, 4, 5, 6)
    algebra_layer = HighestRankTensorialAlgebra(shape)
    
    T = torch.randn(*shape)
    U = torch.randn(*shape)
    
    result = algebra_layer(T, U)

    print("Result shape:", result.shape)
    print("Einsum expression:", algebra_layer._einsum_expr)
    print("Result tensor:\n", result)
