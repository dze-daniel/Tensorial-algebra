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
    
        if side == "right":
            # A*B = I  ⇒ M^L(A) @ B = I
            M = self.number_as_operator(A, side="left")  # (n×n)
        elif side == "left":
            # B*A = I  ⇒ M^R(A) @ B = I
            M = self.number_as_operator(A, side="right")  # (n×n)
        else:
            raise ValueError("side must be 'left' or 'right'")

        # Решаем M @ B = I (B shape (n,)), I shape (n,)
        I_vec = I.reshape(-1)
        B = torch.linalg.lstsq(M, I_vec).solution  # правый/левый обратный
        return B, I  # B shape (n,), I shape (n,1)
    
    
    def divide(dividend, divider, side = "right"):
        divider_inv, I = self.search_inverse(divider, side=side)
        if side == "right":
            return torch.einsum("i,j,ijk->k", dividend, divider_inv, self.T)
        else:
            return torch.einsum("i,j,ijk->k", divider_inv, dividend, self.T)
    
    @torch.no_grad()
    def classify_algebra(self, tol=1e-10, quasi_tol=1e-8, norm_trials=256, rng=None):
        """
        Классификация свойств алгебры по тензору T.

        Возвращает dict с флагами:
        associative, left_alternative, right_alternative, flexible,
        jacobi_identity, commutative, anti_commutative,
        quasi_commutative (и omega), normed_multiplicative, lie_algebra,
        и 'notes' с метриками ошибок.
        """
        T = self.T
        n = self.n
        notes = {}

        # ---------- Ассоциативность ----------
        left_assoc  = torch.einsum('ijm,mkl->ijkl', T, T)
        right_assoc = torch.einsum('jkm,iml->ijkl', T, T)
        assoc_err = (left_assoc - right_assoc).abs().max().item()
        associative = assoc_err <= tol
        if not associative:
            notes['assoc_max_abs_err'] = assoc_err

        # ---------- Альтернативность ----------
        L1 = torch.einsum('iim,mjl->ijl', T, T)
        R1 = torch.einsum('ijm,iml->ijl', T, T)
        left_alt_err = (L1 - R1).abs().max().item()
        left_alternative = left_alt_err <= tol
        if not left_alternative:
            notes['left_alt_max_abs_err'] = left_alt_err

        L2 = torch.einsum('ijm,mjl->ijl', T, T)
        R2 = torch.einsum('jjm,iml->ijl', T, T)
        right_alt_err = (L2 - R2).abs().max().item()
        right_alternative = right_alt_err <= tol
        if not right_alternative:
            notes['right_alt_max_abs_err'] = right_alt_err

        # ---------- Гибкость ----------
        Lf = torch.einsum('jim,iml->jil', T, T)
        Rf = torch.einsum('ijm,mil->jil', T, T)
        flex_err = (Lf - Rf).abs().max().item()
        flexible = flex_err <= tol
        if not flexible:
            notes['flex_max_abs_err'] = flex_err

        # ---------- Коммутативность / антикоммутативность ----------
        Tswap = T.transpose(0, 1)
        comm_err  = (T - Tswap).abs().max().item()
        anti_err  = (T + Tswap).abs().max().item()
        commutative        = comm_err <= tol
        anti_commutative   = anti_err <= tol

        # ---------- Квазикоммутативность ----------
        denom = (Tswap*Tswap).sum()
        if denom > 0:
            omega = (T*Tswap).sum() / denom
            residual = torch.norm(T - omega*Tswap) / (torch.norm(T) + 1e-18)
            quasi_commutative = (residual.item() < quasi_tol) and (abs(omega.item()-1.0) > 1e-12) and (abs(omega.item()+1.0) > 1e-12)
            notes['quasi_commut_residual'] = residual.item()
            omega_val = omega.item()
        else:
            quasi_commutative = False
            omega_val = None
            notes['quasi_commut_residual'] = None

        # ---------- Идентичность Якоби (для коммутатора) ----------
        C = T - Tswap
        Jac = (torch.einsum('jkm,iml->ijkl', C, C) +
               torch.einsum('kim,jml->ijkl', C, C) +
               torch.einsum('ijm,kml->ijkl', C, C))
        jacobi_err = Jac.abs().max().item()
        jacobi_identity = jacobi_err <= tol
        if not jacobi_identity:
            notes['jacobi_max_abs_err'] = jacobi_err

        # Ли-алгебра по коммутатору: билинейность + антисимметрия (у C) + Якоби
        lie_algebra = jacobi_identity

        # ---------- Нормированность (эмпирически, евклидова норма) ----------
        if rng is None:
            gen = torch.Generator(device=T.device)
            gen.manual_seed(0)
        else:
            gen = rng
        ok = True
        max_rel_err = 0.0
        for _ in range(norm_trials):
            a = torch.randn(n, dtype=torch.float64, generator=gen, device=T.device)
            b = torch.randn(n, dtype=torch.float64, generator=gen, device=T.device)
            ab = torch.einsum('i,j,ijk->k', a, b, T)
            na = torch.linalg.norm(a).item()
            nb = torch.linalg.norm(b).item()
            nab = torch.linalg.norm(ab).item()
            if na < 1e-14 or nb < 1e-14:
                continue
            rel = abs(nab - na*nb) / (na*nb)
            max_rel_err = max(max_rel_err, rel)
            if rel > 1e-8:
                ok = False
        normed_multiplicative = ok
        notes['norm_mult_max_rel_err'] = max_rel_err

        return {
            'associative': associative,
            'left_alternative': left_alternative,
            'right_alternative': right_alternative,
            'flexible': flexible,
            'jacobi_identity': jacobi_identity,
            'commutative': commutative,
            'anti_commutative': anti_commutative,
            'quasi_commutative': quasi_commutative,
            'quasi_commutative_omega': omega_val,
            'normed_multiplicative': normed_multiplicative,
            'lie_algebra': lie_algebra,
            'notes': notes
        }



if __name__ == "__main__":

    # Функция для печати свойств
    def check_props(name, alg):
        print(f"\n{name}:")
        props = alg.classify_algebra()
        for k, v in props.items():
            print(f"  {k}: {v}")

    # --- 1. Кватернионы ---
    # Базис: 1, i, j, k
    T_quat = torch.zeros((4, 4, 4), dtype=torch.float64)
    # Правила: i^2=j^2=k^2=ijk=-1
    e = {0: "1", 1: "i", 2: "j", 3: "k"}
    mult = {
        (0, 0): (1, 0), (0, 1): (1, 1), (0, 2): (1, 2), (0, 3): (1, 3),
        (1, 0): (1, 1), (1, 1): (-1, 0), (1, 2): (1, 3), (1, 3): (-1, 2),
        (2, 0): (1, 2), (2, 1): (-1, 3), (2, 2): (-1, 0), (2, 3): (1, 1),
        (3, 0): (1, 3), (3, 1): (1, 2), (3, 2): (-1, 1), (3, 3): (-1, 0),
    }
    for (i, j), (sgn, k) in mult.items():
        T_quat[i, j, k] = sgn
    alg_quat = TensorialAlgebra(4)
    alg_quat.T.data = T_quat
    check_props("Кватернионы", alg_quat)

    # --- 2. Векторное произведение в R^3 ---
    T_cross = torch.zeros((3, 3, 3), dtype=torch.float64)
    T_cross[0, 1, 2] = 1
    T_cross[1, 0, 2] = -1
    T_cross[2, 0, 1] = 1
    T_cross[0, 2, 1] = -1
    T_cross[1, 2, 0] = 1
    T_cross[2, 1, 0] = -1
    alg_cross = TensorialAlgebra(3)
    alg_cross.T.data = T_cross
    check_props("Векторное произведение", alg_cross)

    # --- 3. Случайная алгебра ---
    torch.manual_seed(0)
    T_rand = torch.randn((3, 3, 3), dtype=torch.float64)
    alg_rand = TensorialAlgebra(3)
    alg_rand.T.data = T_rand
    check_props("Случайная алгебра", alg_rand)

