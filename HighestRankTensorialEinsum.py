import unicodedata
import string
from dataclasses import dataclass
import torch
import torch.nn as nn


# ----------- Запись про каждый символ исходной формулы -----------
@dataclass
class SymbolEntry:
    orig: str            # исходный символ в формуле
    norm: str            # нормализованный одиночный символ (напр., '❤️' -> '❤')
    arg: int             # 0=A, 1=B, 2=Out
    pos_in_arg: int      # позиция внутри аргумента (с повторениями)
    is_tensorial: bool   # True если тензориальная ось (не ASCII буква)
    set_index: int       # порядковый индекс в своём множестве (заполняется позже)


class HighestRankTensorialEinsum(nn.Module):
    """
    Тензориальная свёртка высших рангов с менеджментом формулы.
    Порядок ключей в tensorial_dims и порядок появления символов в формуле НЕ важен:
    алгебры подаются в einsum в порядке, вычисленном из самой формулы.

    - Латинские a..zA..Z → обычные эйнштейновы индексы.
    - Прочие (например $, %, ❤, 爱, …) → тензориальные оси с собственным T ∈ R^{n×n×n}.
    """

    # ---------- Unicode утилиты ----------
    @staticmethod
    def _clean_expression(expr: str) -> str:
        """Удаляет VARIATION SELECTOR-16 (U+FE0F), чтобы '❤️' всегда стал '❤'."""
        return ''.join(c for c in expr if ord(c) != 0xFE0F)

    @staticmethod
    def _normalize_char(c: str) -> str:
        """
        Нормализует символ к одиночному базовому (NFKD) и возвращает первый
        не-M/C/Z символ. Пример: '❤️' -> '❤', '️' -> '' (отбрасывается).
        """
        norm = unicodedata.normalize("NFKD", c)
        for ch in norm:
            if unicodedata.category(ch)[0] not in ('M', 'C', 'Z'):
                return ch
        return ''

    def __init__(self, expr: str, tensorial_dims: dict[str, int]):
        super().__init__()
        expr = self._clean_expression(expr)
        self.original_expr = expr

        # Нормализуем ключи dim'ов (чтобы '❤️' и '❤' не путались)
        self.tensorial_dims = {self._normalize_char(k): v for k, v in tensorial_dims.items()}

        # Парсинг "A,B->O"
        try:
            lhs, rhs = expr.split("->")
            A_expr, B_expr = lhs.split(",")
        except ValueError:
            raise ValueError("Ожидалось выражение вида 'A,B->O'.")

        # 1) Первичный список символов (с повторениями) и мета
        self.primary: list[SymbolEntry] = []
        self._build_primary(A_expr, B_expr, rhs)

        # 2) Множества и индексация
        self.einstein_syms, self.tensorial_syms = self._build_sets()

        # 2a) Проверка, что все тензориальные оси имеют заявленный размер
        missing = [s for s in self.tensorial_syms if s not in self.tensorial_dims]
        if missing:
            raise ValueError(f"Не заданы размерности для тензориальных осей: {missing}")

        # 3) Назначаем ASCII-метки:
        #    - Эйнштейновы: по одной букве
        #    - Тензориальные: по тройке (i_A, j_B, k_Out) на символ
        used_latin = self._collect_used_latin(A_expr, B_expr, rhs)
        self.einstein_map, self.tensorial_triples = self._assign_ascii_letters(used_latin)

        # 4) Собираем чистую ASCII einsum-строку
        self.einsum_expr, algebra_symbols_in_expr = self._assemble_einsum(A_expr, B_expr, rhs)

        # 5) Создаём параметры-алгебры ПО САМИМ СИМВОЛАМ, а подаём их в einsum
        #    в порядке algebra_symbols_in_expr (чтобы порядок был независим).
        self._algebra_by_symbol = nn.ParameterDict({
            sym: nn.Parameter(torch.randn(self.tensorial_dims[sym],
                                          self.tensorial_dims[sym],
                                          self.tensorial_dims[sym]))
            for sym in self.tensorial_syms
        })
        self._algebra_symbols_in_expr = algebra_symbols_in_expr  # порядок подачи в einsum

    # ---------- Шаг 1: первичный список ----------
    def _build_primary(self, A: str, B: str, O: str):
        for arg_idx, s in enumerate([A, B, O]):
            pos = 0
            for ch in s:
                n = self._normalize_char(ch)
                if not n:
                    continue
                is_tensorial = not (n.isascii() and n.isalpha())
                self.primary.append(SymbolEntry(
                    orig=ch, norm=n, arg=arg_idx, pos_in_arg=pos,
                    is_tensorial=is_tensorial, set_index=-1
                ))
                pos += 1

    # ---------- Шаг 2: множества и индексация ----------
    def _build_sets(self):
        ein_order, ein_seen = [], set()
        ten_order, ten_seen = [], set()
        # множества по порядку первого появления (внутри каждого типа)
        for e in self.primary:
            if e.is_tensorial:
                if e.norm not in ten_seen:
                    ten_seen.add(e.norm)
                    ten_order.append(e.norm)
            else:
                if e.norm not in ein_seen:
                    ein_seen.add(e.norm)
                    ein_order.append(e.norm)
        ein_idx = {s: i for i, s in enumerate(ein_order)}
        ten_idx = {s: i for i, s in enumerate(ten_order)}
        for e in self.primary:
            e.set_index = ten_idx[e.norm] if e.is_tensorial else ein_idx[e.norm]
        return ein_order, ten_order

    # ---------- Поддержка: какие латинские уже заняты в формуле ----------
    def _collect_used_latin(self, A: str, B: str, O: str):
        used = set()
        for ch in (A + B + O):
            n = self._normalize_char(ch)
            if n and n.isascii() and n.isalpha():
                used.add(n)
        return used

    # ---------- Шаг 3: резервирование ASCII ----------
    def _assign_ascii_letters(self, used_latin: set[str]):
        letters = list(string.ascii_letters)   # 'a'..'z' + 'A'..'Z'
        used = set(used_latin)                 # сохраняем уже занятые формулой латинские
        def next_free():
            while letters and letters[0] in used:
                letters.pop(0)
            if not letters:
                raise RuntimeError("Закончилось место в ascii_letters.")
            return letters.pop(0)

        einstein_map = {}
        for s in self.einstein_syms:
            c = next_free()
            used.add(c)
            einstein_map[s] = c

        tensorial_triples = {}
        for s in self.tensorial_syms:
            i = next_free(); used.add(i)
            j = next_free(); used.add(j)
            k = next_free(); used.add(k)
            tensorial_triples[s] = (i, j, k)

        return einstein_map, tensorial_triples

    # ---------- Шаг 4: сборка einsum-строки (и списка алгебр по формуле) ----------
    def _assemble_einsum(self, A: str, B: str, O: str):
        parts = {0: [], 1: [], 2: []}
        for e in self.primary:
            parts[e.arg].append(e)

        def build_side(items, which: int):
            out = []
            for it in items:
                if it.is_tensorial:
                    i, j, k = self.tensorial_triples[it.norm]
                    out.append(i if which == 0 else (j if which == 1 else k))
                else:
                    out.append(self.einstein_map[it.norm])
            return ''.join(out)

        A_new = build_side(parts[0], 0)
        B_new = build_side(parts[1], 1)
        O_new = build_side(parts[2], 2)

        # Порядок алгебр берём по порядку первого появления тензориальной оси в ИСХОДНОЙ формуле.
        # Это стабильно и не зависит от порядка в словаре dim'ов.
        algebra_symbols_in_expr = []
        seen = set()
        for e in self.primary:
            if e.is_tensorial and e.norm not in seen:
                seen.add(e.norm)
                algebra_symbols_in_expr.append(e.norm)

        algebra_terms = []
        for sym in algebra_symbols_in_expr:
            i, j, k = self.tensorial_triples[sym]
            algebra_terms.append(i + j + k)

        einsum = f"{A_new},{B_new}," + ",".join(algebra_terms) + f"->{O_new}"
        return einsum, algebra_symbols_in_expr

    # ---------- Проверка согласованности размеров ----------
    def _collect_operand_labels(self):
        lhs, rhs = self.einsum_expr.split("->")
        labels = lhs.split(",")
        return labels[:2], labels[2:], rhs  # (A,B), [alg...], out

    def _check_shapes(self, A: torch.Tensor, B: torch.Tensor):
        (labA, labB), alg_labels, out_labels = self._collect_operand_labels()

        # Собираем операнды в точном порядке, который закодировали в einsum:
        # A, B, затем T(sym) в порядке algebra_symbols_in_expr
        operands = [A, B] + [self._algebra_by_symbol[sym] for sym in self._algebra_symbols_in_expr]
        labels   = [labA, labB] + alg_labels

        # 1) Ранги должны совпасть с длиной строки меток
        for t, labs in zip(operands, labels):
            if len(t.shape) != len(labs):
                raise ValueError(f"Несоответствие ранга: shape={t.shape}, labels='{labs}'")

        # 2) Каждая метка должна иметь один и тот же размер везде, где встретилась
        sym2dim = {}
        for t, labs in zip(operands, labels):
            for d, s in zip(t.shape, labs):
                if s in sym2dim and sym2dim[s] != d:
                    raise ValueError(f"Конфликт по символу '{s}': {d} != {sym2dim[s]}")
                sym2dim.setdefault(s, d)

        # 3) Доп.проверка тензориальных: i/j/k должны равняться заявленному dim
        for sym in self.tensorial_syms:
            dim = self.tensorial_dims[sym]
            i, j, k = self.tensorial_triples[sym]
            for s in (i, j, k):
                if s in sym2dim and sym2dim[s] != dim:
                    raise ValueError(
                        f"Тензориальная ось '{sym}' должна иметь размер {dim}, "
                        f"но метка '{s}' имеет {sym2dim[s]}"
                    )

    # ---------- forward ----------
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        self._check_shapes(A, B)
        # Передаём алгебры в ТОМ ЖЕ порядке, что и их метки в einsum:
        algebras_in_order = [self._algebra_by_symbol[sym] for sym in self._algebra_symbols_in_expr]
        return torch.einsum(self.einsum_expr, A, B, *algebras_in_order)


# ================== Пример ==================
if __name__ == "__main__":
    # Формула с перемешанными индексами, в т.ч. тензориальными
    expr = "nmk$%&❤爱,爱knp$%&❤->mp$%&❤爱"

    # Все размерности разные (чтобы легко ловить ошибки)
    # Тензориальные оси:
    #   $=2, %=3, &=5, ❤=7, 爱=11   (все разные)
    dims = {'$': 2, '%': 3, '&': 5, '❤️': 7, '爱': 11}  # '❤️' нормализуется до '❤'

    layer = HighestRankTensorialEinsum(expr, dims)

    # Эйнштейновы индексы: m=13, n=17, k=19, p=23 — тоже разные
    m, n, k, p = 13, 17, 19, 23
    ds , dp, da, d_heart, d_ai = 2, 3, 5, 7, 11

    # Формула слева:  A = n m k $ % & ❤ 爱 → (n, m, k, $, %, &, ❤, 爱)
    A = torch.randn(n, m, k, ds, dp, da, d_heart, d_ai)
    # Справа: B = 爱 k n p $ % & ❤ → (爱, k, n, p, $, %, &, ❤)
    B = torch.randn(d_ai, k, n, p, ds, dp, da, d_heart)

    out = layer(A, B)
    print("EINSUM:", layer.einsum_expr)
    print("OUT SHAPE:", out.shape)
    # Ожидаемые оси результата по формуле: m p $ % & ❤ 爱 → (m, p, $, %, &, ❤, 爱)
    assert list(out.shape) == [m, p, ds, dp, da, d_heart, d_ai]
    print("✓ Размерности результата совпали.")
