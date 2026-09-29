from fractions import Fraction
from typing import List


class SimplexSolver:
    """
    Универсальный класс для решения задач симплекс-методом
    - Решаем только задачи W -> min
    - Принимает задачу в общем виде, сам приводит к каноническому виду
    """

    def __init__(self, c, A, b, signs, maximize=True):
        """
        c        - коэффициенты целевой функции
        A        - матрица ограничений
        b        - правая часть
        signs    - знаки ограничений: '<=', '>=', '='
        maximize - True если исходно Z -> max, False если W -> min
        """
        self.n = len(c)
        self.m = len(A)

        # приводим целевую функцию к минимизации
        factor = -1 if maximize else 1
        self.c_orig = [Fraction(factor * v) for v in c]

        self.A_orig = [[Fraction(v) for v in row] for row in A]
        self.b_orig = [Fraction(v) for v in b]
        self.signs_orig = list(signs)

        # сюда сложим имена переменных
        self.var_names: List[str] = []

        self._build_canonical()

    # Приведение к каноническому виду
    def _build_canonical(self):
        """Преобразует задачу к виду: A x = b, x >= 0, W -> min"""
        A = []
        b = []
        signs = []

        # копируем строки, исправляя знак b < 0
        for i in range(self.m):
            row = list(self.A_orig[i])
            bi = self.b_orig[i]
            sign = self.signs_orig[i]
            if bi < 0:
                row = [-v for v in row]
                bi = -bi
                sign = {'<=': '>=', '>=': '<=', '=': '='}[sign]
            A.append(row)
            b.append(bi)
            signs.append(sign)

        # дополнительные переменные для неравенств
        extra_count = sum(1 for s in signs if s in ('<=', '>='))
        self.n_extra = extra_count

        for i in range(self.m):
            row = A[i] + [Fraction(0)] * extra_count
            if signs[i] == '<=':
                k = sum(1 for s in signs[:i] if s in ('<=', '>='))
                row[self.n + k] = Fraction(1)
            elif signs[i] == '>=':
                k = sum(1 for s in signs[:i] if s in ('<=', '>='))
                row[self.n + k] = Fraction(-1)
            A[i] = row

        self.A = A
        self.b = b
        self.signs = signs
        self.c = self.c_orig + [Fraction(0)] * extra_count

        # имена переменных
        self.var_names = [f"x{i+1}" for i in range(self.n)]
        idx = self.n
        for s in signs:
            if s in ('<=', '>='):
                idx += 1
                self.var_names.append(f"x{idx}")

        self.total_vars = len(self.c)

        # ищем начальный базис
        self.basis = self._find_initial_basis()

    def _find_initial_basis(self):
        """Ищет единичные столбцы для начального базиса"""
        basis = [None] * self.m
        used_rows = set()

        for j in range(self.total_vars):
            col = [self.A[i][j] for i in range(self.m)]
            ones = [i for i, v in enumerate(col) if v == 1]
            others_zero = all(v == 0 or v == 1 for v in col)
            if len(ones) == 1 and others_zero:
                i = ones[0]
                if i not in used_rows and basis[i] is None:
                    basis[i] = j
                    used_rows.add(i)

        return basis

    # Вспомогательная задача
    def _build_auxiliary(self):
        """Добавляет искусственные переменные туда, где нет базиса"""
        art_indices = []

        for i in range(self.m):
            if self.basis[i] is None:
                # добавляем искусственную переменную
                for k in range(self.m):
                    self.A[k].append(Fraction(1) if k == i else Fraction(0))
                self.c.append(Fraction(0))
                self.total_vars += 1
                self.var_names.append(f"a{len(art_indices) + 1}")
                self.basis[i] = self.total_vars - 1
                art_indices.append(self.basis[i])

        # целевая функция вспомогательной задачи: W' = сумма искусственных -> min
        aux_c = [Fraction(0)] * self.total_vars
        for j in art_indices:
            aux_c[j] = Fraction(1)

        # выражаем W' через свободные переменные
        aux_b = Fraction(0)
        for i in range(self.m):
            if self.basis[i] in art_indices:
                for j in range(self.total_vars):
                    aux_c[j] -= self.A[i][j]
                aux_b -= self.b[i]

        return aux_c, aux_b, art_indices

    # Симплекс-итерации
    def _simplex(self, c, b_offset, basis, A, b, forbid=None):
        """
        Выполняет симплекс-итерации
        c        - строка целевой функции
        b_offset - свободный член целевой функции
        basis    - текущий базис
        A, b     - текущая таблица
        forbid   - столбцы, которые нельзя вводить в базис
        Возвращает (status, A, b, c, b_offset, basis)
        """
        if forbid is None:
            forbid = set()

        A = [row[:] for row in A]
        b = b[:]
        c = c[:]
        basis = list(basis)

        max_iter = 1000
        for _ in range(max_iter):
            # разрешающий столбец
            j_star = None
            min_val = Fraction(0)
            for j in range(self.total_vars):
                if j in forbid:
                    continue
                if c[j] < min_val:
                    min_val = c[j]
                    j_star = j

            if j_star is None:
                return 'optimal', A, b, c, b_offset, basis

            # разрешающая строка
            i_star = None
            best_ratio = None
            for i in range(self.m):
                if A[i][j_star] > 0:
                    ratio = b[i] / A[i][j_star]
                    if best_ratio is None or ratio < best_ratio:
                        best_ratio = ratio
                        i_star = i

            if i_star is None:
                return 'unbounded', A, b, c, b_offset, basis

            # пересчёт
            pivot = A[i_star][j_star]
            A[i_star] = [v / pivot for v in A[i_star]]
            b[i_star] = b[i_star] / pivot

            for i in range(self.m):
                if i != i_star and A[i][j_star] != 0:
                    factor = A[i][j_star]
                    A[i] = [A[i][k] - factor * A[i_star][k] for k in range(self.total_vars)]
                    b[i] = b[i] - factor * b[i_star]

            factor = c[j_star]
            if factor != 0:
                c = [c[k] - factor * A[i_star][k] for k in range(self.total_vars)]
                b_offset = b_offset - factor * b[i_star]

            basis[i_star] = j_star

        return 'iteration_limit', A, b, c, b_offset, basis

    # Главный метод решения
    def solve(self):
        """Решает задачу, возвращает (status, x, W)"""
        A = [row[:] for row in self.A]
        b = self.b[:]
        basis = list(self.basis)

        # если базис не полный - решаем вспомогательную задачу
        if any(v is None for v in basis):
            aux_c, aux_b, art_indices = self._build_auxiliary()
            # после _build_auxiliary обновляются self.A, self.c, self.basis
            A = [row[:] for row in self.A]
            b = self.b[:]
            basis = list(self.basis)

            status, A, b, c_aux, b_aux, basis = self._simplex(
                aux_c, aux_b, basis, A, b
            )

            if status != 'optimal':
                return status, None, None

            # если W' > 0 - допустимого решения нет
            if b_aux != 0:
                return 'infeasible', None, None

            # убираем искусственные переменные из базиса
            for i in range(self.m):
                if basis[i] in art_indices:
                    for j in range(self.total_vars):
                        if j not in art_indices and A[i][j] != 0:
                            pivot = A[i][j]
                            A[i] = [v / pivot for v in A[i]]
                            b[i] = b[i] / pivot
                            for k in range(self.m):
                                if k != i and A[k][j] != 0:
                                    f = A[k][j]
                                    A[k] = [A[k][t] - f * A[i][t] for t in range(self.total_vars)]
                                    b[k] = b[k] - f * b[i]
                            basis[i] = j
                            break

            forbid = set(art_indices)
        else:
            forbid = set()

        # приводим основную целевую функцию к текущему базису
        c = self.c[:]
        b_offset = Fraction(0)
        for i in range(self.m):
            if basis[i] is not None and c[basis[i]] != 0:
                factor = c[basis[i]]
                c = [c[k] - factor * A[i][k] for k in range(self.total_vars)]
                b_offset = b_offset - factor * b[i]

        status, A, b, c, b_offset, basis = self._simplex(
            c, b_offset, basis, A, b, forbid=forbid
        )

        if status != 'optimal':
            return status, None, None

        x = [Fraction(0)] * self.total_vars
        for i in range(self.m):
            x[basis[i]] = b[i]

        return 'optimal', x, b_offset


# Решение нашей задачи (вариант 13)
if __name__ == "__main__":
    # Z = 3x1 + x2 + 2x3 + 4x4 -> max
    # 2x1 + x2 + x3 <= 8
    # x1 + x3 + x4 = 6
    # x2 + x4 >= 4

    c = [3, 1, 2, 4]
    A = [
        [2, 1, 1, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
    ]
    b = [8, 6, 4]
    signs = ['<=', '=', '>=']

    solver = SimplexSolver(c, A, b, signs, maximize=True)
    status, x, W = solver.solve()

    if status == 'optimal':
        print("Найдено оптимальное решение; статус:", status)
        print("Переменные:")
        for name, val in zip(solver.var_names, x):
            if val != 0:
                print(f"  {name} = {val}")
        print("Итоговое значение функции: ", W)