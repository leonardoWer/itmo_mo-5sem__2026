"""
Лабораторная № 3. Метод множителей Лагранжа, вариант 13
"""

from dataclasses import dataclass

import sympy as sp


EPS = 1e-8
x, y = sp.symbols("x y", real=True)
lam = sp.Symbol("lambda", real=True)
VARIABLES = (x, y)
MULTIPLIERS = (lam,)
F = (x - 1) ** 2 + (y + 2) ** 2
CONSTRAINTS = (x**2 + y**2 - 1,)


@dataclass(frozen=True)
class Candidate:
    point: tuple
    multipliers: tuple
    value: sp.Expr
    residuals: tuple
    system_residuals: tuple
    jacobian_rank: int
    feasible: bool


def is_zero(value):
    """
    Проверяем. Допуск нужен только для численных остатков
    """
    value = sp.simplify(value)
    if value == 0:
        return True
    if value.is_number and value.is_real is True and value.is_finite is True:
        return abs(float(sp.N(value, 30))) <= EPS
    return False


def solve_problem():
    lagrangian = F + sum(a * h for a, h in zip(MULTIPLIERS, CONSTRAINTS))
    equations = tuple(sp.diff(lagrangian, v) for v in VARIABLES) + CONSTRAINTS
    unknowns = VARIABLES + MULTIPLIERS
    jacobian = sp.Matrix(CONSTRAINTS).jacobian(VARIABLES)

    # При одном ограничении особые точки ищем из h = 0 и grad(h) = 0
    singular_points = sp.solve(
        list(CONSTRAINTS) + list(jacobian), VARIABLES, dict=True
    )
    if singular_points:
        raise RuntimeError("Найдены особые точки: требуется отдельное исследование.")

    solutions = sp.solve(equations, unknowns, dict=True)
    if not solutions:
        raise RuntimeError("Решатель не вернул кандидатов; нужен ручной анализ.")

    candidates = []
    for solution in solutions:
        if any(v not in solution for v in unknowns):
            raise RuntimeError("Получено неполное решение или семейство решений.")
        values = tuple(sp.simplify(solution[v]) for v in unknowns)
        if any(v.free_symbols for v in values):
            raise RuntimeError("Получено семейство: нельзя сравнивать лишь его представителей.")
        if any(v.is_real is False for v in values):
            continue
        if any(v.is_real is not True or v.is_finite is not True for v in values):
            raise RuntimeError("Не удалось подтвердить действительность и конечность решения.")

        point = values[:len(VARIABLES)]
        point_subs = dict(zip(VARIABLES, point))
        residuals = tuple(sp.simplify(h.subs(point_subs)) for h in CONSTRAINTS)
        system_residuals = tuple(sp.simplify(e.subs(solution)) for e in equations)
        feasible = all(is_zero(r) for r in residuals)
        candidate = Candidate(
            point=point,
            multipliers=values[len(VARIABLES):],
            value=sp.simplify(F.subs(point_subs)),
            residuals=residuals,
            system_residuals=system_residuals,
            jacobian_rank=jacobian.subs(point_subs).rank(),
            feasible=feasible,
        )
        candidates.append(candidate)
        if not feasible or not all(is_zero(r) for r in system_residuals):
            raise RuntimeError(f"Решение не прошло подстановку: {solution}")
        if candidate.jacobian_rank != len(CONSTRAINTS):
            raise RuntimeError("Кандидат нерегулярен: требуется отдельная классификация.")

    if not candidates:
        raise RuntimeError("Не найдены действительные кандидаты.")
    candidates.sort(key=lambda c: float(sp.N(c.value, 30)))
    return lagrangian, equations, jacobian, candidates


def main():
    lagrangian, equations, jacobian, candidates = solve_problem()
    print("Лабораторная работа № 3. Вариант 13 (13 mod 20 = 13)")
    print("Целевая функция:", F)
    print("Ограничения:", tuple(sp.Eq(h, 0) for h in CONSTRAINTS))
    print("Область определения: R^2; допустимое множество: единичная окружность.")
    print("Якобиан ограничений:", jacobian)
    gradient_norm_squared = sum(e**2 for e in jacobian)
    
    # На h = 0 это выражение равно квадрату нормы градиента
    print("Квадрат нормы grad(h) на окружности:",
          sp.simplify(gradient_norm_squared - 4 * CONSTRAINTS[0]))
    print("Особых допустимых точек нет; ранг Якобиана на окружности равен 1.")
    print("Функция Лагранжа:", lagrangian)
    print("Уравнения стационарности и ограничение:")
    for equation in equations:
        print(" ", sp.Eq(equation, 0))

    for number, candidate in enumerate(candidates, start=1):
        print(f"\nКандидат {number}:")
        print("  точка:", candidate.point)
        print("  численно:", tuple(float(sp.N(v, 12)) for v in candidate.point))
        print("  множители:", candidate.multipliers)
        print("  значение функции:", candidate.value)
        print(f"  численное значение: {float(sp.N(candidate.value, 30)):.10f}")
        print("  невязки ограничений:", candidate.residuals)
        print("  невязки всей системы:", candidate.system_residuals)
        print("  ранг Якобиана:", candidate.jacobian_rank)
        print("  допустимость:", "да" if candidate.feasible else "нет")

    print("\nИтог:")
    for label, candidate in (("минимум", candidates[0]), ("максимум", candidates[-1])):
        print(f"  глобальный {label}: {candidate.value}, точка {candidate.point}")
    print("  Основание: окружность компактна, f непрерывна, все точки регулярны")
    print("  Полнота списка доказана в отчёте: lambda != -1 и (1+lambda)^2 = 5")
    print("  Поэтому достаточно сравнить значения во всех найденных кандидатах")
    print("  EPS =", EPS, "; здесь все невязки равны нулю точно")


if __name__ == "__main__":
    main()
