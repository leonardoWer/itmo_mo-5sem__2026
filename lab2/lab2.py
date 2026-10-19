"""
Лабораторная работа №2. Метод ломаных (Пиявского)
Поиск глобального экстремума липшицевой функции на отрезке [a, b]
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

# ПАРСЕР ФУНКЦИИ ИЗ СТРОКИ
SAFE_NAMESPACE = {
    "x": None,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": np.abs, "pi": np.pi, "e": np.e,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
}
def make_function(expr: str) -> Callable[[float], float]:
    """Превращает строку в вектор функцию numpy"""
    code = compile(expr, "<func>", "eval")

    def f(x):
        env = dict(SAFE_NAMESPACE)
        env["x"] = x
        return eval(code, {"__builtins__": {}}, env)

    return f


# МЕТОД ЛОМАНЫХ
@dataclass
class Result:
    x_min: float
    f_min: float
    iterations: int
    elapsed: float
    history: List[Tuple[float, float]]
    L: float
    converged: bool  # достигнута ли точность eps (а не max_iter)


def estimate_lipschitz(f: Callable, a: float, b: float, n: int = 1000) -> float:
    """
    Оценка константы Липшица L = max |f'(x)| численно.
    Только для оценки L, не для поиска минимума.
    """
    xs = np.linspace(a, b, n)
    ys = f(xs)
    slopes = np.abs(np.diff(ys) / np.diff(xs))
    L = float(np.max(slopes))
    return L * 1.1 if L > 0 else 1.0


def _vertex(x1: float, y1: float, x2: float, y2: float, L: float) -> float:
    """
    Точка пересечения двух конусов (вершина ломаной) между испытаниями
    (x1, y1) и (x2, y2), x1 < x2.
    x = (y1 - y2 + L*(x1 + x2)) / (2L)
    """
    return (y1 - y2 + L * (x1 + x2)) / (2.0 * L)


def _broken_value_at(xv: float, points: List[Tuple[float, float]],
                     L: float) -> float:
    """
    Значение ломаной phi(x) = max_i (y_i - L*|xv - x_i|) в точке xv.
    Считаем максимум по ВСЕМ испытаниям, а не только по соседям —
    это защищает от ошибок, если в вершине активен "чужой" конус.
    """
    best = -np.inf
    for xi, yi in points:
        v = yi - L * abs(xv - xi)
        if v > best:
            best = v
    return best


def broken_lines(f: Callable, a: float, b: float, eps: float,
                 L: float | None = None,
                 maximize: bool = False,
                 max_iter: int = 100_000) -> Result:
    """
    Метод ломаных (Пиявского).
    """
    t0 = time.perf_counter()

    if maximize:
        g = lambda x: -f(x)
    else:
        g = f

    if L is None:
        L = estimate_lipschitz(g, a, b)

    # Начальные испытания: концы отрезка
    points = [(a, float(g(a))), (b, float(g(b)))]

    converged = False
    for it in range(1, max_iter + 1):
        points.sort(key=lambda p: p[0])

        # Шаг 1: для каждого интервала между соседними испытаниями
        # находим вершину ломаной и её высоту (значение ломаной там).
        best_R = -np.inf
        best_x = None

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            if x2 - x1 < 1e-15:
                continue
            xv = _vertex(x1, y1, x2, y2, L)
            # Защита: вершина должна лежать внутри интервала
            if xv <= x1 or xv >= x2:
                # конусы не пересекаются внутри — берём середину
                xv = 0.5 * (x1 + x2)
            R = _broken_value_at(xv, points, L)
            if R > best_R:
                best_R = R
                best_x = xv

        # Шаг 2: критерий остановки.
        # f_best — лучшая верхняя оценка минимума (лучшее испытание).
        # best_R — лучшая нижняя оценка (максимум ломаной, ломаная под графиком).
        # Истинный минимум f* лежит в [best_R, f_best].
        # Если f_best - best_R <= eps, значит f_best - f* <= eps — точность достигнута.
        f_best = min(y for _, y in points)
        if best_x is None or (f_best - best_R) <= eps:
            converged = (best_x is not None) and ((f_best - best_R) <= eps)
            break

        # Шаг 3: новое испытание в вершине ломаной.
        # Защита от зацикливания: если точка слишком близка к уже имеющейся,
        # сдвигаем в середину текущего "лучшего" интервала.
        if any(abs(best_x - px) < 1e-12 for px, _ in points):
            # ищем интервал, где достигается best_R, и берём его середину
            for i in range(len(points) - 1):
                x1, _ = points[i]
                x2, _ = points[i + 1]
                if x1 <= best_x <= x2 and (x2 - x1) > 1e-12:
                    best_x = 0.5 * (x1 + x2)
                    break

        y_new = float(g(best_x))
        points.append((best_x, y_new))

    elapsed = time.perf_counter() - t0

    best_x, best_y = min(points, key=lambda p: p[1])
    f_val = -best_y if maximize else best_y

    return Result(
        x_min=best_x,
        f_min=f_val,
        iterations=len(points) - 2,
        elapsed=elapsed,
        history=points,
        L=L,
        converged=converged,
    )


# ВИЗУАЛИЗАЦИЯ
def plot_result(f: Callable, a: float, b: float, res: Result,
                title: str = "", maximize: bool = False):
    """Строит график функции, ломаную и найденную точку"""
    if maximize:
        g_plot = lambda x: -f(x)
    else:
        g_plot = f

    L = res.L
    pts = sorted(res.history, key=lambda p: p[0])
    xs_hist = np.array([p[0] for p in pts])
    ys_hist = np.array([p[1] for p in pts])

    xs_plot = np.linspace(a, b, 800)
    lower = np.full_like(xs_plot, -np.inf)
    for xi, yi in pts:
        cone = yi - L * np.abs(xs_plot - xi)
        lower = np.maximum(lower, cone)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs_plot, g_plot(xs_plot), 'b-', label='f(x)', linewidth=1.8)
    ax.plot(xs_plot, lower, 'r--', label='ломаная φ(x)', linewidth=1.2)
    ax.plot(xs_hist, ys_hist, 'ko', markersize=4, label='испытания')

    ax.plot(res.x_min, res.f_min, 'g*', markersize=16,
            label=f'экстремум ({res.x_min:.6f}, {res.f_min:.6f})')
    ax.axvline(res.x_min, color='g', linestyle=':', alpha=0.6)

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    status = "сошлось" if res.converged else "НЕ сошлось (max_iter)"
    ax.set_title((title or f'Метод ломаных: x*={res.x_min:.6f}, '
                          f'f*={res.f_min:.6f}, итераций={res.iterations}')
                 + f" [{status}]")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ТЕСТОВЫЕ ФУНКЦИИ
def test1_sin(x):
    """f(x) = x + sin(pi*x) на [-2, 2] (два локальных минимума)"""
    return x + np.sin(np.pi * x)


def test2_rastrigin_1d(x):
    """1D Растригин: f(x) = x^2 - 10*cos(2*pi*x) + 10"""
    return x ** 2 - 10 * np.cos(2 * np.pi * x) + 10


def test3_ackley_1d(x):
    """1D Экли (упрощённая)"""
    return (-20 * np.exp(-0.2 * np.abs(x))
            - np.exp(np.cos(2 * np.pi * x)) + np.e + 20)


def test4_multimodal(x):
    """f(x) = x*sin(x) + x^2/20 на [-10, 10]"""
    return x * np.sin(x) + x ** 2 / 20.0


def run_demo(name: str, f: Callable, a: float, b: float,
             eps: float, maximize: bool = False):
    print("\n")
    print(f"  {name}")
    print(f"  отрезок [{a}, {b}], eps = {eps}, "
          f"{'Максимум' if maximize else 'Минимум'}")
    print('=' * 60)

    res = broken_lines(f, a, b, eps, maximize=maximize)
    print(f"  L (оценка)        : {res.L:.4f}")
    print(f"  x*                : {res.x_min:.8f}")
    print(f"  f(x*)             : {res.f_min:.8f}")
    print(f"  итераций          : {res.iterations}")
    print(f"  сошлось           : {res.converged}")
    print(f"  время             : {res.elapsed * 1000:.2f} мс")

    plot_result(f, a, b, res,
                title=f"{name} | x*={res.x_min:.6f}, f*={res.f_min:.6f}",
                maximize=maximize)


if __name__ == "__main__":
    run_demo("f(x) = x + sin(πx)", test1_sin, -2, 2, 1e-3)
    run_demo("1D Растригин", test2_rastrigin_1d, -5.12, 5.12, 1e-3)
    run_demo("1D Экли", test3_ackley_1d, -2, 2, 1e-3)
    run_demo("f(x) = x·sin(x) + x²/20", test4_multimodal, -10, 10, 1e-3)
    run_demo("f(x) = x·sin(x) + x²/20 (MAX)", test4_multimodal, -10, 10, 1e-3,
             maximize=True)

    user_f = make_function("x + sin(3.14159*x)")
    run_demo("Ввод с консоли: f(x) = x + sin(3.14159x)",
             user_f, -2, 2, 1e-3)