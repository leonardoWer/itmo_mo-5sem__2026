"""
Лабораторная работа №2. Метод ломаных (Пиявского)
Поиск глобального экстремума липшицевой функции на отрезке [a, b]
"""

import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IMGS_DIR = "imgs"

SAFE_NAMESPACE = {
    "x": None,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": np.abs, "pi": np.pi, "e": np.e,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
}


def make_function(expr: str) -> Callable[[float], float]:
    """Превращает строку в функцию от x."""
    code = compile(expr, "<func>", "eval")

    def f(x):
        env = dict(SAFE_NAMESPACE)
        env["x"] = x
        return eval(code, {"__builtins__": {}}, env)

    return f


@dataclass
class Result:
    x_min: float
    f_min: float
    iterations: int
    elapsed: float
    history: List[Tuple[float, float]]
    L: float
    converged: bool
    gap: float
    stop_reason: str


def estimate_lipschitz(f: Callable, a: float, b: float, n: int = 1000) -> float:
    """Численная оценка константы Липшица по максимальному наклону"""
    xs = np.linspace(a, b, n)
    ys = f(xs)
    # постоянная функция возвращает скаляр, приводим к массиву,
    # иначе np.diff падает на 0-мерном объекте
    ys = np.broadcast_to(np.asarray(ys, dtype=float), xs.shape).copy()
    if not np.all(np.isfinite(ys)):
        raise ValueError(
            "Функция содержит NaN или Inf на отрезке. "
            "Проверьте область определения."
        )
    slopes = np.abs(np.diff(ys) / np.diff(xs))
    L = float(np.max(slopes)) if slopes.size else 0.0
    # для константы L = 0, а методу нужна положительная константа
    if not np.isfinite(L) or L <= 0:
        return 1.0
    return L * 1.1


def is_constant_on(f: Callable, a: float, b: float, n: int = 200) -> bool:
    """Проверяет, что функция постоянна на отрезке"""
    xs = np.linspace(a, b, n)
    ys = f(xs)
    ys = np.broadcast_to(np.asarray(ys, dtype=float), xs.shape).copy()
    if not np.all(np.isfinite(ys)):
        raise ValueError(
            "Функция содержит NaN или Inf на отрезке. "
            "Проверьте область определения."
        )
    return bool(np.max(ys) - np.min(ys) <= 1e-12)


def _vertex(x1: float, y1: float, x2: float, y2: float, L: float) -> float:
    """Точка пересечения двух конусов между соседними испытаниями."""
    return (y1 - y2 + L * (x1 + x2)) / (2.0 * L)


def _broken_value_at(xv: float, points: List[Tuple[float, float]],
                     L: float) -> float:
    """Значение ломаной phi(x) = max_i (y_i - L*|x - x_i|) в точке xv."""
    best = -np.inf
    for xi, yi in points:
        v = yi - L * abs(xv - xi)
        if v > best:
            best = v
    return best


def broken_lines(f: Callable, a: float, b: float, eps: float,
                 L: Optional[float] = None,
                 maximize: bool = False,
                 max_iter: int = 100_000) -> Result:
    """
    Метод ломаных Пиявского
    Ищем интервал с минимальной характеристикой,
    останавливаемся когда разность между лучшим испытанием и минимумом
    ломаной не больше eps
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("Границы отрезка должны быть конечными числами.")
    if a >= b:
        raise ValueError("Требуется a < b.")
    if not (np.isfinite(eps) and eps > 0):
        raise ValueError("Точность eps должна быть положительным числом.")
    if L is not None and (not np.isfinite(L) or L <= 0):
        raise ValueError("Константа Липшица L должна быть положительной.")

    t0 = time.perf_counter()

    g = (lambda x: -f(x)) if maximize else f

    if L is None:
        L = estimate_lipschitz(g, a, b)

    if is_constant_on(g, a, b):
        ya = float(g(a))
        return Result(
            x_min=a,
            f_min=(-ya if maximize else ya),
            iterations=0,
            elapsed=time.perf_counter() - t0,
            history=[(a, ya), (b, float(g(b)))],
            L=L,
            converged=True,
            gap=0.0,
            stop_reason="функция постоянна на отрезке",
        )

    ya = float(g(a))
    yb = float(g(b))
    if not (np.isfinite(ya) and np.isfinite(yb)):
        raise ValueError(
            "Функция содержит NaN или Inf на границах отрезка. "
            "Проверьте область определения."
        )
    points = [(a, ya), (b, yb)]

    converged = False
    stop_reason = "достигнут max_iter"
    gap = np.inf

    for _ in range(1, max_iter + 1):
        points.sort(key=lambda p: p[0])

        # ищем интервал с минимальной характеристикой
        best_R = np.inf
        best_x = None
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            if x2 - x1 < 1e-15:
                continue
            xv = _vertex(x1, y1, x2, y2, L)
            # вершина должна лежать внутри интервала
            if xv <= x1 or xv >= x2:
                xv = 0.5 * (x1 + x2)
            R = _broken_value_at(xv, points, L)
            if R < best_R:
                best_R = R
                best_x = xv

        # критерий остановки: истинный минимум лежит в [best_R, f_best]
        f_best = min(y for _, y in points)
        gap = f_best - best_R
        if best_x is None or gap <= eps:
            converged = (best_x is not None) and (gap <= eps)
            stop_reason = ("достигнута точность eps"
                           if converged else "не удалось построить вершину")
            break

        # защита от зацикливания: если точка почти совпала с имеющейся,
        # сдвигаем в середину текущего интервала
        if any(abs(best_x - px) < 1e-12 for px, _ in points):
            for i in range(len(points) - 1):
                x1, _ = points[i]
                x2, _ = points[i + 1]
                if x1 <= best_x <= x2 and (x2 - x1) > 1e-12:
                    best_x = 0.5 * (x1 + x2)
                    break

        y_new = float(g(best_x))
        if not np.isfinite(y_new):
            raise ValueError(
                f"Функция вернула NaN или Inf в точке x {best_x}. "
                "Проверьте область определения."
            )
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
        gap=gap,
        stop_reason=stop_reason,
    )


def _safe_filename(name: str) -> str:
    """Убирает из имени файла символы, недопустимые в путях."""
    keep = "-_."
    return "".join(c if c.isalnum() or c in keep else "_" for c in name)


def plot_result(f: Callable, a: float, b: float, res: Result,
                title: str = "", maximize: bool = False,
                filename: str = "plot") -> str:
    """Строит график функции, ломаную и найденную точку, сохраняет в imgs/."""
    os.makedirs(IMGS_DIR, exist_ok=True)

    g_plot = (lambda x: -f(x)) if maximize else f

    L = res.L
    pts = sorted(res.history, key=lambda p: p[0])
    xs_hist = np.array([p[0] for p in pts])
    ys_hist = np.array([p[1] for p in pts])

    xs_plot = np.linspace(a, b, 800)
    # постоянная функция возвращает скаляр, приводим к массиву,
    # иначе matplotlib ругается на размерности
    ys_plot = np.broadcast_to(
        np.asarray(g_plot(xs_plot), dtype=float), xs_plot.shape
    )

    lower = np.full_like(xs_plot, -np.inf)
    for xi, yi in pts:
        cone = yi - L * np.abs(xs_plot - xi)
        lower = np.maximum(lower, cone)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs_plot, ys_plot, 'b-', label='f(x)', linewidth=1.8)
    ax.plot(xs_plot, lower, 'r--', label='ломаная φ(x)', linewidth=1.2)
    ax.plot(xs_hist, ys_hist, 'ko', markersize=4, label='испытания')

    # при максимизации график рисуется в координатах -f,
    # поэтому звезду ставим по -f_min чтобы знаки совпали
    y_star_plot = -res.f_min if maximize else res.f_min
    ax.plot(res.x_min, y_star_plot, 'g*', markersize=16,
            label=f'экстремум ({res.x_min:.6f}, {res.f_min:.6f})')
    ax.axvline(res.x_min, color='g', linestyle=':', alpha=0.6)

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    status = "сошлось" if res.converged else f"НЕ сошлось: {res.stop_reason}"
    ax.set_title((title or f'Метод ломаных: x* {res.x_min:.6f}, '
                           f'f* {res.f_min:.6f}, итераций {res.iterations}')
                 + f" [{status}]")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(IMGS_DIR, _safe_filename(filename) + ".png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def test1_sin(x):
    return x + np.sin(np.pi * x)


def test2_rastrigin_1d(x):
    return x ** 2 - 10 * np.cos(2 * np.pi * x) + 10


def test3_ackley_1d(x):
    return (-20 * np.exp(-0.2 * np.abs(x))
            - np.exp(np.cos(2 * np.pi * x)) + np.e + 20)


def test4_multimodal(x):
    return x * np.sin(x) + x ** 2 / 20.0


def run_demo(name: str, f: Callable, a: float, b: float,
             eps: float, maximize: bool = False,
             filename: str = "plot"):
    """Запускает метод, печатает отчёт и сохраняет график."""
    print("\n")
    print(f"  {name}")
    print(f"  отрезок [{a}, {b}], eps {eps}, "
          f"{'Максимум' if maximize else 'Минимум'}")
    print('-' * 60)

    try:
        res = broken_lines(f, a, b, eps, maximize=maximize)
    except ValueError as e:
        print(f"  ОШИБКА: {e}")
        return

    print(f"  L (оценка)         : {res.L:.4f}")
    print(f"  x*                 : {res.x_min:.8f}")
    print(f"  f(x*)              : {res.f_min:.8f}")
    print(f"  итераций           : {res.iterations}")
    print(f"  сошлось            : {res.converged}")
    print(f"  разрыв f_best-φ    : {res.gap:.3e}")
    print(f"  причина остановки  : {res.stop_reason}")
    print(f"  время              : {res.elapsed * 1000:.2f} мс")

    try:
        path = plot_result(
            f, a, b, res,
            title=f"{name} | x* {res.x_min:.6f}, f* {res.f_min:.6f}",
            maximize=maximize,
            filename=filename,
        )
        print(f"  график сохранён    : {path}")
    except Exception as e:
        print(f"  график не построен : {e}")


def interactive_input():
    """Спрашивает у пользователя функцию, границы, точность и режим."""
    print("\n=== Ручной ввод ===")
    expr = input("Введите функцию от x (например, x + sin(3.14159*x)): ").strip()
    if not expr:
        print("Пустая функция, пропускаем.")
        return
    try:
        f = make_function(expr)
    except Exception as e:
        print(f"Не удалось разобрать функцию: {e}")
        return

    try:
        a = float(input("Введите левую границу a: ").strip())
        b = float(input("Введите правую границу b: ").strip())
        eps = float(input("Введите точность eps: ").strip())
    except ValueError:
        print("Границы и точность должны быть числами.")
        return

    mode = input("Искать максимум? (y/N): ").strip().lower() == 'y'
    run_demo(f"Пользовательская функция: {expr}", f, a, b, eps,
             maximize=mode, filename="user_input")


if __name__ == "__main__":
    run_demo("1D Растригин [-1.5, 2] (проверка замечания)",
             test2_rastrigin_1d, -1.5, 2.0, 1e-3,
             filename="rastrigin_check")

    run_demo("f(x) = x + sin(πx)", test1_sin, -2, 2, 1e-3,
             filename="sin_pi")

    run_demo("1D Растригин", test2_rastrigin_1d, -5.12, 5.12, 1e-3,
             filename="rastrigin")

    run_demo("1D Экли", test3_ackley_1d, -2, 2, 1e-3,
             filename="ackley")

    run_demo("f(x) = x·sin(x) + x²/20", test4_multimodal, -10, 10, 1e-3,
             filename="multimodal_min")

    run_demo("f(x) = x·sin(x) + x²/20 (MAX)", test4_multimodal, -10, 10, 1e-3,
             maximize=True, filename="multimodal_max")

    # проверка постоянной функции
    run_demo("Постоянная f(x) = 5", make_function("5"), -1, 1, 1e-3,
             filename="constant")

    user_f = make_function("x + sin(3.14159*x)")
    run_demo("Ввод с консоли: f(x) = x + sin(3.14159x)",
             user_f, -2, 2, 1e-3, filename="console_example")

    # interactive_input()
