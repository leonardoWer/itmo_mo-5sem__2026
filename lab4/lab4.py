"""Управление инвестиционным портфелем, лабораторная работа 4"""

from dataclasses import dataclass
from decimal import Decimal
from functools import cache


@dataclass(frozen=True)
class Portfolio:
    initial: tuple
    lots: tuple
    minimums: tuple
    fees: tuple
    market: tuple


    def __post_init__(self):
        if (len(self.initial) != 4 or len(self.lots) != 3
                or len(self.minimums) != 3 or len(self.fees) != 3):
            raise ValueError("Нужны три актива и остаток свободных средств")
        if (any(x < 0 for x in self.initial + self.minimums)
                or any(x <= 0 for x in self.lots)
                or any(not 0 <= fee < 100 for fee in self.fees)):
            raise ValueError("Недопустимые остатки, размеры пакетов или комиссии")
        if not self.market or any(
            len(stage) != 3 or sum(p for p, _ in stage) != 10
            or any(p <= 0 or len(rates) != 3 or min(rates) <= 0 for p, rates in stage)
            for stage in self.market
        ):
            raise ValueError("У каждого этапа должны быть три сценария с корректными весами")


# Активы идут в порядке ЦБ1, ЦБ2, депозит
# Вес 6 соответствует вероятности 0,6
DATA = Portfolio(
    initial=(100, 800, 400, 600),
    lots=(25, 200, 100),
    minimums=(30, 150, 100),
    fees=(4, 7, 5),
    market=(
        ((6, (120, 110, 107)), (3, (105, 102, 103)), (1, (80, 95, 100))),
        ((3, (140, 115, 101)), (2, (105, 100, 100)), (5, (60, 90, 100))),
        ((4, (115, 112, 105)), (4, (105, 101, 101)), (2, (70, 94, 100))),
    ),
)
SCENARIOS = "Б", "Н", "П"


class Bellman:
    def __init__(self, data):
        self.data = data
        self.steps = len(data.market)
        # После каждого этапа масштаб растёт в 100 раз, чтобы хранить деньги целыми
        self.scales = tuple(100 ** (k + 1) for k in range(self.steps + 1))
        self.initial = tuple(x * self.scales[0] for x in data.initial)
        # Запоминаем лучший результат и сделки для каждой пары (этап, портфель)
        self.value = cache(self._value)
        self.weights = self.upper_weights()

    def upper_weights(self):
        """Верхняя оценка портфеля до сделок на каждом этапе"""
        weights = [None] * self.steps + [(1, 1, 1, 1)]
        for step in reversed(range(self.steps)):
            next_weights = weights[step + 1]
            if next_weights is None:
                break
            cash = next_weights[3] * 1000
            carry = [
                sum(p * rates[i] * next_weights[i] for p, rates in self.data.market[step])
                for i in range(3)
            ]
            if any(value * 100 > cash * (100 + fee)
                   for value, fee in zip(carry, self.data.fees)):
                break
            # Для каждого актива выбираем более высокую оценку: удержать или продать
            # При такой продаже можно игнорировать размер пакета и минимальный остаток
            assets = tuple(max(value, cash * (100 - fee) // 100)
                           for value, fee in zip(carry, self.data.fees))
            weights[step] = (*assets, cash)
        return weights

    def trades(self, step, state):
        """Перечисляет допустимые сделки и возвращает портфель сразу после каждой"""
        scale = self.scales[step]
        asset_rows = []
        # Отрицательная граница показывает, сколько пакетов можно продать
        # Положительная граница означает, что нужно докупить актив до минимума
        limits = [(minimum * scale - amount, lot * scale)
                  for minimum, amount, lot in zip(self.data.minimums, state, self.data.lots)]
        lower = [-(-gap // size) for gap, size in limits]
        sells = [
            max(0, -count) * size * (100 - fee) // 100
            for count, (_, size), fee in zip(lower, limits, self.data.fees)
        ]
        available = state[3] + sum(sells)
        for i, ((_, size), fee) in enumerate(zip(limits, self.data.fees)):
            buy_cost = size * (100 + fee) // 100
            sell_income = size * (100 - fee) // 100
            # На покупку можно потратить остаток денег и выручку от других активов
            upper = max(0, (available - sells[i]) // buy_cost)
            asset_rows.append([
                (count, state[i] + count * size,
                 -count * (buy_cost if count > 0 else sell_income))
                for count in range(lower[i], upper + 1)
            ])
        for n1, a1, cash1 in asset_rows[0]:
            for n2, a2, cash2 in asset_rows[1]:
                balance = state[3] + cash1 + cash2
                for n3, a3, cash3 in asset_rows[2]:
                    if balance + cash3 >= 0:
                        yield (n1, n2, n3), (a1, a2, a3, balance + cash3)

    def successors(self, step, after):
        # На рынке меняются стоимости активов, а деньги только переводятся в новый масштаб
        a1, a2, a3, cash = after
        for p, (g1, g2, g3) in self.data.market[step]:
            yield p, (a1 * g1, a2 * g2, a3 * g3, cash * 100)

    def continuation_weights(self, step):
        """Верхняя оценка после сделки с учётом всех сценариев этапа"""
        weights = self.weights[step + 1]
        if weights is None:
            return None
        return (
            *(sum(p * rates[i] * weights[i] for p, rates in self.data.market[step])
              for i in range(3)),
            1000 * weights[3],
        )

    def last_trade(self, step, state, weights):
        # Если любая сделка снижает итог, покупаем только до обязательного минимума
        # Если денег на это не хватает, перебираем варианты с продажами
        if any(not weights[3] * (100 - fee) < 100 * w < weights[3] * (100 + fee)
               for w, fee in zip(weights, self.data.fees)):
            return None
        scale = self.scales[step]
        lots = tuple(lot * scale for lot in self.data.lots)
        action = tuple(max(0, -((amount - minimum * scale) // lot))
                       for amount, minimum, lot in zip(state, self.data.minimums, lots))
        buys = tuple(count * lot for count, lot in zip(action, lots))
        cash = state[3] - sum(buy * (100 + fee) // 100
                              for buy, fee in zip(buys, self.data.fees))
        if cash >= 0:
            after = (*(amount + buy for amount, buy in zip(state, buys)), cash)
            return score(weights, after), (action,)
        return None

    def _value(self, step, state):
        if step == self.steps:
            # Этапы закончились: складываем стоимости активов и свободные деньги
            return sum(state), ()
        weights = self.continuation_weights(step)
        if step == self.steps - 1:
            direct = self.last_trade(step, state, weights)
            if direct is not None:
                return direct
        if weights is None:
            candidates = ((None, action, after) for action, after in self.trades(step, state))
        else:
            candidates = sorted(
                ((score(weights, after), action, after)
                 for action, after in self.trades(step, state)),
                reverse=True,
            )
        best = None
        actions = []
        for upper, action, after in candidates:
            # Дальше оценки только ниже, поэтому поиск можно закончить
            # При равенстве проверяем сделку: она тоже может оказаться оптимальной
            if upper is not None and best is not None and upper < best:
                break
            if step == self.steps - 1:
                result = upper
            else:
                result = self.expected_value(step, after)
            if result is None:
                continue
            if best is None or result > best:
                best, actions = result, [action]
            elif result == best:
                actions.append(action)
        return best, tuple(sorted(actions))

    def expected_value(self, step, after):
        # Для одной сделки складываем лучшие продолжения с весами трёх сценариев
        result = 0
        for p, state in self.successors(step, after):
            value, _ = self.value(step + 1, state)
            if value is None:
                return None
            result += p * value
        return result

    def money(self, value, step):
        return Decimal(value) / (self.scales[-1] * 10 ** (self.steps - step))

    def strategy(self, step=0, state=None, history="-", probability=1):
        if state is None:
            state = self.initial
        value, actions = self.value(step, state)
        if value is None:
            raise ValueError("Допустимую стратегию построить нельзя")
        if step == self.steps:
            yield step, history, probability, state, (), state, value
            return
        # Берём первую лучшую сделку и заново проверяем её среди допустимых
        # От портфеля после сделки строим отдельное продолжение для каждого сценария
        after = next(after for action, after in self.trades(step, state) if action == actions[0])
        yield step, history, probability, state, actions, after, value
        for name, (p, next_state) in zip(SCENARIOS, self.successors(step, after)):
            next_history = name if history == "-" else history + name
            yield from self.strategy(step + 1, next_state, next_history, probability * p)


def score(weights, state):
    a1, a2, a3, cash = state
    w1, w2, w3, w_cash = weights
    return w1 * a1 + w2 * a2 + w3 * a3 + w_cash * cash


def format_state(state, scale):
    return "(" + ", ".join(f"{Decimal(x) / scale:.6f}".rstrip("0").rstrip(".")
                           for x in state) + ")"


def main():
    model = Bellman(DATA)
    value, _ = model.value(0, model.initial)
    realized_value = 0
    probability_sum = 0
    print("История сценариев: Б - благоприятный, Н - нейтральный, П - негативный")
    print("Портфель: (ЦБ1, ЦБ2, депозит, свободные средства)")
    print("Сделки указаны в пакетах: плюс - покупка, минус - продажа\n")
    for step, history, probability, state, actions, after, result in model.strategy():
        scale = model.scales[step]
        if step < model.steps:
            print(f"Этап {step + 1}, {history}: {format_state(state, scale)} -> "
                  f"{format_state(after, scale)}, сделки {actions}, "
                  f"V = {model.money(result, step):.6f}")
        else:
            realized_value += probability * result
            probability_sum += probability
            print(f"Итог {history}: P = {Decimal(probability) / 10 ** step}, "
                  f"капитал = {model.money(result, step):.6f}")
    if probability_sum != 10 ** model.steps or realized_value != value:
        raise RuntimeError("Проверка прямого прохода не пройдена")
    print(f"\nОжидаемый капитал: {model.money(value, 0):.8f} д.е.")
    profit = model.money(value, 0) - sum(DATA.initial)
    print(f"Ожидаемый доход: {profit:.8f} д.е.")


if __name__ == "__main__":
    main()