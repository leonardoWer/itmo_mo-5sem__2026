"""Проверки расчёта Беллмана и независимый перебор стратегий"""

import sys
import unittest
from collections import Counter
from dataclasses import replace
from decimal import Decimal
from itertools import product
from math import ceil, floor

from lab4 import Bellman, DATA, Portfolio

D = Decimal
SMALL = Portfolio(
    initial=(1, 0, 0, 0),
    lots=(1, 10, 10),
    minimums=(0, 0, 0),
    fees=(0, 0, 0),
    market=(
        ((5, (200, 100, 100)), (3, (100, 100, 100)), (2, (50, 100, 100))),
        ((3, (80, 100, 100)), (4, (70, 100, 100)), (3, (60, 100, 100))),
        ((3, (160, 100, 100)), (4, (140, 100, 100)), (3, (120, 100, 100))),
    ),
)
ZERO = replace(DATA, initial=(0, 0, 0, 0), minimums=(0, 0, 0))
ONE_STAGE = replace(DATA, market=DATA.market[-1:])
LOW_CASH = replace(ONE_STAGE, initial=(20, 800, 400, 0))
TIE = replace(
    ONE_STAGE,
    initial=(0, 0, 0, 26),
    minimums=(0, 0, 0),
    market=(((4, (104, 100, 100)), (4, (104, 100, 100)), (2, (104, 100, 100))),),
)


def grid_trades(data, state):
    """Строит общую сетку пакетов, затем проверяет каждую тройку напрямую"""
    capital = sum(state)
    ranges = (
        range(ceil((minimum - amount) / D(lot)), floor((capital - amount) / D(lot)) + 1)
        for amount, minimum, lot in zip(state, data.minimums, data.lots)
    )
    for action in product(*ranges):
        assets = list(state[:3])
        cash = state[3]
        for i, (count, lot, fee) in enumerate(zip(action, data.lots, data.fees)):
            amount = count * lot
            assets[i] += amount
            cash -= amount + D(abs(amount) * fee) / 100
        if cash >= 0:
            yield action, (*assets, cash)


def world_outcomes(stage, after):
    for weight, rates in stage:
        assets = (a * rate / 100 for a, rate in zip(after, rates))
        yield D(weight) / 10, (*assets, after[3])


def all_policies(data, step, state):
    """Перебирает полные деревья решений, сохраняя даже невыгодные продолжения"""
    if step == len(data.market):
        yield sum(state), ()
        return
    for action, after in grid_trades(data, state):
        outcomes = list(world_outcomes(data.market[step], after))
        branches = [all_policies(data, step + 1, next_state) for _, next_state in outcomes]
        # Для каждой истории рынка выбирается своё продолжение
        for combination in product(*branches):
            value = sum(p * result for (p, _), (result, _) in zip(outcomes, combination))
            yield value, (action, tuple(policy for _, policy in combination))


def optimal_last(data, state):
    candidates = list(all_policies(data, 0, tuple(map(D, state))))
    best = max(value for value, _ in candidates)
    return best, tuple(policy[0] for value, policy in candidates if value == best)


def recovered_policy(model):
    plans = {history: actions[0] for step, history, _, _, actions, _, _ in model.strategy()
             if step < model.steps}

    def build(history):
        if history not in plans:
            return ()
        return plans[history], tuple(build(history + scenario) for scenario in "БНП")

    return plans["-"], tuple(build(scenario) for scenario in "БНП")


class PortfolioTests(unittest.TestCase):
    def check_value(self, data, expected_value, expected_actions):
        model = Bellman(data)
        value, actions = model.value(0, model.initial)
        self.assertEqual((model.money(value, 0), actions), (D(expected_value), expected_actions))
        if model.steps == 1:
            self.assertEqual(optimal_last(data, data.initial), (D(expected_value), expected_actions))

    def test_zero_portfolio(self):
        self.check_value(ZERO, "0", ((0, 0, 0),))

    def test_one_stage(self):
        self.check_value(ONE_STAGE, "1943.6", ((0, 0, 0),))

    def test_sale_funds_required_purchase(self):
        self.check_value(LOW_CASH, "1254.1", ((1, 0, -1),))

    def test_equal_actions(self):
        self.check_value(TIE, "26", ((0, 0, 0), (1, 0, 0)))

    def test_negative_initial_value(self):
        with self.assertRaises(ValueError):
            replace(DATA, initial=(-1, 800, 400, 600))

    def test_last_stage_of_main_strategy(self):
        model = Bellman(DATA)
        states_checked = 0
        for step, history, _, state, actions, _, value in model.strategy():
            if step == model.steps - 1:
                with self.subTest(history=history):
                    state_in_money = tuple(D(x) / model.scales[step] for x in state)
                    self.assertEqual(optimal_last(ONE_STAGE, state_in_money),
                                     (model.money(value, step), actions))
                    states_checked += 1
        self.assertEqual(states_checked, 9)

    def test_all_adaptive_policies(self):
        model = Bellman(SMALL)
        value, _ = model.value(0, model.initial)
        self.assertIsNone(model.continuation_weights(0))
        last_weights = model.market_weights(2, (1, 1, 1, 1))
        self.assertIsNone(model.last_trade(2, (0, 0, 0, model.scales[2]), last_weights))
        policies = list(all_policies(SMALL, 0, tuple(map(D, SMALL.initial))))
        self.assertEqual(len(policies), 1116)
        best = max(final_value for final_value, _ in policies)
        optimal = [policy for final_value, policy in policies if final_value == best]
        self.assertEqual(best, D("1.918"))
        self.assertEqual(model.money(value, 0), best)
        self.assertEqual(optimal, [recovered_policy(model)])
        self.assertEqual(tuple(branch[0] for branch in optimal[0][1]),
                         ((-2, 0, 0), (-1, 0, 0), (0, 0, 0)))


class CountedBellman(Bellman):
    def __init__(self, data):
        self.visits = Counter()
        super().__init__(data)

    def _value(self, step, state):
        self.visits[step] += 1
        return super()._value(step, state)


def count_main_states():
    """Считает уникальные портфели перед сделками с отсечением и без него"""
    model = CountedBellman(DATA)
    model.value(0, model.initial)
    pruned = tuple(model.visits[step] for step in range(model.steps))
    second_stage = {state for _, after in model.trades(0, model.initial)
                    for _, state in model.successors(0, after)}
    # В основном варианте деньги целые в д.е.; храним остатки как биты
    # Для каждого набора стоимостей активов единичный бит означает один остаток
    third_stage = {}
    visits_before_deduplication = 0
    for state in second_stage:
        for _, after in model.trades(1, state):
            for _, following in model.successors(1, after):
                a1, a2, a3, cash = following
                cash_index, remainder = divmod(cash, model.scales[2])
                assert remainder == 0
                assets = (a1, a2, a3)
                third_stage[assets] = third_stage.get(assets, 0) | (1 << cash_index)
                visits_before_deduplication += 1
    unpruned = (1, len(second_stage), sum(bits.bit_count() for bits in third_stage.values()))
    print("Этапы 1, 2, 3. Уникальные состояния перед сделками")
    print("С отсечением:", pruned, "всего", sum(pruned))
    print("Без отсечения:", unpruned, "всего", sum(unpruned))
    print("Переходов на этап 3 до удаления повторов:", visits_before_deduplication)


if __name__ == "__main__":
    if sys.argv[1:] == ["--count-states"]:
        count_main_states()
    else:
        unittest.main()