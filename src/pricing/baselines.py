"""Базовые эвристические стратегии (бейзлайны) и единый интерфейс стратегий.

Любая стратегия в проекте (как оптимальная из ДП, так и эвристика) реализует
интерфейс ``PolicyFn(state: StrategyState, sc: Scenario) -> int`` и возвращает
индекс ценового уровня ``0 .. sc.n_levels - 1``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .scenario import Scenario
from .solver import Solution


@dataclass(frozen=True)
class StrategyState:
    """Наблюдаемое состояние игры в момент хода игрока."""
    period: int
    player: str          # "A" или "B"
    my_seats: int
    opp_seats: int
    my_level: int
    opp_level: int       # -1, если соперник распродан (opp_seats == 0)
    my_revenue: float
    opp_revenue: float


PolicyFn = Callable[[StrategyState, Scenario], int]


def dp_policy(sol: Solution) -> PolicyFn:
    """Оборачивает решение обратной индукции в единый интерфейс PolicyFn."""
    def _choose(state: StrategyState, sc: Scenario) -> int:
        ca, cb = (
            (state.my_seats, state.opp_seats)
            if state.player == "A"
            else (state.opp_seats, state.my_seats)
        )
        opp_lvl = sc.start_price if state.opp_level < 0 else state.opp_level
        return sol.best_price(state.period, ca, cb, opp_lvl)

    return _choose


def _expected_demand_pair(
    sc: Scenario,
    period: int,
    player: str,
    my_level: int,
    opp_level: int,
    opp_seats: int,
) -> tuple[float, float]:
    """Ожидаемый спрос (мой, соперника) с учётом возможного ухода соперника с рынка."""
    if opp_seats <= 0 or opp_level < 0:
        d_my = sc.mono_a[period, my_level] if player == "A" else sc.mono_b[period, my_level]
        return float(d_my), 0.0
    if player == "A":
        return float(sc.demand_a[period, my_level, opp_level]), float(sc.demand_b[period, my_level, opp_level])
    return float(sc.demand_b[period, opp_level, my_level]), float(sc.demand_a[period, opp_level, my_level])


def _expected_my_demand(state: StrategyState, sc: Scenario, my_level: int, period: int | None = None) -> float:
    k = state.period if period is None else period
    d_my, _ = _expected_demand_pair(sc, k, state.player, my_level, state.opp_level, state.opp_seats)
    return d_my


def heuristic_myopic_1step(state: StrategyState, sc: Scenario) -> int:
    """Жадная максимизация ожидаемой выручки за текущий период."""
    best_idx = 0
    best_profit = -1.0
    for idx in range(sc.n_levels):
        sold = min(state.my_seats, _expected_my_demand(state, sc, idx))
        profit = sold * sc.prices[idx]
        if profit > best_profit:
            best_profit = profit
            best_idx = idx
    return best_idx


def heuristic_myopic_2step(state: StrategyState, sc: Scenario) -> int:
    """Жадная максимизация ожидаемой выручки на 2 периода вперёд при неизменной цене соперника."""
    best_idx = 0
    best_profit = -1.0
    k = state.period

    for idx in range(sc.n_levels):
        d1, d_opp1 = _expected_demand_pair(sc, k, state.player, idx, state.opp_level, state.opp_seats)
        sold1 = min(state.my_seats, d1)
        seats_left = state.my_seats - sold1
        profit = sold1 * sc.prices[idx]

        if k + 1 < sc.n_periods and seats_left > 0:
            opp_left = max(0.0, state.opp_seats - d_opp1)
            d2, _ = _expected_demand_pair(
                sc, k + 1, state.player, idx, state.opp_level, int(round(opp_left))
            )
            sold2 = min(seats_left, d2)
            profit += sold2 * sc.prices[idx]

        if profit > best_profit:
            best_profit = profit
            best_idx = idx
    return best_idx


def heuristic_target_pace(state: StrategyState, sc: Scenario) -> int:
    """Поддержание равномерного темпа продаж до вылета."""
    periods_left = max(1, sc.n_periods - state.period)
    target_sales = state.my_seats / periods_left

    best_idx = 0
    min_diff = float("inf")
    for idx in range(sc.n_levels):
        diff = abs(_expected_my_demand(state, sc, idx) - target_sales)
        if diff < min_diff:
            min_diff = diff
            best_idx = idx
    return best_idx


def heuristic_s_curve(state: StrategyState, sc: Scenario) -> int:
    """Трёхфазная стратегия: премиум в начале, темп в середине, распродажа в конце."""
    progress = state.period / sc.n_periods
    periods_left = max(1, sc.n_periods - state.period)

    if progress < 0.30:
        min_idx = min(sc.n_levels - 1, round(sc.n_levels * 2 / 3))
        best_idx = min_idx
        best_profit = -1.0
        for idx in range(min_idx, sc.n_levels):
            sold = min(state.my_seats, _expected_my_demand(state, sc, idx))
            profit = sold * sc.prices[idx]
            if profit > best_profit:
                best_profit = profit
                best_idx = idx
        return best_idx

    if progress >= 0.75:
        target_sales = state.my_seats / periods_left
        for idx in range(sc.n_levels):
            if _expected_my_demand(state, sc, idx) >= target_sales:
                return idx
        return 0

    return heuristic_target_pace(state, sc)


def heuristic_undercut(state: StrategyState, sc: Scenario) -> int:
    """Ставит цену на 1 уровень ниже соперника (с полом на нижней шестой сетки)."""
    if state.opp_seats <= 0 or state.opp_level < 0:
        return heuristic_myopic_1step(state, sc)
    floor_idx = min(sc.n_levels - 1, sc.n_levels // 6)
    return max(floor_idx, state.opp_level - 1)


def heuristic_tit_for_tat(state: StrategyState, sc: Scenario) -> int:
    """Повторяет текущую цену соперника (если соперник на рынке)."""
    if state.opp_seats <= 0 or state.opp_level < 0:
        return heuristic_myopic_1step(state, sc)
    return state.opp_level


def heuristic_shadow_price(state: StrategyState, sc: Scenario) -> int:
    """Ориентируется на теневую стоимость места с учётом дефицита."""
    periods_left = max(1, sc.n_periods - state.period)
    scarcity = periods_left / max(1, state.my_seats)
    median_price = sc.prices[sc.n_levels // 2]
    shadow_value = min(sc.prices[-1] * 0.95, median_price * scarcity * 8.0)

    best_idx = sc.n_levels - 1
    best_utility = -float("inf")
    for idx in range(sc.n_levels):
        net_margin = sc.prices[idx] - shadow_value
        if net_margin <= 0:
            continue
        sold = min(state.my_seats, _expected_my_demand(state, sc, idx))
        utility = sold * net_margin
        if utility > best_utility:
            best_utility = utility
            best_idx = idx
    return best_idx


def heuristic_stackelberg_1ply(state: StrategyState, sc: Scenario) -> int:
    """Прогноз на 2 периода с учётом жадного ответа соперника в следующем периоде."""
    best_idx = 0
    best_profit = -1.0
    k = state.period

    for my_idx in range(sc.n_levels):
        d_my1, d_opp1 = _expected_demand_pair(sc, k, state.player, my_idx, state.opp_level, state.opp_seats)
        sold_my1 = min(state.my_seats, d_my1)
        sold_opp1 = min(state.opp_seats, d_opp1)

        my_left = state.my_seats - sold_my1
        opp_left = max(0, int(round(state.opp_seats - sold_opp1)))
        profit = sold_my1 * sc.prices[my_idx]

        if k + 1 < sc.n_periods and my_left > 0:
            if opp_left == 0:
                opp_next_idx = -1
            else:
                best_opp_profit = -1.0
                opp_next_idx = 0
                opp_player = "B" if state.player == "A" else "A"
                for cand_opp in range(sc.n_levels):
                    d_opp2, _ = _expected_demand_pair(
                        sc, k + 1, opp_player, cand_opp, my_idx, int(round(my_left))
                    )
                    opp_p = min(opp_left, d_opp2) * sc.prices[cand_opp]
                    if opp_p > best_opp_profit:
                        best_opp_profit = opp_p
                        opp_next_idx = cand_opp

            d_my2, _ = _expected_demand_pair(sc, k + 1, state.player, my_idx, opp_next_idx, opp_left)
            profit += min(my_left, d_my2) * sc.prices[my_idx]

        if profit > best_profit:
            best_profit = profit
            best_idx = my_idx

    return best_idx


ALL_BASELINES: dict[str, PolicyFn] = {
    "Myopic 1-Step": heuristic_myopic_1step,
    "Myopic 2-Step": heuristic_myopic_2step,
    "Target Pace": heuristic_target_pace,
    "S-Curve": heuristic_s_curve,
    "Undercut": heuristic_undercut,
    "Tit-for-Tat": heuristic_tit_for_tat,
    "Shadow Price": heuristic_shadow_price,
    "Stackelberg 1-Ply": heuristic_stackelberg_1ply,
}
