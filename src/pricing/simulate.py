"""Розыгрыш продаж по найденным стратегиям и эвристикам: траектория цен и проверка Монте-Карло."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines import PolicyFn, StrategyState, dp_policy
from .scenario import EXACT, Scenario
from .solver import Solution


@dataclass
class Step:
    period: int          # период [T_k, T_{k+1})
    mover: str           # кто ходил в T_k ('-' для T_0)
    level_a: int         # уровни цен (0..K-1) в этом периоде
    level_b: int
    price_a: float
    price_b: float
    seats_a: int         # места в начале периода
    seats_b: int
    demand_a: float      # сколько хотели купить
    demand_b: float
    sold_a: int
    sold_b: int
    revenue_a: float
    revenue_b: float


def _means(sc: Scenario, k: int, la: int, lb: int, ca: int, cb: int):
    """Спрос в периоде k при уровнях (la, lb) с учётом того, кто уже ушёл с рынка."""
    da = np.where(cb > 0, sc.demand_a[k, la, lb], sc.mono_a[k, la])
    db = np.where(ca > 0, sc.demand_b[k, la, lb], sc.mono_b[k, lb])
    return da, db


def _resolve_strategies(
    strategy_a: Solution | PolicyFn,
    strategy_b: Solution | PolicyFn | None,
    scenario: Scenario | None,
) -> tuple[Scenario, PolicyFn, PolicyFn]:
    """Приводит аргументы play/monte_carlo к (Scenario, policy_a, policy_b)."""
    sc = scenario
    if sc is None:
        if isinstance(strategy_a, Solution):
            sc = strategy_a.scenario
        elif isinstance(strategy_b, Solution):
            sc = strategy_b.scenario
        else:
            raise ValueError("Передайте scenario, если обе стратегии — функции PolicyFn")

    if strategy_b is None:
        if not isinstance(strategy_a, Solution):
            raise ValueError("Если задана одна стратегия, она должна быть объектом Solution")
        strategy_b = strategy_a

    fn_a = dp_policy(strategy_a) if isinstance(strategy_a, Solution) else strategy_a
    fn_b = dp_policy(strategy_b) if isinstance(strategy_b, Solution) else strategy_b
    return sc, fn_a, fn_b


def _ask_level(fn: PolicyFn, state: StrategyState, sc: Scenario) -> int:
    lvl = fn(state, sc)
    if not isinstance(lvl, (int, np.integer)) or not (0 <= lvl < sc.n_levels):
        raise ValueError(f"Стратегия {state.player} вернула недопустимый уровень цены: {lvl}")
    return int(lvl)


def play(
    strategy_a: Solution | PolicyFn,
    strategy_b: Solution | PolicyFn | None = None,
    *,
    scenario: Scenario | None = None,
    seed: int | None = None,
) -> list[Step]:
    """Одна партия.

    Можно передать либо один ``Solution`` (тогда обе компании играют по ДП),
    либо пару стратегий (``Solution`` или ``PolicyFn`` из ``baselines``) и ``scenario``.
    """
    sc, fn_a, fn_b = _resolve_strategies(strategy_a, strategy_b, scenario)
    rng = np.random.default_rng(seed)
    ca, cb = sc.capacity_a, sc.capacity_b
    la = lb = sc.start_price
    rev_a = rev_b = 0.0
    steps = []

    for k in range(sc.n_periods):
        mover = "-"
        if k >= 1:
            mover = sc.mover(k)
            if mover == "A" and ca > 0:
                opp_lvl = -1 if cb == 0 else lb
                la = _ask_level(fn_a, StrategyState(k, "A", ca, cb, la, opp_lvl, rev_a, rev_b), sc)
            elif mover == "B" and cb > 0:
                opp_lvl = -1 if ca == 0 else la
                lb = _ask_level(fn_b, StrategyState(k, "B", cb, ca, lb, opp_lvl, rev_b, rev_a), sc)

        da, db = (float(x) for x in _means(sc, k, la, lb, ca, cb))
        if sc.mode != EXACT:
            da, db = float(rng.poisson(da)), float(rng.poisson(db))
        sa, sb = int(min(da, ca)), int(min(db, cb))
        pa, pb = float(sc.prices[la]), float(sc.prices[lb])
        ra_step, rb_step = pa * sa, pb * sb
        rev_a += ra_step
        rev_b += rb_step
        steps.append(Step(k, mover, la, lb, pa, pb, ca, cb, da, db, sa, sb, ra_step, rb_step))
        ca, cb = ca - sa, cb - sb
    return steps


def format_path(steps: list[Step], times=None) -> str:
    head = (f"{'период':>8} {'ход':>4} {'цена A':>9} {'цена B':>9} {'места A':>8} {'места B':>8} "
            f"{'спрос A':>8} {'спрос B':>8} {'прод. A':>8} {'прод. B':>8} {'выручка A':>11} {'выручка B':>11}")
    out = [head, "-" * len(head)]
    for s in steps:
        per = f"{s.period}" if times is None else f"{times[s.period]:g}–{times[s.period + 1]:g}"
        dem = (lambda x: f"{x:.0f}" if float(x).is_integer() else f"{x:.2f}")
        out.append(f"{per:>8} {s.mover:>4} {s.price_a:>9.0f} {s.price_b:>9.0f} {s.seats_a:>8} {s.seats_b:>8} "
                   f"{dem(s.demand_a):>8} {dem(s.demand_b):>8} {s.sold_a:>8} {s.sold_b:>8} "
                   f"{s.revenue_a:>11.0f} {s.revenue_b:>11.0f}")
    ta, tb = sum(s.revenue_a for s in steps), sum(s.revenue_b for s in steps)
    out.append("-" * len(head))
    out.append(f"{'итого':>8} {'':>4} {'':>9} {'':>9} {'':>8} {'':>8} {'':>8} {'':>8} "
               f"{sum(s.sold_a for s in steps):>8} {sum(s.sold_b for s in steps):>8} {ta:>11.0f} {tb:>11.0f}")
    return "\n".join(out)


@dataclass
class MonteCarlo:
    n: int
    mean_a: float
    mean_b: float
    se_a: float          # стандартная ошибка среднего
    se_b: float

    def agrees(self, rev_a: float, rev_b: float, z: float = 4.0) -> bool:
        """Совпадает ли с расчётной выручкой в пределах z стандартных ошибок."""
        ok_a = abs(self.mean_a - rev_a) <= z * self.se_a + 1e-6 * max(1.0, abs(rev_a))
        ok_b = abs(self.mean_b - rev_b) <= z * self.se_b + 1e-6 * max(1.0, abs(rev_b))
        return ok_a and ok_b


def monte_carlo(
    strategy_a: Solution | PolicyFn,
    n_sims: int = 20000,
    seed: int = 0,
    *,
    strategy_b: Solution | PolicyFn | None = None,
    scenario: Scenario | None = None,
) -> MonteCarlo:
    """Разыгрывает ``n_sims`` партий и усредняет выручку.

    Если передан один ``Solution`` (без ``strategy_b``), использует быстрый
    векторизованный прогон; иначе прогоняет партии через ``play``.
    """
    se = (lambda x: float(x.std(ddof=1) / np.sqrt(n_sims)) if n_sims > 1 else 0.0)

    if isinstance(strategy_a, Solution) and strategy_b is None:
        sol = strategy_a
        sc = sol.scenario
        rng = np.random.default_rng(seed)
        ca = np.full(n_sims, sc.capacity_a)
        cb = np.full(n_sims, sc.capacity_b)
        la = np.full(n_sims, sc.start_price)
        lb = np.full(n_sims, sc.start_price)
        rev_a = np.zeros(n_sims)
        rev_b = np.zeros(n_sims)
        for k in range(sc.n_periods):
            if k >= 1:
                if sc.mover(k) == "A":
                    new = sol.policy[k, ca, cb, lb]
                    la = np.where(new >= 0, new, la)
                else:
                    new = sol.policy[k, ca, cb, la]
                    lb = np.where(new >= 0, new, lb)
            da, db = _means(sc, k, la, lb, ca, cb)
            if sc.mode != EXACT:
                da, db = rng.poisson(da), rng.poisson(db)
            sa, sb = np.minimum(da, ca).astype(np.int64), np.minimum(db, cb).astype(np.int64)
            rev_a += sc.prices[la] * sa
            rev_b += sc.prices[lb] * sb
            ca, cb = ca - sa, cb - sb
        return MonteCarlo(n_sims, float(rev_a.mean()), float(rev_b.mean()), se(rev_a), se(rev_b))

    sc, fn_a, fn_b = _resolve_strategies(strategy_a, strategy_b, scenario)
    rng = np.random.default_rng(seed)
    rev_a = np.zeros(n_sims)
    rev_b = np.zeros(n_sims)
    for i in range(n_sims):
        sim_seed = int(rng.integers(0, 2**31 - 1))
        steps = play(fn_a, fn_b, scenario=sc, seed=sim_seed)
        rev_a[i] = sum(s.revenue_a for s in steps)
        rev_b[i] = sum(s.revenue_b for s in steps)
    return MonteCarlo(n_sims, float(rev_a.mean()), float(rev_b.mean()), se(rev_a), se(rev_b))
