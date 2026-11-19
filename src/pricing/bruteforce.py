"""Полный перебор дерева игры — независимая проверка решателя на малых задачах.

Написан нарочно «в лоб» и без кода из solver.py: рекурсия по узлам, в каждом узле перебор
всех цен ходящего, в каждом периоде — перебор всех исходов продаж с их вероятностями
(Пуассон считается через math, без scipy). Работает только на маленьких задачах
(до ~4 мест, 4 цен и 4 периодов), зато ошибиться в нём трудно.
"""
from __future__ import annotations

from functools import lru_cache
import math

from .scenario import EXACT, Scenario


def _sales_dist(mean, seats, mode):
    """[(вероятность, продано)] при спросе mean и seats свободных местах."""
    if seats == 0:
        return [(1.0, 0)]
    if mode == EXACT:
        return [(1.0, min(int(mean), seats))]
    out, acc = [], 0.0
    for s in range(seats):
        pr = math.exp(-mean) * mean ** s / math.factorial(s)
        out.append((pr, s))
        acc += pr
    out.append((max(0.0, 1.0 - acc), seats))      # спрос >= seats: продали всё
    return out


def brute_force(sc: Scenario, memo: bool = True, tol: float = 1e-9):
    """Возвращает функцию value(node, cA, cB, level_a, level_b) -> (выручка A, выручка B)
    с узла T_node до вылета (для node >= 1 цена ходящего в аргументах не важна),
    и функцию choice(node, cA, cB, level_a, level_b) -> выбранный уровень ходящего."""
    K, N, p = sc.n_levels, sc.n_periods, sc.prices
    eps = tol * float(p[-1]) * max(sc.capacity_a + sc.capacity_b, 1)

    def period(k, ca, cb, la, lb):
        mean_a = sc.demand_a[k, la, lb] if cb > 0 else sc.mono_a[k, la]
        mean_b = sc.demand_b[k, la, lb] if ca > 0 else sc.mono_b[k, lb]
        va = vb = 0.0
        for pa_, sa in _sales_dist(float(mean_a), ca, sc.mode):
            for pb_, sb in _sales_dist(float(mean_b), cb, sc.mode):
                pr = pa_ * pb_
                if pr == 0.0:
                    continue
                na, nb = node(k + 1, ca - sa, cb - sb, la, lb)
                va += pr * (p[la] * sa + na)
                vb += pr * (p[lb] * sb + nb)
        return va, vb

    def options(k, ca, cb, la, lb):
        """[(уровень, (vA, vB))] для всех ходов ходящего в узле k."""
        if Scenario.mover(k) == "A":
            return [(i, period(k, ca, cb, i, lb)) for i in range(K)]
        return [(i, period(k, ca, cb, la, i)) for i in range(K)]

    def best(k, ca, cb, la, lb):
        own = 0 if Scenario.mover(k) == "A" else 1
        if (own == 0 and ca == 0) or (own == 1 and cb == 0):
            return -1, period(k, ca, cb, la, lb)        # ходящему продавать нечего
        best_i, best_v = None, None
        for i, v in options(k, ca, cb, la, lb):       # по возрастанию цены
            if best_v is None or v[own] > best_v[own] + eps:
                best_i, best_v = i, v
        return best_i, best_v

    def node(k, ca, cb, la, lb):
        if k == N:
            return 0.0, 0.0
        if k == 0:
            return period(0, ca, cb, la, lb)
        return best(k, ca, cb, la, lb)[1]

    if memo:
        node = lru_cache(maxsize=None)(node)

    def choice(k, ca, cb, la, lb):
        return best(k, ca, cb, la, lb)[0]

    return node, choice
