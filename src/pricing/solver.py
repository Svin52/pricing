"""Решение игры обратной индукцией (динамическим программированием).

Состояние в узле T_k (k >= 1), перед ходом:
    (cA, cB, j) — остатки мест A и B и текущая цена того, кто НЕ ходит (уровень j).
Цена самого ходящего значения не имеет: он всё равно назначает новую.

Ходящий выбирает уровень i, цены держатся на период [T_k, T_{k+1}), в нём проходят продажи,
затем в T_{k+1} ходит соперник, и для него «цена соперника» — только что выбранная i.

Уравнения (для хода A в узле k; для B симметрично):

    Q_X(k, cA, cB, j; i) = p_X * E[продажи X] + E[ V_X(k+1, cA - sA, cB - sB, i) ],   X ∈ {A, B}
    i*(k, cA, cB, j)     = наименьший i из argmax_i Q_A(k, cA, cB, j; i)
    V_X(k, cA, cB, j)    = Q_X(k, cA, cB, j; i*)

V_X(N, ·) = 0: непроданные к вылету места сгорают. Каждая компания максимизирует свою
выручку (издержек на место в модели нет, поэтому «прибыль» = выручка).

Продажи за период: sA = min(DA, cA), sB = min(DB, cB). Если места кончились посреди
периода, лишние покупатели уходят (к сопернику не перетекают). Если у компании мест 0
в начале периода, её нет на рынке — спрос на соперника берётся из таблицы mono_*.

Техника: при фиксированных ценах продажи A и B независимы, поэтому ожидание по исходам
периода — это произведение матриц  M_A · V · M_Bᵀ,  где M[c, c'] = P(остаток c' | было c).
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .scenario import EXACT, Scenario


# ----------------------------------------------------------------- распределения продаж

def demand_pmf(x, cap: int, mode: str) -> np.ndarray:
    """P(спрос = d) для d = 0..cap. Хвост (d > cap) в массив не входит."""
    x = np.asarray(x)
    d = np.arange(cap + 1)
    if mode == EXACT:
        return (x[..., None] == d).astype(float)
    from scipy.stats import poisson
    return poisson.pmf(d, np.asarray(x, dtype=float)[..., None])


def seats_matrix(pmf: np.ndarray) -> np.ndarray:
    """M[..., c, c'] = P(после периода осталось c' мест | в начале было c)."""
    cap = pmf.shape[-1] - 1
    c = np.arange(cap + 1)
    sold = c[:, None] - c[None, :]                      # c - c'
    valid = (sold >= 0) & (c[None, :] >= 1)             # c' = 0 обрабатываем отдельно
    M = pmf[..., np.where(valid, sold, 0)] * valid
    p_less = np.concatenate([np.zeros(pmf.shape[:-1] + (1,)), np.cumsum(pmf, axis=-1)[..., :-1]], axis=-1)
    M[..., :, 0] = np.clip(1.0 - p_less, 0.0, 1.0)      # P(спрос >= c): распродали всё
    return M


def expected_sales(M: np.ndarray) -> np.ndarray:
    """E[продано | было c мест] для каждого c."""
    cap = M.shape[-1] - 1
    c = np.arange(cap + 1)
    sold = np.maximum(c[:, None] - c[None, :], 0)
    return np.einsum("...ij,ij->...i", M, sold)


# ----------------------------------------------------------------- результат

@dataclass
class Solution:
    """Равновесие игры.

    value_a[k, cA, cB, j] — ожидаемая выручка A с узла T_k до вылета, если остатки (cA, cB),
        а цена того, кто не ходит в T_k, — уровень j. Для k = 0 — если обе компании
        стартуют на уровне j. value_a[N] = 0.
    policy[k, cA, cB, j] — уровень, который выберет ходящий в T_k (k >= 1);
        -1, если у ходящего мест нет (выбирать нечего). policy[0] не используется.
    """
    scenario: Scenario
    value_a: np.ndarray
    value_b: np.ndarray
    policy: np.ndarray
    seconds: float = 0.0

    # --- итоговая выручка
    def revenue(self, seats_a: int | None = None, seats_b: int | None = None,
                start_price: int | None = None) -> tuple[float, float]:
        """Ожидаемая выручка (A, B) за весь горизонт продаж."""
        sc = self.scenario
        a = sc.capacity_a if seats_a is None else seats_a
        b = sc.capacity_b if seats_b is None else seats_b
        s = sc.start_price if start_price is None else start_price
        return float(self.value_a[0, a, b, s]), float(self.value_b[0, a, b, s])

    # --- стратегии
    def best_price(self, node: int, seats_a: int, seats_b: int, opponent_level: int) -> int:
        """Уровень цены, который поставит ходящий в T_node. Для соперника это и есть
        прогноз его цены: «если я поставлю j, он ответит вот так»."""
        if not 1 <= node < self.scenario.n_periods:
            raise ValueError(f"ходы есть только в узлах 1..{self.scenario.n_periods - 1}")
        return int(self.policy[node, seats_a, seats_b, opponent_level])

    def response_table(self, node: int, seats_a: int, seats_b: int) -> np.ndarray:
        """Ответ ходящего на каждую цену соперника (массив длины K)."""
        return self.policy[node, seats_a, seats_b, :].copy()


# ----------------------------------------------------------------- решатель

class _Period:
    """Всё, что нужно для одного периода: распределения спроса и ожидания по исходам."""

    def __init__(self, sc: Scenario, k: int, pmf_a, pmf_b, pmf_am, pmf_bm):
        self.sc, self.k = sc, k
        self.pmf_a, self.pmf_b, self.pmf_am, self.pmf_bm = pmf_a[k], pmf_b[k], pmf_am[k], pmf_bm[k]
        cA, cB = sc.capacity_a, sc.capacity_b
        self.b_has = (np.arange(cB + 1) >= 1)[None, None, :]    # B на рынке
        self.a_has = (np.arange(cA + 1) >= 1)[None, :, None]    # A на рынке

    def q_values(self, ia: np.ndarray, ib: np.ndarray, f_a: np.ndarray, f_b: np.ndarray):
        """Для пар цен (ia[m], ib[m]) и продолжений f_* (m или 1, cA+1, cB+1) возвращает
        Q_A, Q_B формы (m, cA+1, cB+1): выручка периода + ожидаемое продолжение."""
        p = self.sc.prices
        MA = seats_matrix(self.pmf_a[ia, ib])      # (m, cA+1, cA+1), B на рынке
        MB = seats_matrix(self.pmf_b[ia, ib])      # (m, cB+1, cB+1), A на рынке
        MAm = seats_matrix(self.pmf_am[ia])        # A одна на рынке
        MBm = seats_matrix(self.pmf_bm[ib])        # B одна на рынке
        MBt, MBmt = np.swapaxes(MB, -1, -2), np.swapaxes(MBm, -1, -2)

        def expect(F):
            E = MA @ F @ MBt
            E[:, :, 0] = (MAm @ F[..., :, 0:1])[..., 0]       # B распродана
            E[:, 0, :] = (F[..., 0:1, :] @ MBmt)[:, 0, :]     # A распродана
            return E

        rev_a = p[ia][:, None, None] * np.where(self.b_has, expected_sales(MA)[:, :, None],
                                                expected_sales(MAm)[:, :, None])
        rev_b = p[ib][:, None, None] * np.where(self.a_has, expected_sales(MB)[:, None, :],
                                                expected_sales(MBm)[:, None, :])
        return rev_a + expect(f_a), rev_b + expect(f_b)


def solve(sc: Scenario, tol: float = 1e-9, verbose: bool = False) -> Solution:
    """Находит равновесные стратегии обратной индукцией.

    tol — относительный допуск при сравнении выручки: варианты, отличающиеся меньше
    чем на tol * (макс. цена * всего мест), считаются равными, и берётся меньшая цена.
    """
    t0 = time.perf_counter()
    K, N = sc.n_levels, sc.n_periods
    cA, cB = sc.capacity_a, sc.capacity_b
    eps = tol * sc.prices[-1] * max(cA + cB, 1)

    pmf_a = demand_pmf(sc.demand_a, cA, sc.mode)   # (N, K, K, cA+1)
    pmf_b = demand_pmf(sc.demand_b, cB, sc.mode)
    pmf_am = demand_pmf(sc.mono_a, cA, sc.mode)    # (N, K, cA+1)
    pmf_bm = demand_pmf(sc.mono_b, cB, sc.mode)

    VA = np.zeros((N + 1, cA + 1, cB + 1, K))
    VB = np.zeros_like(VA)
    policy = np.full((N + 1, cA + 1, cB + 1, K), -1, dtype=np.int16)
    levels = np.arange(K)

    for k in range(N - 1, -1, -1):
        per = _Period(sc, k, pmf_a, pmf_b, pmf_am, pmf_bm)
        nxt_a, nxt_b = VA[k + 1], VB[k + 1]               # (cA+1, cB+1, K)

        if k == 0:
            # T_0: ходов нет, обе компании на уровне j; в T_1 ходит A, цена B для неё = j
            qa, qb = per.q_values(levels, levels, np.moveaxis(nxt_a, -1, 0), np.moveaxis(nxt_b, -1, 0))
            VA[0], VB[0] = np.moveaxis(qa, 0, -1), np.moveaxis(qb, 0, -1)
            break

        mover_a = (k % 2 == 1)
        best_own = best_a = best_b = None
        best_i = np.zeros((K, cA + 1, cB + 1), dtype=np.int16)
        for i in range(K):
            ia, ib = (np.full(K, i), levels) if mover_a else (levels, np.full(K, i))
            qa, qb = per.q_values(ia, ib, nxt_a[:, :, i], nxt_b[:, :, i])
            own = qa if mover_a else qb
            if i == 0:
                best_own, best_a, best_b = own, qa, qb
                continue
            better = own > best_own + eps     # строго лучше; при равенстве остаётся меньшая цена
            best_own = np.where(better, own, best_own)
            best_a = np.where(better, qa, best_a)
            best_b = np.where(better, qb, best_b)
            best_i[better] = i

        VA[k], VB[k] = np.moveaxis(best_a, 0, -1), np.moveaxis(best_b, 0, -1)
        pol = np.moveaxis(best_i, 0, -1)
        if mover_a:
            pol[0, :, :] = -1
        else:
            pol[:, 0, :] = -1
        policy[k] = pol
        if verbose:
            print(f"  узел T_{k} ({'A' if mover_a else 'B'}) готов, {time.perf_counter() - t0:.1f} с")

    return Solution(sc, VA, VB, policy, seconds=time.perf_counter() - t0)
