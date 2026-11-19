"""Спрос по мультиномиальной логит-модели (MNL).

Покупатель выбирает из трёх вариантов: A, B или не лететь. Полезности

    u_A = alpha_A - beta * p_A,    u_B = alpha_B - beta * p_B,    u_отказ = 0,

вероятность выбрать вариант пропорциональна exp(u):

    q_A = e^{u_A} / (1 + e^{u_A} + e^{u_B}).

Если B распродана, она исчезает из выбора: q_A = e^{u_A} / (1 + e^{u_A}).

За период приходит ``arrivals[k]`` покупателей. Спрос на A = arrivals[k] * q_A:
* в режиме ``exact`` округляется до целого (столько людей точно захотят купить);
* в режиме ``poisson`` это среднее: если число пришедших — Пуассон(λ), то по свойству
  «прореживания» число желающих купить у A — Пуассон(λ q_A), у B — Пуассон(λ q_B),
  и они независимы.
"""
from __future__ import annotations

import numpy as np

from .scenario import EXACT, Scenario


def mnl_shares(prices, alpha_a: float, alpha_b: float, beta: float):
    """Доли выбора. Возвращает (q_a[iA, iB], q_b[iA, iB], q_a_mono[iA], q_b_mono[iB])."""
    p = np.asarray(prices, dtype=float)
    u_a = alpha_a - beta * p          # (K,)
    u_b = alpha_b - beta * p
    # численно устойчиво: делим числитель и знаменатель на e^m
    m = np.maximum(0.0, np.maximum(u_a[:, None], u_b[None, :]))
    e_a = np.exp(u_a[:, None] - m)
    e_b = np.exp(u_b[None, :] - m)
    e_0 = np.exp(-m)
    den = e_0 + e_a + e_b
    q_a = e_a / den
    q_b = e_b / den
    q_a_mono = 1.0 / (1.0 + np.exp(-u_a))
    q_b_mono = 1.0 / (1.0 + np.exp(-u_b))
    return q_a, q_b, q_a_mono, q_b_mono


def _per_period(x, n, name):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name}: одно число или список длины {n}")
    return arr


def mnl_tables(prices, arrivals, alpha_a, alpha_b, beta, mode=EXACT):
    """Таблицы спроса (demand_a, demand_b, mono_a, mono_b) для всех периодов.

    ``arrivals`` — список покупателей за период (его длина = число периодов N).
    ``alpha_a``, ``alpha_b``, ``beta`` — число или список длины N (можно менять по периодам,
    например делать покупателей ближе к вылету менее чувствительными к цене).
    """
    lam = np.asarray(arrivals, dtype=float)
    if lam.ndim != 1 or len(lam) < 1 or np.any(lam < 0):
        raise ValueError("arrivals: список неотрицательных чисел, по одному на период")
    n, K = len(lam), len(prices)
    a_a, a_b, b = (_per_period(alpha_a, n, "alpha_A"), _per_period(alpha_b, n, "alpha_B"),
                   _per_period(beta, n, "beta"))
    d_a, d_b = np.zeros((n, K, K)), np.zeros((n, K, K))
    m_a, m_b = np.zeros((n, K)), np.zeros((n, K))
    for k in range(n):
        q_a, q_b, qa1, qb1 = mnl_shares(prices, a_a[k], a_b[k], b[k])
        d_a[k], d_b[k] = lam[k] * q_a, lam[k] * q_b
        m_a[k], m_b[k] = lam[k] * qa1, lam[k] * qb1
    if mode == EXACT:
        d_a, d_b, m_a, m_b = (np.rint(x).astype(np.int64) for x in (d_a, d_b, m_a, m_b))
    return d_a, d_b, m_a, m_b


def mnl_scenario(prices, capacity_a, capacity_b, arrivals, alpha_a, alpha_b, beta,
                 mode=EXACT, start_price="max", times=None, name="") -> Scenario:
    d_a, d_b, m_a, m_b = mnl_tables(prices, arrivals, alpha_a, alpha_b, beta, mode)
    return Scenario(prices, capacity_a, capacity_b, d_a, d_b, m_a, m_b,
                    mode=mode, start_price=start_price, times=times, name=name)
