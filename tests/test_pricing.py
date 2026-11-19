"""Тесты решателя, симулятора и базовых эвристик.

Запуск из корня репозитория:  python -m pytest -q
"""
import itertools
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pricing import (  # noqa: E402
    ALL_BASELINES,
    EXACT,
    POISSON,
    Scenario,
    brute_force,
    dp_policy,
    load_scenario,
    mnl_scenario,
    monte_carlo,
    play,
    solve,
)
from pricing.solver import demand_pmf, expected_sales, seats_matrix  # noqa: E402


def random_table_scenario(rng, K, N, ca, cb, mode, start):
    """Произвольные таблицы спроса (не MNL), чтобы проверить решатель в общем виде."""
    prices = np.sort(rng.choice(np.arange(1, 20), size=K, replace=False)).astype(float) * 100
    if mode == EXACT:
        gen = lambda *s: rng.integers(0, 4, size=s)
    else:
        gen = lambda *s: rng.uniform(0, 2.5, size=s)
    return Scenario(prices, ca, cb, gen(N, K, K), gen(N, K, K), gen(N, K), gen(N, K),
                    mode=mode, start_price=start)


def compare_with_brute_force(sc):
    sol = solve(sc)
    node, choice = brute_force(sc)
    K, N = sc.n_levels, sc.n_periods
    for ca, cb, j in itertools.product(range(sc.capacity_a + 1), range(sc.capacity_b + 1), range(K)):
        va, vb = node(0, ca, cb, j, j)
        assert sol.value_a[0, ca, cb, j] == pytest.approx(va, abs=1e-7)
        assert sol.value_b[0, ca, cb, j] == pytest.approx(vb, abs=1e-7)
        for k in range(1, N):
            la, lb = (0, j) if sc.mover(k) == "A" else (j, 0)
            va, vb = node(k, ca, cb, la, lb)
            assert sol.value_a[k, ca, cb, j] == pytest.approx(va, abs=1e-7)
            assert sol.value_b[k, ca, cb, j] == pytest.approx(vb, abs=1e-7)
            assert sol.policy[k, ca, cb, j] == choice(k, ca, cb, la, lb)


@pytest.mark.parametrize("mode", [EXACT, POISSON])
@pytest.mark.parametrize("seed", range(6))
def test_matches_brute_force_random_tables(mode, seed):
    rng = np.random.default_rng(seed)
    K = int(rng.integers(2, 5))
    N = int(rng.integers(1, 5))
    ca, cb = int(rng.integers(0, 4)), int(rng.integers(0, 4))
    start = "max" if seed % 2 else "min"
    compare_with_brute_force(random_table_scenario(rng, K, N, ca, cb, mode, start))


@pytest.mark.parametrize("mode", [EXACT, POISSON])
def test_matches_brute_force_mnl(mode):
    prices = [1000, 2000, 3000, 4000]
    sc = mnl_scenario(prices, 4, 3, arrivals=[3, 2, 4, 3], alpha_a=2.0, alpha_b=1.6,
                      beta=0.0008, mode=mode)
    compare_with_brute_force(sc)


def test_brute_force_memo_equals_plain():
    rng = np.random.default_rng(42)
    sc = random_table_scenario(rng, 3, 3, 2, 2, POISSON, "max")
    a, _ = brute_force(sc, memo=True)
    b, _ = brute_force(sc, memo=False)
    assert a(0, 2, 2, 2, 2) == pytest.approx(b(0, 2, 2, 2, 2), abs=1e-12)


@pytest.mark.parametrize("mode", [EXACT, POISSON])
def test_monte_carlo_agrees(mode):
    sc = mnl_scenario(np.linspace(3000, 14500, 24), 20, 15, arrivals=[6] * 10,
                      alpha_a=4.5, alpha_b=4.2, beta=0.0005, mode=mode)
    sol = solve(sc)
    ra, rb = sol.revenue()
    mc = monte_carlo(sol, 20000, seed=3)
    assert mc.agrees(ra, rb)
    if mode == EXACT:
        assert mc.se_a == 0 and mc.mean_a == pytest.approx(ra)


def test_exact_path_sums_to_revenue():
    sc = load_scenario(ROOT / "experiments" / "base.json")
    sol = solve(sc)
    steps = play(sol)
    ra, rb = sol.revenue()
    assert sum(s.revenue_a for s in steps) == pytest.approx(ra)
    assert sum(s.revenue_b for s in steps) == pytest.approx(rb)
    assert steps[0].price_a == steps[0].price_b == sc.prices[-1]
    assert [s.mover for s in steps[1:4]] == ["A", "B", "A"]
    assert all(s.sold_a <= s.seats_a and s.sold_b <= s.seats_b for s in steps)


def test_tie_takes_lower_price():
    prices = np.array([100.0, 200.0, 400.0])
    q = np.array([4, 2, 1])
    N, K = 3, 3
    d_a = np.broadcast_to(q[None, :, None], (N, K, K)).copy()
    d_b = np.broadcast_to(q[None, None, :], (N, K, K)).copy()
    sc = Scenario(prices, 100, 100, d_a, d_b, d_a[:, :, 0], d_b[:, 0, :], start_price="max")
    sol = solve(sc)
    assert np.all(sol.policy[1:N, 20:, 20:, :] == 0)


def test_sold_out_mover_has_no_choice():
    sc = mnl_scenario([1000, 2000, 3000], 3, 3, arrivals=[2] * 4, alpha_a=1, alpha_b=1, beta=0.001)
    sol = solve(sc)
    assert np.all(sol.policy[1, 0, :, :] == -1)
    assert np.all(sol.policy[2, :, 0, :] == -1)
    assert np.all(sol.policy[1, 1:, :, :] >= 0)


def test_sold_out_competitor_uses_mono_demand():
    prices = [100.0, 200.0]
    N, K = 2, 2
    d = np.zeros((N, K, K), dtype=int)
    mono = np.full((N, K), 5, dtype=int)
    sc = Scenario(prices, 10, 0, d, d, mono, mono, start_price="min")
    sol = solve(sc)
    assert sol.revenue() == pytest.approx((5 * 100 + 5 * 200, 0.0))


def test_unsold_seats_burn():
    sc = mnl_scenario([100, 200], 5, 5, arrivals=[0, 0, 0], alpha_a=1, alpha_b=1, beta=0.01)
    assert solve(sc).revenue() == (0.0, 0.0)


def test_seats_matrix_rows_are_distributions():
    for mode, x in ((EXACT, np.array([0, 2, 7])), (POISSON, np.array([0.0, 1.3, 9.0]))):
        M = seats_matrix(demand_pmf(x, 5, mode))
        assert np.allclose(M.sum(-1), 1.0)
        assert np.all(np.triu(M[..., :, :], 1) == 0)
    M = seats_matrix(demand_pmf(np.array([2]), 5, EXACT))[0]
    assert np.allclose(expected_sales(M), [0, 1, 2, 2, 2, 2])


def test_load_examples():
    for f in ("base.json", "base_poisson.json", "table_example.json"):
        sc = load_scenario(ROOT / "experiments" / f)
        assert sc.n_levels == 24


def test_dp_policy_matches_play_sol():
    sc = load_scenario(ROOT / "experiments" / "base.json")
    sol = solve(sc)
    steps_direct = play(sol)
    steps_wrapped = play(dp_policy(sol), dp_policy(sol), scenario=sc)
    assert [(s.level_a, s.level_b, s.sold_a, s.sold_b) for s in steps_direct] == [
        (s.level_a, s.level_b, s.sold_a, s.sold_b) for s in steps_wrapped
    ]


@pytest.mark.parametrize("name,fn", list(ALL_BASELINES.items()))
def test_baselines_run_against_dp(name, fn):
    sc = mnl_scenario(np.linspace(3000, 14500, 24), 20, 20, arrivals=[6] * 6,
                      alpha_a=4.5, alpha_b=4.2, beta=0.0005, mode=POISSON)
    sol = solve(sc)
    steps = play(sol, fn, scenario=sc, seed=7)
    assert len(steps) == sc.n_periods
    mc = monte_carlo(sol, 20, seed=7, strategy_b=fn, scenario=sc)
    assert mc.n == 20
    assert mc.mean_a >= 0 and mc.mean_b >= 0
