"""Круговой турнир: равновесие ДП (обратная индукция) против всех эвристик.

Запуск из корня репозитория:
    python experiments/tournament.py
    python experiments/tournament.py experiments/base_poisson.json --mc 200
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from pricing import (  # noqa: E402
    ALL_BASELINES,
    EXACT,
    PolicyFn,
    dp_policy,
    load_scenario,
    monte_carlo,
    play,
    solve,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run_tournament(scenario_path: pathlib.Path, mc_sims: int = 200, seed: int = 42) -> None:
    sc = load_scenario(scenario_path)
    print(sc.describe())

    sol = solve(sc)
    print(f"\nДП решено за {sol.seconds:.2f} с")

    players: dict[str, PolicyFn] = {"DP (Nash)": dp_policy(sol), **ALL_BASELINES}
    names = list(players.keys())
    stats = {n: {"total_rev": 0.0, "wins": 0, "games": 0, "leftovers": 0.0} for n in names}

    for name_a in names:
        for name_b in names:
            if sc.mode == EXACT:
                steps = play(players[name_a], players[name_b], scenario=sc)
                rev_a = sum(s.revenue_a for s in steps)
                rev_b = sum(s.revenue_b for s in steps)
                left_a = steps[-1].seats_a - steps[-1].sold_a
                left_b = steps[-1].seats_b - steps[-1].sold_b
            else:
                mc = monte_carlo(
                    players[name_a],
                    mc_sims,
                    seed=seed,
                    strategy_b=players[name_b],
                    scenario=sc,
                )
                rev_a, rev_b = mc.mean_a, mc.mean_b
                steps = play(players[name_a], players[name_b], scenario=sc, seed=seed)
                left_a = steps[-1].seats_a - steps[-1].sold_a
                left_b = steps[-1].seats_b - steps[-1].sold_b

            for name, rev, opp_rev, left in (
                (name_a, rev_a, rev_b, left_a),
                (name_b, rev_b, rev_a, left_b),
            ):
                stats[name]["total_rev"] += rev
                stats[name]["games"] += 1
                stats[name]["leftovers"] += left
                if rev > opp_rev:
                    stats[name]["wins"] += 1

    ranking = sorted(
        stats.items(),
        key=lambda item: item[1]["total_rev"] / item[1]["games"],
        reverse=True,
    )

    print(f"\n{'Место':<6} | {'Стратегия':<19} | {'Ср. выручка':<12} | {'Побед':<9} | {'Ср. остаток'}")
    print("-" * 70)
    for rank, (name, s) in enumerate(ranking, start=1):
        avg_rev = s["total_rev"] / s["games"]
        avg_left = s["leftovers"] / s["games"]
        print(f"{rank:<6} | {name:<19} | {avg_rev:<12.1f} | {s['wins']:>2}/{s['games']:<6} | {avg_left:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Турнир ДП против эвристик.")
    ap.add_argument(
        "scenario",
        nargs="?",
        default=str(ROOT / "experiments" / "base.json"),
        help="Путь к JSON-сценарию (по умолчанию experiments/base.json)",
    )
    ap.add_argument("--mc", type=int, default=200, help="число партий Монте-Карло для режима poisson")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_tournament(pathlib.Path(args.scenario), mc_sims=args.mc, seed=args.seed)


if __name__ == "__main__":
    main()
