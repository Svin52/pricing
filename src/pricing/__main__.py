"""Запуск из командной строки:

    python -m pricing experiments/base.json
    python -m pricing experiments/base_poisson.json --mc 20000
    python -m pricing experiments/base.json --save results/base.npz --csv results/base_policy.csv
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

from .scenario import EXACT, load_scenario
from .simulate import format_path, monte_carlo, play
from .solver import solve


def save_csv(sol, path):
    """Стратегии в «длинном» формате: одна строка — одно состояние в одном узле."""
    sc = sol.scenario
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node", "mover", "seats_a", "seats_b", "opponent_price", "best_price",
                    "value_a", "value_b"])
        for k in range(1, sc.n_periods):
            mv = sc.mover(k)
            pol = sol.policy[k]
            for ca in range(sc.capacity_a + 1):
                for cb in range(sc.capacity_b + 1):
                    for j in range(sc.n_levels):
                        i = int(pol[ca, cb, j])
                        w.writerow([k, mv, ca, cb, f"{sc.prices[j]:g}",
                                    "" if i < 0 else f"{sc.prices[i]:g}",
                                    f"{sol.value_a[k, ca, cb, j]:.2f}", f"{sol.value_b[k, ca, cb, j]:.2f}"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pricing",
                                 description="Равновесные цены двух авиакомпаний (обратная индукция).")
    ap.add_argument("scenario", help="JSON-файл сценария")
    ap.add_argument("--mc", type=int, default=0, help="проверить выручку Монте-Карло на N партиях")
    ap.add_argument("--seed", type=int, default=1, help="seed для случайной траектории и Монте-Карло")
    ap.add_argument("--save", help="сохранить стратегии и выручку в .npz")
    ap.add_argument("--csv", help="сохранить стратегии в CSV (для небольших задач)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    sc = load_scenario(args.scenario)
    print(sc.describe())
    sol = solve(sc, verbose=args.verbose)
    ra, rb = sol.revenue()
    print(f"\nРешено за {sol.seconds:.2f} с")
    print(f"Ожидаемая выручка: A = {ra:,.0f}   B = {rb:,.0f}".replace(",", " "))

    steps = play(sol, seed=args.seed)
    title = "Равновесная траектория" if sc.mode == EXACT else f"Одна случайная партия (seed={args.seed})"
    print(f"\n{title}:\n{format_path(steps, sc.times)}")

    if args.mc:
        mc = monte_carlo(sol, args.mc, seed=args.seed)
        print(f"\nМонте-Карло, {mc.n} партий:")
        print(f"  A: {mc.mean_a:,.0f} ± {mc.se_a:,.0f}   (расчёт {ra:,.0f})".replace(",", " "))
        print(f"  B: {mc.mean_b:,.0f} ± {mc.se_b:,.0f}   (расчёт {rb:,.0f})".replace(",", " "))
        print("  совпадает в пределах погрешности" if mc.agrees(ra, rb) else "  РАСХОЖДЕНИЕ!")

    if args.save:
        pathlib.Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, prices=sc.prices, policy=sol.policy,
                            value_a=sol.value_a, value_b=sol.value_b,
                            revenue=np.array([ra, rb]), start_price=sc.start_price)
        print(f"\nСтратегии сохранены: {args.save}")
    if args.csv:
        pathlib.Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        save_csv(sol, args.csv)
        print(f"CSV сохранён: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
