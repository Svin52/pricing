"""Сценарий рынка: сетка цен, вместимости, периоды и спрос.

Обозначения (используются во всём пакете):

* K — число ценовых уровней, цены ``prices[0] < prices[1] < ... < prices[K-1]``;
  в коде цена всегда задаётся **индексом уровня** 0..K-1.
* Временные метки T_0 < T_1 < ... < T_N. Период k — это промежуток [T_k, T_{k+1}),
  всего N периодов (k = 0..N-1). T_N — вылет.
* ``demand_a[k, iA, iB]`` — спрос на компанию A в периоде k, если A стоит на уровне iA,
  а B на уровне iB. Аналогично ``demand_b[k, iA, iB]`` (индексы тоже в порядке «A, потом B»).
* ``mono_a[k, iA]`` — спрос на A в периоде k, если B уже всё распродала и ушла с рынка.
  Аналогично ``mono_b[k, iB]``.
* ``mode``:
    - ``"exact"`` — спрос известен точно: в таблицах целые числа покупателей;
    - ``"poisson"`` — спрос случайный: в таблицах средние, число желающих купить
      у каждой компании распределено по Пуассону с этим средним.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib

import numpy as np

EXACT = "exact"
POISSON = "poisson"
MODES = (EXACT, POISSON)


@dataclass
class Scenario:
    prices: np.ndarray
    capacity_a: int
    capacity_b: int
    demand_a: np.ndarray
    demand_b: np.ndarray
    mono_a: np.ndarray
    mono_b: np.ndarray
    mode: str = EXACT
    start_price: int | str = "max"
    times: np.ndarray | None = None
    name: str = ""

    def __post_init__(self):
        self.prices = np.asarray(self.prices, dtype=float)
        if self.prices.ndim != 1 or len(self.prices) < 1:
            raise ValueError("prices: нужен одномерный список цен")
        if np.any(np.diff(self.prices) <= 0):
            raise ValueError("prices: цены должны строго возрастать")
        if np.any(self.prices < 0):
            raise ValueError("prices: цены не могут быть отрицательными")
        K = len(self.prices)

        for nm in ("capacity_a", "capacity_b"):
            v = getattr(self, nm)
            if int(v) != v or v < 0:
                raise ValueError(f"{nm}: нужно целое неотрицательное число мест")
            setattr(self, nm, int(v))

        if self.mode not in MODES:
            raise ValueError(f"mode: допустимо {MODES}, получено {self.mode!r}")

        self.demand_a = np.asarray(self.demand_a, dtype=float)
        self.demand_b = np.asarray(self.demand_b, dtype=float)
        if self.demand_a.ndim != 3 or self.demand_a.shape[1:] != (K, K):
            raise ValueError(f"demand_a: ожидается форма (N, {K}, {K}), получено {self.demand_a.shape}")
        if self.demand_b.shape != self.demand_a.shape:
            raise ValueError("demand_b: форма должна совпадать с demand_a")
        N = self.demand_a.shape[0]
        if N < 1:
            raise ValueError("нужен хотя бы один период")

        self.mono_a = np.asarray(self.mono_a, dtype=float)
        self.mono_b = np.asarray(self.mono_b, dtype=float)
        for nm in ("mono_a", "mono_b"):
            if getattr(self, nm).shape != (N, K):
                raise ValueError(f"{nm}: ожидается форма ({N}, {K}), получено {getattr(self, nm).shape}")

        for nm in ("demand_a", "demand_b", "mono_a", "mono_b"):
            arr = getattr(self, nm)
            if not np.all(np.isfinite(arr)) or np.any(arr < 0):
                raise ValueError(f"{nm}: спрос должен быть конечным и неотрицательным")
            if self.mode == EXACT:
                if not np.allclose(arr, np.rint(arr)):
                    raise ValueError(f"{nm}: в режиме exact спрос — целое число покупателей")
                setattr(self, nm, np.rint(arr).astype(np.int64))

        if isinstance(self.start_price, str):
            if self.start_price == "max":
                self.start_price = K - 1
            elif self.start_price == "min":
                self.start_price = 0
            else:
                raise ValueError("start_price: 'max', 'min' или индекс уровня 0..K-1")
        if int(self.start_price) != self.start_price or not 0 <= self.start_price < K:
            raise ValueError("start_price: индекс уровня должен быть в 0..K-1")
        self.start_price = int(self.start_price)

        if self.times is not None:
            self.times = np.asarray(self.times, dtype=float)
            if self.times.shape != (N + 1,):
                raise ValueError(f"times: нужно N+1 = {N + 1} меток, получено {self.times.shape}")
            if np.any(np.diff(self.times) <= 0):
                raise ValueError("times: метки должны строго возрастать")

    @property
    def n_levels(self) -> int:
        return len(self.prices)

    @property
    def n_periods(self) -> int:
        return self.demand_a.shape[0]

    @staticmethod
    def mover(node: int) -> str:
        """Кто ходит в узле T_node (node >= 1): A в нечётных, B в чётных."""
        if node < 1:
            raise ValueError("в T_0 никто не ходит: цены стартовые")
        return "A" if node % 2 == 1 else "B"

    def describe(self) -> str:
        p = self.prices
        lines = [
            f"Сценарий: {self.name or '(без названия)'}",
            f"  ценовых уровней: {self.n_levels} ({p[0]:g} … {p[-1]:g})",
            f"  мест: A = {self.capacity_a}, B = {self.capacity_b}",
            f"  периодов: {self.n_periods} (ходов: {max(self.n_periods - 1, 0)})",
            f"  спрос: {'точный' if self.mode == EXACT else 'случайный (Пуассон)'}",
            f"  стартовая цена в T_0: уровень {self.start_price + 1} = {p[self.start_price]:g}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------- загрузка из файла

def _price_grid(spec) -> np.ndarray:
    if isinstance(spec, dict):
        return np.linspace(float(spec["min"]), float(spec["max"]), int(spec.get("levels", 24)))
    return np.asarray(spec, dtype=float)


def load_scenario(path: str | pathlib.Path) -> Scenario:
    """Читает сценарий из JSON (формат описан в docs/solver.md)."""
    from .demand import mnl_scenario  # локальный импорт, чтобы не было цикла

    path = pathlib.Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    prices = _price_grid(cfg["prices"])
    cap = cfg["capacity"]
    cap_a, cap_b = (cap["A"], cap["B"]) if isinstance(cap, dict) else (cap, cap)
    dem = cfg["demand"]
    mode = dem.get("mode", EXACT)
    start = cfg.get("start_price", "max")
    name = cfg.get("name", path.stem)
    times = cfg.get("times")

    if dem["type"] == "mnl":
        alpha = dem["alpha"]
        if "arrivals" in dem:
            arrivals = dem["arrivals"]
            n = cfg.get("periods") or (len(times) - 1 if times else None)
            if np.isscalar(arrivals):
                if n is None:
                    raise ValueError("укажите periods или times, если arrivals — одно число")
                arrivals = [arrivals] * n
        elif "arrival_rate" in dem:
            if times is None:
                raise ValueError("arrival_rate задаётся вместе с times")
            arrivals = float(dem["arrival_rate"]) * np.diff(np.asarray(times, dtype=float))
        else:
            raise ValueError("demand: нужен arrivals (покупателей за период) или arrival_rate + times")
        return mnl_scenario(prices, cap_a, cap_b, arrivals=arrivals,
                            alpha_a=alpha["A"], alpha_b=alpha["B"], beta=dem["beta"],
                            mode=mode, start_price=start, times=times, name=name)

    if dem["type"] == "table":
        data = np.load(path.parent / dem["file"])
        d_a, d_b = data["demand_a"], data["demand_b"]
        mono_a = data["mono_a"] if "mono_a" in data else None
        mono_b = data["mono_b"] if "mono_b" in data else None
        mono_a, mono_b = default_mono(d_a, d_b, mono_a, mono_b)
        return Scenario(prices, cap_a, cap_b, d_a, d_b, mono_a, mono_b,
                        mode=mode, start_price=start, times=times, name=name)

    raise ValueError(f"demand.type: 'mnl' или 'table', получено {dem['type']!r}")


def default_mono(demand_a, demand_b, mono_a=None, mono_b=None):
    """Если спрос «без конкурента» не задан, берём спрос при цене конкурента
    на самом высоком уровне — ближайшее к «цене +∞», что есть в таблице."""
    if mono_a is None:
        mono_a = np.asarray(demand_a)[:, :, -1]
    if mono_b is None:
        mono_b = np.asarray(demand_b)[:, -1, :]
    return mono_a, mono_b
