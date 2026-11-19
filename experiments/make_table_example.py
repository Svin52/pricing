"""Пример, как подготовить свои таблицы спроса для режима "table".

Здесь таблицы для наглядности считаются по MNL, но их можно заполнить чем угодно:
demand_a[k, iA, iB] — сколько человек купят у A в периоде k при уровнях цен (iA, iB),
demand_b[k, iA, iB] — то же для B, mono_a[k, iA] / mono_b[k, iB] — спрос, когда конкурент
распродан (необязательно; если не задать, берётся спрос при цене конкурента на высшем уровне).

Запуск из корня репозитория:  python experiments/make_table_example.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from pricing import mnl_tables  # noqa: E402

prices = np.linspace(3000, 14500, 24)
# спрос растёт к вылету, а чувствительность к цене падает (ближе к вылету летят те, кому надо)
arrivals = np.linspace(4, 12, 20)
beta = np.linspace(0.0006, 0.0003, 20)
d_a, d_b, m_a, m_b = mnl_tables(prices, arrivals, alpha_a=4.5, alpha_b=4.2, beta=beta, mode="exact")
out = pathlib.Path(__file__).with_name("table_example.npz")
np.savez(out, demand_a=d_a, demand_b=d_b, mono_a=m_a, mono_b=m_b)
print(f"сохранено: {out}, форма demand_a = {d_a.shape}")
