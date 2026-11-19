# Как запускать проект

## 1. Установка зависимостей

Из корня репозитория (рекомендуется в виртуальном окружении `.venv`):

```powershell
pip install -r requirements.txt
pip install -e .
```

*(Если не делать `pip install -e .`, то при запуске `python -m pricing ...` задайте переменную окружения: `$env:PYTHONPATH="src"` в PowerShell или `PYTHONPATH=src` в bash.)*

---

## 2. Основные команды

### Расчёт равновесия (обратная индукция)
```powershell
# Точный спрос (выведет ожидаемую выручку и равновесную траекторию цен)
python -m pricing experiments/base.json

# Случайный спрос (Пуассон) + проверка методом Монте-Карло на 20 000 партиях
python -m pricing experiments/base_poisson.json --mc 20000

# Сохранить таблицы стратегий в .npz или .csv
python -m pricing experiments/base.json --save results/base.npz --csv results/base.csv
```

### Круговой турнир: ДП против 8 эвристик
```powershell
# На базовом сценарии (точный спрос)
python experiments/tournament.py

# На случайном спросе (усреднение по 200 партиям Монте-Карло на каждую пару)
python experiments/tournament.py experiments/base_poisson.json --mc 200
```

### Генерация примера таблиц спроса (`.npz`)
```powershell
python experiments/make_table_example.py
```

### Запуск тестов
```powershell
python -m pytest -q
```

---

## 3. Использование из Python

```python
from pricing import (
    ALL_BASELINES,
    format_path,
    load_scenario,
    monte_carlo,
    play,
    solve,
)

sc = load_scenario("experiments/base.json")
sol = solve(sc)

# Ожидаемая выручка и равновесная траектория ДП vs ДП
print(sol.revenue())
print(format_path(play(sol)))

# Матч ДП против одной из эвристик (например, Undercut)
steps = play(sol, ALL_BASELINES["Undercut"], scenario=sc)
print(format_path(steps))
```

Подробнее:
- [solver.md](solver.md) — устройство решателя обратной индукцией, формат JSON-сценария и проверка корректности.
- [heuristics.md](heuristics.md) — описание 8 эвристик, правила турнира и добавление своих стратегий.
