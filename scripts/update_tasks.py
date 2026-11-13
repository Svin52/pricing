"""Пересобирает трекер задач в README.md из GitHub Issues.

Открытые issues -> «Активные задачи», закрытые как completed -> «История».
Закрытые как not planned (то есть удалённые из трекера) и удалённые issues не показываются.
"""
import json
import os
import re
import sys
import urllib.request
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

README = "README.md"
START, END = "<!-- TASKS:START -->", "<!-- TASKS:END -->"
TZ = ZoneInfo(os.environ.get("TASKS_TZ", "Europe/Moscow"))
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
          "августа", "сентября", "октября", "ноября", "декабря"]


def fetch_issues(repo, token):
    issues, page = [], 1
    while True:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/issues?state=all&per_page=100&page={page}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req) as r:
            batch = json.load(r)
        issues += [i for i in batch if "pull_request" not in i]
        if len(batch) < 100:
            return issues
        page += 1


DATE_RE = (r"(?:(?P<y1>\d{4})[-./](?P<m1>\d{1,2})[-./](?P<d1>\d{1,2})"
           r"|(?P<d2>\d{1,2})[-./](?P<m2>\d{1,2})(?:[-./](?P<y2>\d{4}|\d{2}))?)"
           r"(?:[ ,T]+(?:в\s*)?(?P<hh>\d{1,2}):(?P<mm>\d{2}))?")


def parse_date(s, today=None):
    """Дата в начале строки: 2026-11-14, 14-11-2026, 14.11.2026, 14/11/26, 14.11 (текущий год), можно со временем.
    Возвращает ((date, time|None), длина совпадения) или None."""
    m = re.match(r"\s*" + DATE_RE, s or "")
    if not m:
        return None
    g = m.groupdict()
    if g["y1"]:
        y, mo, dd = int(g["y1"]), int(g["m1"]), int(g["d1"])
    else:
        dd, mo = int(g["d2"]), int(g["m2"])
        y = g["y2"]
        if not y and len(g["m2"]) != 2:  # «1.5 часа» — не дата
            return None
        y = (today or date.today()).year if not y else int(y) + (2000 if len(y) == 2 else 0)
    try:
        d = date(y, mo, dd)
        t = time(int(g["hh"]), int(g["mm"])) if g["hh"] else None
    except ValueError:
        return None
    return (d, t), m.end()


def parse_deadline(body):
    """Поле «Дедлайн» из шаблона или строка «Дедлайн: ...» в тексте issue."""
    if not body:
        return None
    m = re.search(r"###\s*Дедлайн\s*\n+([^\n#]*)", body) or re.search(r"Дедлайн\s*:\s*([^\n]*)", body, re.I)
    r = parse_date(m.group(1)) if m else None
    return r[0] if r else None


def split_title(title):
    """«14-11-2026 сделать что-то» -> (дедлайн, «сделать что-то»). Дата должна стоять в начале названия."""
    r = parse_date(title)
    if not r:
        return None, title
    rest = title[r[1]:]
    if rest and not re.match(r"[\s:—–\-|.,)]", rest):  # дата должна быть отдельным словом
        return None, title
    rest = re.sub(r"^[\s:—–\-|.,)]+", "", rest).strip()
    return (r[0], rest or title)


def plural(n, one, few, many):
    w = one if n % 10 == 1 and n % 100 != 11 else few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many
    return f"{n} {w}"


def fmt_date(d, t=None, today=None):
    s = f"{d.day} {MONTHS[d.month - 1]}"
    if today is None or d.year != today.year:
        s += f" {d.year}"
    if t:
        s += f", {t:%H:%M}"
    return s


def due_moment(dl):
    d, t = dl
    return datetime.combine(d, t or time(23, 59, 59), TZ)


def status(dl, now):
    d, _ = dl
    days = (d - now.date()).days
    if due_moment(dl) < now:
        return "🔴 просрочено сегодня" if days == 0 else f"🔴 просрочено на {plural(-days, 'день', 'дня', 'дней')}"
    if days == 0:
        return "🟠 сегодня"
    if days == 1:
        return "🟠 завтра"
    if days <= 3:
        return f"🟡 через {plural(days, 'день', 'дня', 'дней')}"
    return f"🟢 через {plural(days, 'день', 'дня', 'дней')}"


def esc(s):
    return s.replace("|", "\\|").replace("\n", " ").strip()


def build(issues, repo, now):
    today = now.date()
    active, done = [], []
    for i in issues:
        if "pull_request" in i:
            continue
        dl, i["title"] = split_title(i["title"])
        dl = dl or parse_deadline(i.get("body"))
        if i["state"] == "open":
            active.append((i, dl))
        elif i.get("state_reason") == "completed":
            done.append((i, dl))
    active.sort(key=lambda x: (x[1] is None, due_moment(x[1]) if x[1] else 0, x[0]["number"]))
    done.sort(key=lambda x: x[0]["closed_at"], reverse=True)

    new = f"https://github.com/{repo}/issues/new?template=task.yml"
    out = [START, "",
           f"**[➕ Новая задача]({new})** · [все задачи](https://github.com/{repo}/issues)", "",
           f"### Активные задачи ({len(active)})", ""]
    if active:
        out += ["| Задача | Дедлайн | Статус |", "|---|---|---|"]
        for i, dl in active:
            out.append(f"| [{esc(i['title'])}]({i['html_url']}) | "
                       f"{fmt_date(*dl, today=today) if dl else '—'} | {status(dl, now) if dl else 'без дедлайна'} |")
    else:
        out.append("_Активных задач нет._")
    out += ["", f"### История выполненных ({len(done)})", ""]
    if done:
        out += ["| Задача | Выполнено | Дедлайн |", "|---|---|---|"]
        for i, dl in done:
            closed = datetime.fromisoformat(i["closed_at"].replace("Z", "+00:00")).astimezone(TZ)
            if dl:
                mark = "✅ в срок" if closed <= due_moment(dl) else "⚠️ с опозданием"
                dl_txt = f"{fmt_date(*dl, today=today)} · {mark}"
            else:
                dl_txt = "—"
            out.append(f"| ~~[{esc(i['title'])}]({i['html_url']})~~ | {fmt_date(closed.date(), today=today)} | {dl_txt} |")
    else:
        out.append("_Пока ничего не выполнено._")
    out += ["", END]
    return "\n".join(out)


def apply_event(issues, event_path):
    """Список issues в API обновляется с задержкой в несколько секунд, поэтому только что созданная
    или изменённая задача может в него не попасть. Берём её актуальную версию из самого события."""
    if not event_path or not os.path.exists(event_path):
        return issues
    event = json.load(open(event_path, encoding="utf-8"))
    issue = event.get("issue")
    if not issue:
        return issues
    rest = [i for i in issues if i["number"] != issue["number"]]
    return rest if event.get("action") in ("deleted", "transferred") else rest + [issue]


def main():
    repo, token = os.environ["REPO"], os.environ["GITHUB_TOKEN"]
    issues = json.load(open(os.environ["ISSUES_JSON"])) if os.environ.get("ISSUES_JSON") else fetch_issues(repo, token)
    issues = apply_event(issues, os.environ.get("GITHUB_EVENT_PATH"))
    now = datetime.now(TZ)
    block = build(issues, repo, now)
    text = open(README, encoding="utf-8").read()
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n## Трекер задач\n\n" + block + "\n"
    if text == open(README, encoding="utf-8").read():
        print("Задачи не изменились")
        return
    open(README, "w", encoding="utf-8").write(text)
    print("README обновлён")


if __name__ == "__main__":
    sys.exit(main())
