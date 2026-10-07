#!/usr/bin/env python3
"""Hämta V75/V85-starter från ATG:s publika racinginfo-API.

Källa: https://www.atg.se/services/racinginfo/v1/api/
Inofficiellt dokumenterat — respektera ATG:s villkor. Minst SLEEP_S sekunder
mellan anrop. Lägger inga spel.

Historiskt stickprov: `python scripts/fetch_atg_sample.py`
Kommande omgång: `python scripts/predict_live.py` (använder samma parser).
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = "https://www.atg.se/services/racinginfo/v1/api/"
UA = {"User-Agent": "trav-mvp/0.1 (personal research)"}
SLEEP_S = 0.4
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "atg_starters_sample.csv"
LIVE_DIR = ROOT / "data" / "live"
STOCKHOLM = ZoneInfo("Europe/Stockholm")

# Stickprovet tar första spelade V-omgången den dagen, i den här ordningen.
SAMPLE_GAME_TYPES = ("V85", "V75", "V86", "V64", "V65")
# Live-raden är bara lördagens stora spel.
LIVE_GAME_TYPES = ("V75", "V85")
# Inte spelbart: redan kört eller inställt. Övriga statusar med id provas.
SKIP_STATUSES = {"cancelled", "results", "deleted", "finished", "abandoned", "final"}

DATES = [
    "2025-10-25", "2025-11-01", "2025-11-08", "2025-11-15", "2025-11-22",
    "2025-12-06", "2025-12-13", "2025-12-20",
    "2024-06-15", "2024-06-22", "2024-06-29", "2024-07-06", "2024-07-13",
    "2024-08-03", "2024-08-10", "2024-09-07", "2024-09-14",
    "2024-10-05", "2024-10-12", "2024-11-02",
]

_last_request = 0.0


def stockholm_today() -> date:
    return datetime.now(STOCKHOLM).date()


def get(path: str):
    """GET mot racinginfo. Väntar så att det är minst SLEEP_S mellan anrop."""
    global _last_request
    wait = SLEEP_S - (time.monotonic() - _last_request)
    if _last_request and wait > 0:
        time.sleep(wait)
    url = BASE + path.lstrip("/")
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    finally:
        _last_request = time.monotonic()


def distance_class(dist):
    if dist is None:
        return "unk"
    if dist < 1800:
        return "short"
    if dist < 2300:
        return "medium"
    return "long"


def person_name(person: dict | None) -> str:
    person = person or {}
    return f"{person.get('firstName', '')} {person.get('lastName', '')}".strip()


def year_win_pct(person: dict | None, year: int) -> float:
    """Segerandel för tävlingsåret, annars senaste tidigare år i payloaden."""
    years = ((person or {}).get("statistics") or {}).get("years") or {}
    if not isinstance(years, dict):
        return 0.0
    block = years.get(str(year))
    if not isinstance(block, dict):
        earlier = []
        for key, val in years.items():
            if isinstance(val, dict) and str(key).isdigit() and int(key) <= year:
                earlier.append((int(key), val))
        block = max(earlier)[1] if earlier else {}
    return (block.get("winPercentage") or 0) / 10000.0


def norm_id(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return str(int(value))
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _place(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return int(number) if number == int(number) else number


def _event_from_row(row: dict) -> tuple:
    return (
        str(row.get("date") or "")[:10],
        norm_id(row.get("trainer_id")),
        row.get("distance"),
        row.get("distance_class"),
        _place(row.get("place")),
        row.get("final_odds"),
    )


def is_scratched(start: dict) -> bool:
    """Struken innan start. `out` är något annat: hästen startade men galopperade/diskades."""
    if start.get("scratched") is True:
        return True
    result = start.get("result") or {}
    return bool(result.get("scratched"))


def active_starts(race: dict, *, require_results: bool) -> list[dict]:
    starts = []
    for start in race.get("starts") or []:
        if is_scratched(start):
            continue
        if not (start.get("horse") or {}).get("id"):
            continue
        result = start.get("result") or {}
        if require_results:
            if result.get("place") is None:
                continue
        elif start.get("out") is True:
            continue
        starts.append(start)
    return starts


def rows_from_game(
    game: dict,
    gtype: str | None = None,
    *,
    require_results: bool = True,
) -> list[dict]:
    """Platta ut en spelomgång till starter-rader.

    require_results=True hoppar över strukna/utan placering och lopp med
    färre än 5 i mål (samma filter som det historiska stickprovet).
    require_results=False behåller kommande starter som fortfarande kan spelas.
    """
    gtype = gtype or game.get("type") or ""
    game_id = game.get("id")
    rows: list[dict] = []
    for leg, race in enumerate(game.get("races") or [], start=1):
        if race.get("sport") and race["sport"] != "trot":
            continue
        status = (race.get("status") or "").lower()
        if status in {"cancelled", "deleted", "abandoned"}:
            continue
        track = race.get("track") or {}
        dist = race.get("distance")
        dclass = distance_class(dist)
        field = active_starts(race, require_results=require_results)
        if require_results and len(field) < 5:
            continue
        if not field:
            continue
        odds_list = []
        for start in field:
            pools = start.get("pools") or {}
            vinnare = pools.get("vinnare") or {}
            odds = vinnare.get("finalOdds") or vinnare.get("odds")
            if odds:
                odds_list.append(odds / 100.0)
            else:
                odds_list.append((start.get("result") or {}).get("finalOdds"))
        inv = [1.0 / odds if odds and odds > 1 else None for odds in odds_list]
        inv_sum = sum(x for x in inv if x)
        race_date = race.get("date")
        try:
            race_year = int(str(race_date)[:4])
        except (TypeError, ValueError):
            race_year = stockholm_today().year
        for start, odds, impi in zip(field, odds_list, inv):
            horse = start.get("horse") or {}
            driver = start.get("driver") or {}
            trainer = horse.get("trainer") or {}
            result = start.get("result") or {}
            pools = start.get("pools") or {}
            vdist = (
                pools.get("V85") or pools.get("V75") or pools.get(gtype) or {}
            ).get("betDistribution")
            market_share = None
            if isinstance(vdist, int):
                market_share = vdist / 10000.0 if vdist > 100 else vdist / 100.0
            implied = (impi / inv_sum) if impi and inv_sum else None
            if market_share is None and implied is not None:
                market_share = implied
            stats = horse.get("statistics") or {}
            life = stats.get("life") or {}
            last5 = stats.get("lastFiveStarts") or {}
            records = life.get("records") or []
            dist_rec = [rec for rec in records if rec.get("distance") == dclass]
            places = [rec.get("place") for rec in dist_rec if rec.get("place")]
            dist_form = (1.0 / min(places)) if places else None
            shoes = horse.get("shoes") or {}
            sulky = (horse.get("sulky") or {}).get("type") or {}
            place = result.get("place")
            rows.append(
                {
                    "date": race_date,
                    "leg": leg,
                    "game_type": gtype,
                    "game_id": game_id,
                    "game_status": game.get("status"),
                    "race_id": race.get("id"),
                    "race_name": race.get("name"),
                    "race_status": race.get("status"),
                    "track_id": track.get("id"),
                    "track_name": track.get("name"),
                    "track_condition": track.get("condition"),
                    "distance": dist,
                    "distance_class": dclass,
                    "start_method": race.get("startMethod"),
                    "field_size": len(field),
                    "start_number": start.get("number"),
                    "post_position": start.get("postPosition"),
                    "horse_id": horse.get("id"),
                    "horse_name": horse.get("name"),
                    "horse_age": horse.get("age"),
                    "horse_sex": horse.get("sex"),
                    "trainer_id": trainer.get("id"),
                    "trainer_name": person_name(trainer),
                    "driver_id": driver.get("id"),
                    "driver_name": person_name(driver),
                    "place": place,
                    "won": 1 if place == 1 else (0 if place is not None else None),
                    "final_odds": odds,
                    "market_implied_prob": round(implied, 6) if implied else None,
                    "market_share_pool": round(market_share, 6) if market_share else None,
                    "vinnare_odds_raw": (pools.get("vinnare") or {}).get("odds"),
                    "bet_distribution_raw": vdist,
                    "life_starts": life.get("starts") or 0,
                    "life_win_pct": (life.get("winPercentage") or 0) / 10000.0,
                    "life_place_pct": (life.get("placePercentage") or 0) / 10000.0,
                    "avg_odds_last5": (last5.get("averageOdds") or 0) / 100.0
                    if last5.get("averageOdds")
                    else None,
                    "distance_form": dist_form,
                    "shoe_front": (shoes.get("front") or {}).get("hasShoe"),
                    "shoe_back": (shoes.get("back") or {}).get("hasShoe"),
                    "shoe_changed": bool(
                        (shoes.get("front") or {}).get("changed")
                        or (shoes.get("back") or {}).get("changed")
                    ),
                    "sulky_type": sulky.get("code"),
                    "sulky_changed": sulky.get("changed"),
                    "trainer_win_pct_ytd": year_win_pct(trainer, race_year),
                    "driver_win_pct_ytd": year_win_pct(driver, race_year),
                }
            )
    return rows


def history_from_csv(path: Path | None = None) -> dict[str, list[tuple]]:
    """Tidigare starter i stickprovet, nyckel = horse_id."""
    csv_path = Path(path) if path else OUT
    events: dict[str, list[tuple]] = defaultdict(list)
    if not csv_path.exists():
        return events
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            horse_id = norm_id(row.get("horse_id"))
            if not horse_id:
                continue
            events[horse_id].append(_event_from_row(row))
    return events


def apply_history(rows: list[dict], extra_events: dict[str, list[tuple]] | None = None) -> None:
    """Sätt trainer_change, distance_form och recent_form från tidigare starter.

    extra_events används för live-starter: historiken kommer från stickprovet,
    inte från omgången som ska tippas.
    """
    history: dict[str, list[tuple]] = defaultdict(list)
    if extra_events:
        for horse_id, events in extra_events.items():
            history[norm_id(horse_id) or ""].extend(events)
    for row in rows:
        horse_id = norm_id(row.get("horse_id"))
        if horse_id:
            history[horse_id].append(_event_from_row(row))
    history_sorted = {
        horse_id: sorted(events, key=lambda item: item[0])
        for horse_id, events in history.items()
        if horse_id
    }

    def prior(horse_id, row_date):
        key = norm_id(horse_id)
        return [item for item in history_sorted.get(key or "", []) if item[0] < str(row_date)[:10]]

    for row in rows:
        prev = prior(row.get("horse_id"), row.get("date"))
        if prev:
            last_trainer = prev[-1][1]
            current = norm_id(row.get("trainer_id"))
            row["trainer_change"] = (
                1 if last_trainer and current and last_trainer != current else 0
            )
            same = [item for item in prev if item[3] == row.get("distance_class")][-5:]
            same_places = [item[4] for item in same if item[4]]
            if same_places:
                row["distance_form"] = round(
                    sum(1.0 / place for place in same_places) / len(same_places), 4
                )
                row["distance_starts_prior"] = len(same_places)
            else:
                row["distance_starts_prior"] = 0
            last5_places = [item[4] for item in prev[-5:] if item[4]]
            row["recent_form"] = (
                round(sum(1.0 / place for place in last5_places) / len(last5_places), 4)
                if last5_places
                else None
            )
        else:
            row["trainer_change"] = 0
            row["distance_starts_prior"] = 0
            row["recent_form"] = None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def pick_sample_game(calendar: dict) -> tuple[str | None, str | None]:
    games = calendar.get("games") or {}
    for game_type in SAMPLE_GAME_TYPES:
        arr = games.get(game_type) or []
        if arr:
            return arr[0]["id"], game_type
    return None, None


@dataclass
class VGameHit:
    game_type: str
    game_id: str
    date: str
    status: str
    start_time: str | None
    calendar: dict


@dataclass
class DiscoverResult:
    hit: VGameHit | None = None
    placeholders: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def discover_vgame(
    start: date | None = None,
    days: int = 14,
    game_types: tuple[str, ...] = LIVE_GAME_TYPES,
) -> DiscoverResult:
    """Första spelbara V75/V85 i [start, start+days) enligt kalendern.

    Idag vinner om den omgången fortfarande är öppen (inte results/cancelled).
    Platshållare utan id (status scheduled) sparas men väljs inte.
    """
    start = start or stockholm_today()
    result = DiscoverResult()
    for offset in range(max(days, 0)):
        day = start + timedelta(days=offset)
        day_s = day.isoformat()
        print(f"  kalender {day_s}", flush=True)
        try:
            calendar = get(f"calendar/day/{day_s}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            result.errors.append(f"{day_s}: {exc}")
            continue
        games = calendar.get("games") or {}
        found: list[dict] = []
        for game_type in game_types:
            for game in games.get(game_type) or []:
                status = (game.get("status") or "").lower()
                game_id = game.get("id")
                if status in SKIP_STATUSES:
                    continue
                if not game_id:
                    result.placeholders.append(
                        {
                            "date": day_s,
                            "game_type": game_type,
                            "status": status or "scheduled",
                        }
                    )
                    continue
                found.append(
                    {
                        "game_type": game_type,
                        "game_id": game_id,
                        "status": status,
                        "start_time": game.get("startTime") or game.get("scheduledStartTime"),
                    }
                )
        if not found:
            continue
        found.sort(
            key=lambda item: (
                game_types.index(item["game_type"]) if item["game_type"] in game_types else 99,
                item.get("start_time") or "",
            )
        )
        chosen = found[0]
        result.hit = VGameHit(
            game_type=chosen["game_type"],
            game_id=chosen["game_id"],
            date=day_s,
            status=chosen["status"],
            start_time=chosen.get("start_time"),
            calendar=calendar,
        )
        return result
    return result


def fetch_game(game_id: str) -> dict:
    return get(f"games/{game_id}")


def save_live_payloads(hit: VGameHit, game: dict, rows: list[dict]) -> dict[str, Path]:
    """Spara kalender, spel-JSON och starter-CSV under data/live/."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = hit.game_id.replace("/", "_")
    paths = {
        "calendar": LIVE_DIR / f"calendar_{hit.date}.json",
        "game": LIVE_DIR / f"game_{safe_id}.json",
        "starters": LIVE_DIR / f"starters_{safe_id}.csv",
    }
    write_json(paths["calendar"], hit.calendar)
    write_json(paths["game"], game)
    write_csv(paths["starters"], rows)
    return paths


def main():
    rows: list[dict] = []
    for day in DATES:
        try:
            calendar = get(f"calendar/day/{day}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            print("calendar fail", day, exc)
            continue
        game_id, game_type = pick_sample_game(calendar)
        if not game_id:
            print(day, "no V-game")
            continue
        game = get(f"games/{game_id}")
        print(day, game_type, game_id)
        rows.extend(rows_from_game(game, game_type, require_results=True))
    apply_history(rows)
    write_csv(OUT, rows)
    print(f"wrote {OUT} rows={len(rows)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
