#!/usr/bin/env python3
"""Hämta V75/V85-starter från ATG:s publika racinginfo-API.

Källa: https://www.atg.se/services/racinginfo/v1/api/
Inofficiellt dokumenterat — respektera ATG:s villkor. Minst SLEEP_S sekunder
mellan anrop. Lägger inga spel.

Historiskt stickprov: `python scripts/fetch_atg_sample.py`
Kommande omgång: `python scripts/predict_live.py` (använder samma parser).
"""
from __future__ import annotations

import argparse
import csv
import http.client
import json
import math
import sys
import time
import urllib.error
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
# Primär träningsfil: alla avslutade svenska travlopp i hämtningsfönstret.
HISTORY_CSV = ROOT / "data" / "atg_starters.csv"
HISTORY_DIR = ROOT / "data" / "history"
HISTORY_JSONL = HISTORY_DIR / "starters.jsonl"
DONE_RACES = HISTORY_DIR / "done_races.txt"
SEEN_HORSES = HISTORY_DIR / "seen_horses.txt"
TARGET_HORSES = HISTORY_DIR / "target_horses.txt"
LIVE_DIR = ROOT / "data" / "live"
STOCKHOLM = ZoneInfo("Europe/Stockholm")
API_HOST = "www.atg.se"
API_PREFIX = "/services/racinginfo/v1/api/"

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
_conn: http.client.HTTPSConnection | None = None


class HttpStatus(urllib.error.URLError):
    """HTTP-fel från racinginfo. Är en URLError så befintliga anropare fångar det."""

    def __init__(self, status: int, path: str):
        super().__init__(f"HTTP {status} {path}")
        self.status = status


def stockholm_today() -> date:
    return datetime.now(STOCKHOLM).date()


def _connection() -> http.client.HTTPSConnection:
    global _conn
    if _conn is None:
        _conn = http.client.HTTPSConnection(API_HOST, timeout=45)
    return _conn


def _drop_connection() -> None:
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except OSError:
            pass
        _conn = None


def get(path: str, retries: int = 4):
    """GET mot racinginfo. Minst SLEEP_S mellan anrop; TLS-kopplingen återanvänds.

    429/5xx försöks igen med paus. 404 lyfts direkt.
    """
    global _last_request
    rel = path.lstrip("/")
    url_path = API_PREFIX + rel
    last_error: Exception | None = None
    for attempt in range(retries):
        wait = SLEEP_S - (time.monotonic() - _last_request)
        if _last_request and wait > 0:
            time.sleep(wait)
        conn = _connection()
        try:
            conn.request(
                "GET",
                url_path,
                headers={**UA, "Accept": "application/json", "Connection": "keep-alive"},
            )
            response = conn.getresponse()
            body = response.read()
        except (http.client.HTTPException, TimeoutError, OSError, json.JSONDecodeError) as exc:
            _last_request = time.monotonic()
            _drop_connection()
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(8.0, 1.0 * (attempt + 1)))
                continue
            raise
        _last_request = time.monotonic()
        status = response.status
        if status == 200:
            try:
                return json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                _drop_connection()
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(1.0)
                    continue
                raise
        error = HttpStatus(status, rel)
        _drop_connection()
        if status == 404:
            raise error
        last_error = error
        if status in {429, 500, 502, 503, 504} and attempt + 1 < retries:
            time.sleep(min(12.0, 2.0 * (attempt + 1)))
            continue
        raise error
    if last_error:
        raise last_error
    raise urllib.error.URLError(f"GET misslyckades: {rel}")


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


def _race_sort_key(race_id: str, row_date: str) -> tuple:
    """Kronologisk nyckel. Loppnumret är inte nollutfyllt i race_id."""
    parts = str(race_id or "").split("_")
    track = int(parts[-2]) if len(parts) >= 3 and str(parts[-2]).isdigit() else 0
    number = int(parts[-1]) if len(parts) >= 3 and str(parts[-1]).isdigit() else 0
    return (str(row_date or "")[:10], track, number, str(race_id or ""))


def _event_sort_key(item: tuple) -> tuple:
    race_id = item[6] if len(item) > 6 else ""
    return _race_sort_key(race_id, item[0])


def _event_from_row(row: dict) -> tuple:
    return (
        str(row.get("date") or "")[:10],
        norm_id(row.get("trainer_id")),
        row.get("distance"),
        row.get("distance_class"),
        _place(row.get("place")),
        row.get("final_odds"),
        str(row.get("race_id") or ""),
        norm_id(row.get("driver_id")),
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


def _add_event(history: dict[str, list[tuple]], seen: set[tuple], horse_id, event: tuple) -> None:
    hid = norm_id(horse_id)
    if not hid:
        return
    race_id = event[6] if len(event) > 6 else ""
    # Utan race_id (äldre tester) behålls varje rad; med race_id en gång per lopp.
    marker = race_id or f"row-{len(seen)}"
    key = (hid, event[0], marker)
    if key in seen:
        return
    seen.add(key)
    history[hid].append(event)


def apply_history(rows: list[dict], extra_events: dict[str, list[tuple]] | None = None) -> None:
    """Sätt trainer_change, distance_form och recent_form från tidigare starter.

    extra_events används för live-starter: historiken kommer från den sparade
    CSV:n, inte från omgången som ska tippas. Samma dag räknas ett tidigare
    race_id som tidigare start. Distansform från starter sätter
    distance_form_source=starts; annars ligger rekord-fallbacken kvar.
    """
    history: dict[str, list[tuple]] = defaultdict(list)
    seen: set[tuple] = set()
    if extra_events:
        for horse_id, events in extra_events.items():
            for event in events:
                _add_event(history, seen, horse_id, event)
    for row in rows:
        _add_event(history, seen, row.get("horse_id"), _event_from_row(row))
    history_sorted = {
        horse_id: sorted(events, key=_event_sort_key)
        for horse_id, events in history.items()
        if horse_id
    }

    def prior(horse_id, row_date, race_id):
        key = norm_id(horse_id)
        current = _race_sort_key(str(race_id or ""), str(row_date or "")[:10])
        return [item for item in history_sorted.get(key or "", []) if _event_sort_key(item) < current]

    for row in rows:
        prev = prior(row.get("horse_id"), row.get("date"), row.get("race_id"))
        row["prior_starts"] = len(prev)
        if prev:
            last_trainer = prev[-1][1]
            current = norm_id(row.get("trainer_id"))
            row["trainer_change"] = (
                1 if last_trainer and current and last_trainer != current else 0
            )
            last_driver = prev[-1][7] if len(prev[-1]) > 7 else None
            current_driver = norm_id(row.get("driver_id"))
            row["driver_change"] = (
                1 if last_driver and current_driver and last_driver != current_driver else 0
            )
            same = [item for item in prev if item[3] == row.get("distance_class")][-5:]
            same_places = [item[4] for item in same if item[4]]
            if same_places:
                row["distance_form"] = round(
                    sum(1.0 / place for place in same_places) / len(same_places), 4
                )
                row["distance_starts_prior"] = len(same_places)
                row["distance_form_source"] = "starts"
            else:
                row["distance_starts_prior"] = 0
                row["distance_form_source"] = "record" if row.get("distance_form") else "none"
            last5_places = [item[4] for item in prev[-5:] if item[4]]
            row["recent_form"] = (
                round(sum(1.0 / place for place in last5_places) / len(last5_places), 4)
                if last5_places
                else None
            )
        else:
            row["trainer_change"] = 0
            row["driver_change"] = 0
            row["distance_starts_prior"] = 0
            row["recent_form"] = None
            row["distance_form_source"] = "record" if row.get("distance_form") else "none"


PREFERRED_FIELDS = [
    "date",
    "leg",
    "game_type",
    "game_id",
    "game_status",
    "race_id",
    "race_name",
    "race_status",
    "track_id",
    "track_name",
    "track_condition",
    "distance",
    "distance_class",
    "start_method",
    "field_size",
    "start_number",
    "post_position",
    "horse_id",
    "horse_name",
    "horse_age",
    "horse_sex",
    "trainer_id",
    "trainer_name",
    "trainer_change",
    "driver_id",
    "driver_name",
    "driver_change",
    "place",
    "won",
    "final_odds",
    "market_implied_prob",
    "market_share_pool",
    "distance_form",
    "distance_form_source",
    "distance_starts_prior",
    "prior_starts",
    "recent_form",
    "life_win_pct",
    "life_place_pct",
    "life_starts",
    "avg_odds_last5",
    "shoe_front",
    "shoe_back",
    "shoe_changed",
    "sulky_type",
    "sulky_changed",
    "trainer_win_pct_ytd",
    "driver_win_pct_ytd",
    "vinnare_odds_raw",
    "bet_distribution_raw",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    present = {key for row in rows for key in row}
    fields = [key for key in PREFERRED_FIELDS if key in present]
    fields.extend(sorted(present - set(fields)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
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


def rows_from_race(race: dict) -> list[dict]:
    """Platta ut ett ensamt loppsvar (samma fält som en spelomgång med ett lopp)."""
    payload = race
    if not payload.get("date"):
        rid = str(payload.get("id") or "")
        if len(rid) >= 10 and rid[4] == "-" and rid[7] == "-":
            payload = dict(payload)
            payload["date"] = rid[:10]
    game_id = payload.get("id")
    return rows_from_game(
        {
            "id": game_id,
            "type": "race",
            "status": payload.get("status"),
            "races": [payload],
        },
        "race",
        require_results=True,
    )


def swedish_result_race_ids(calendar: dict) -> list[str]:
    """Avslutade travlopp på svenska banor den dagen. Galopp och utland hoppas över."""
    ids: list[str] = []
    for track in calendar.get("tracks") or []:
        if (track.get("countryCode") or "").upper() != "SE":
            continue
        sport = track.get("sport") or "trot"
        if sport != "trot":
            continue
        for race in track.get("races") or []:
            status = (race.get("status") or "").lower()
            if status != "results":
                continue
            race_id = race.get("id")
            if race_id:
                ids.append(str(race_id))
    return ids


def _read_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _append_id(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(value + "\n")


def load_history_rows(path: Path | None = None) -> list[dict]:
    """Läs jsonl-checkpointen och slå ihop dubbletter (race_id, horse_id)."""
    jsonl = Path(path) if path else HISTORY_JSONL
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    if not jsonl.exists():
        return rows
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (str(row.get("race_id") or ""), norm_id(row.get("horse_id")) or "")
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def append_history_rows(rows: list[dict]) -> None:
    if not rows:
        return
    HISTORY_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_JSONL.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def finalize_history(jsonl: Path | None = None, out: Path | None = None) -> int:
    """Bygg tränings-CSV med tränarbyte och distansform från hela checkpointen."""
    rows = load_history_rows(jsonl)
    rows.sort(key=lambda row: (str(row.get("date") or ""), str(row.get("race_id") or ""), row.get("start_number") or 0))
    apply_history(rows)
    target = Path(out) if out else HISTORY_CSV
    write_csv(target, rows)
    horses = {norm_id(row.get("horse_id")) for row in rows if norm_id(row.get("horse_id"))}
    print(f"skrev {target} rader={len(rows)} hästar={len(horses)}", flush=True)
    return len(rows)


def race_ids_from_horse(
    payload: dict,
    before: str,
    limit: int = 8,
    skip: set[str] | None = None,
) -> list[str]:
    """Tidigare lopp-id från hästens resultatlista, nyast först.

    Redan hämtade lopp (skip) räknas inte mot limit, så vi kan gå bakom fönstret.
    """
    records = ((payload.get("results") or {}).get("records")) or []
    ids: list[str] = []
    cutoff = str(before)[:10]
    skip = skip or set()
    for record in records:
        day = str(record.get("date") or "")[:10]
        if not day or day >= cutoff:
            continue
        if record.get("scratched"):
            continue
        race = record.get("race") or {}
        if race.get("sport") and race["sport"] != "trot":
            continue
        race_id = race.get("id")
        if not race_id or str(race_id) in skip or str(race_id) in ids:
            continue
        ids.append(str(race_id))
        if len(ids) >= limit:
            break
    return ids


def _fetch_race_rows(race_id: str, *, swedish_only: bool = True) -> list[dict]:
    race = get(f"races/{race_id}")
    if (race.get("sport") or "trot") != "trot":
        return []
    status = (race.get("status") or "").lower()
    if status in {"cancelled", "deleted", "abandoned", "upcoming", "scheduled"}:
        return []
    track = race.get("track") or {}
    if swedish_only and (track.get("countryCode") or "SE").upper() != "SE":
        return []
    return rows_from_race(race)


def _remember_horses(rows: list[dict], seen: set[str]) -> None:
    for row in rows:
        horse_id = norm_id(row.get("horse_id"))
        if horse_id and horse_id not in seen:
            seen.add(horse_id)
            _append_id(SEEN_HORSES, horse_id)


def fetch_race_into_history(
    race_id: str,
    done: set[str],
    seen: set[str],
    *,
    swedish_only: bool = True,
) -> int:
    """Hämta ett lopp om det inte redan ligger i checkpointen. Returnerar nya rader."""
    if race_id in done:
        return 0
    try:
        rows = _fetch_race_rows(race_id, swedish_only=swedish_only)
    except HttpStatus as exc:
        if exc.status == 404:
            done.add(race_id)
            _append_id(DONE_RACES, race_id)
            return 0
        print(f"  race fail {race_id}: {exc}", flush=True)
        return 0
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        print(f"  race fail {race_id}: {exc}", flush=True)
        return 0
    append_history_rows(rows)
    _remember_horses(rows, seen)
    done.add(race_id)
    _append_id(DONE_RACES, race_id)
    return len(rows)


def iter_history_days(start: date, end: date):
    """Dagar från nyast till äldst, inklusive båda ändar."""
    current = end
    while current >= start:
        yield current
        current -= timedelta(days=1)


def expand_history(start: date, end: date, target: set[str] | None = None) -> None:
    """Hämta avslutade svenska travlopp i [start, end] och skriv checkpoint.

    Rå-JSON sparas inte. Starter appendas till data/history/starters.jsonl.
    Avbruten körning fortsätter via done_races.txt.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    done = _read_ids(DONE_RACES)
    seen = _read_ids(SEEN_HORSES)
    target = target or set()
    print(
        f"Hämtar svenska travresultat {start.isoformat()} … {end.isoformat()} "
        f"(klart sedan tidigare: {len(done)} lopp, {len(seen)} hästar). "
        f"Minst {SLEEP_S}s mellan anrop.",
        flush=True,
    )
    for day in iter_history_days(start, end):
        day_s = day.isoformat()
        try:
            calendar = get(f"calendar/day/{day_s}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            print(f"{day_s} kalender fel: {exc}", flush=True)
            continue
        race_ids = swedish_result_race_ids(calendar)
        new_rows = 0
        fetched = 0
        for race_id in race_ids:
            if race_id in done:
                continue
            added = fetch_race_into_history(race_id, done, seen)
            fetched += 1
            new_rows += added
        hit = len(target & seen) if target else 0
        coverage = (hit / len(target)) if target else 0.0
        extra = f" täckning={hit}/{len(target)}={coverage:.1%}" if target else ""
        print(
            f"{day_s} lopp={len(race_ids)} nya={fetched} rader+={new_rows} "
            f"hästar={len(seen)}{extra}",
            flush=True,
        )


def supplement_horses(horse_ids: list[str], before: str, per_horse: int = 8) -> int:
    """För hästar som saknas i fönstret: häst-endpoint, sedan deras senaste lopp."""
    done = _read_ids(DONE_RACES)
    seen = _read_ids(SEEN_HORSES)
    added = 0
    print(f"Kompletterar {len(horse_ids)} hästar via horses/{{id}} (max {per_horse} lopp).", flush=True)
    for horse_id in horse_ids:
        try:
            payload = get(f"horses/{horse_id}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            print(f"  häst {horse_id} fel: {exc}", flush=True)
            continue
        for race_id in race_ids_from_horse(payload, before, limit=per_horse, skip=done):
            added += fetch_race_into_history(race_id, done, seen, swedish_only=False)
    return added


def parse_cli_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def target_horses_for_game(game_id: str) -> tuple[set[str], str, list[dict]]:
    """Startande häst-id i en omgång. Raderna ingår inte i träningshistoriken."""
    game = fetch_game(game_id)
    gtype = game.get("type") or game_id.split("_", 1)[0]
    rows = rows_from_game(game, gtype, require_results=False)
    horses = {norm_id(row.get("horse_id")) for row in rows if norm_id(row.get("horse_id"))}
    if rows and rows[0].get("date"):
        game_date = str(rows[0]["date"])[:10]
    else:
        parts = game_id.split("_")
        game_date = parts[1] if len(parts) >= 2 else ""
    if not game_date and "_" in game_id:
        parts = game_id.split("_")
        if len(parts) >= 2:
            game_date = parts[1]
    return horses, game_date, rows


def weak_horse_ids(live_rows: list[dict], history_rows: list[dict]) -> list[str]:
    """Hästar utan tidigare start, eller utan distansform från starter i samma klass."""
    copied = [dict(row) for row in live_rows]
    apply_history(copied, extra_events=history_from_rows(history_rows))
    weak: list[str] = []
    for row in copied:
        horse_id = norm_id(row.get("horse_id"))
        if not horse_id:
            continue
        if row.get("prior_starts") and row.get("distance_starts_prior"):
            continue
        weak.append(horse_id)
    return list(dict.fromkeys(weak))


def history_from_rows(rows: list[dict]) -> dict[str, list[tuple]]:
    events: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        horse_id = norm_id(row.get("horse_id"))
        if not horse_id:
            continue
        events[horse_id].append(_event_from_row(row))
    return events


def expand_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Hämta avslutade svenska travlopp från ATG racinginfo till "
            "data/atg_starters.csv. Lägger inga spel."
        )
    )
    parser.add_argument("--expand", action="store_true", help="Bygg den utökade historiken.")
    parser.add_argument("--from-date", default=None, help="Första dag YYYY-MM-DD (inklusive).")
    parser.add_argument("--to-date", default=None, help="Sista dag YYYY-MM-DD (inklusive).")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Om --from-date saknas: så många dagar bakåt från --to-date (standard 365).",
    )
    parser.add_argument(
        "--game-id",
        default=None,
        help="Omgång vars startande hästar styr kompletteringen, t.ex. V85_2026-10-10_5_6.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Bygg bara CSV från data/history/starters.jsonl.",
    )
    parser.add_argument(
        "--no-supplement",
        action="store_true",
        help="Hämta inte häst-endpoints för hästar som saknar distansform.",
    )
    args = parser.parse_args(argv)
    if args.finalize_only:
        finalize_history()
        return 0

    target: set[str] = set()
    live_rows: list[dict] = []
    game_date: str | None = None
    if args.game_id:
        print(f"Hämtar målomgång {args.game_id} för hästlistan…", flush=True)
        target, game_date, live_rows = target_horses_for_game(args.game_id)
        TARGET_HORSES.parent.mkdir(parents=True, exist_ok=True)
        TARGET_HORSES.write_text("\n".join(sorted(target)) + "\n", encoding="utf-8")
        print(f"  {len(target)} startande, tävlingsdag {game_date}", flush=True)

    if args.to_date:
        end = parse_cli_date(args.to_date)
    elif game_date:
        end = parse_cli_date(game_date) - timedelta(days=1)
    else:
        end = stockholm_today() - timedelta(days=1)
    if args.from_date:
        start = parse_cli_date(args.from_date)
    else:
        start = end - timedelta(days=max(args.lookback_days, 1) - 1)
    if start > end:
        print("--from-date ligger efter --to-date", file=sys.stderr)
        return 2

    expand_history(start, end, target)

    if target and live_rows and not args.no_supplement:
        history_rows = load_history_rows()
        weak = weak_horse_ids(live_rows, history_rows)
        before = game_date or (end + timedelta(days=1)).isoformat()
        print(
            f"Hästar utan tidigare start eller utan distansform från starter: {len(weak)}",
            flush=True,
        )
        if weak:
            supplement_horses(weak, before, per_horse=12)
            history_rows = load_history_rows()
            still = weak_horse_ids(live_rows, history_rows)
            print(f"Efter komplettering saknar fortfarande {len(still)} hästar start eller distansform.", flush=True)
            if still:
                supplement_horses(still, before, per_horse=20)

    finalize_history()
    return 0


if __name__ == "__main__":
    try:
        if "--expand" in sys.argv or "--finalize-only" in sys.argv:
            raise SystemExit(expand_main(sys.argv[1:]))
        main()
    except KeyboardInterrupt:
        sys.exit(130)
