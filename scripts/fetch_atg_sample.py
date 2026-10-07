#!/usr/bin/env python3
"""Hämta ett stickprov av V75/V85-starter från ATG:s publika racinginfo-API.

Källa: https://www.atg.se/services/racinginfo/v1/api/
Inofficiellt/odemokratiskt dokumenterat — respektera ATG:s villkor vid bulk.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE = "https://www.atg.se/services/racinginfo/v1/api/"
UA = {"User-Agent": "trav-mvp/0.1 (personal research)"}
OUT = Path(__file__).resolve().parents[1] / "data" / "atg_starters_sample.csv"

DATES = [
    "2025-10-25", "2025-11-01", "2025-11-08", "2025-11-15", "2025-11-22",
    "2025-12-06", "2025-12-13", "2025-12-20",
    "2024-06-15", "2024-06-22", "2024-06-29", "2024-07-06", "2024-07-13",
    "2024-08-03", "2024-08-10", "2024-09-07", "2024-09-14",
    "2024-10-05", "2024-10-12", "2024-11-02",
]


def get(path: str):
    req = urllib.request.Request(BASE + path, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def distance_class(dist):
    if dist is None:
        return "unk"
    if dist < 1800:
        return "short"
    if dist < 2300:
        return "medium"
    return "long"


def main():
    rows = []
    history = defaultdict(list)
    for d in DATES:
        try:
            cal = get(f"calendar/day/{d}")
        except Exception as e:
            print("calendar fail", d, e)
            continue
        games = cal.get("games", {})
        game_id = gtype = None
        for gt in ("V85", "V75", "V86", "V64", "V65"):
            arr = games.get(gt) or []
            if arr:
                game_id, gtype = arr[0]["id"], gt
                break
        if not game_id:
            print(d, "no V-game")
            continue
        game = get(f"games/{game_id}")
        print(d, gtype, game_id)
        for race in game.get("races", []):
            if race.get("sport") and race["sport"] != "trot":
                continue
            track = race.get("track", {})
            dist = race.get("distance")
            dclass = distance_class(dist)
            field = [
                s for s in race.get("starts", [])
                if s.get("result") and s["result"].get("place") is not None
            ]
            if len(field) < 5:
                continue
            odds_list = []
            for s in field:
                pools = s.get("pools") or {}
                v = pools.get("vinnare") or {}
                o = v.get("finalOdds") or v.get("odds")
                if o:
                    odds_list.append(o / 100.0)
                else:
                    odds_list.append(s.get("result", {}).get("finalOdds"))
            inv = [1.0 / o if o and o > 1 else None for o in odds_list]
            s_inv = sum(x for x in inv if x)
            for s, o, impi in zip(field, odds_list, inv):
                h = s.get("horse") or {}
                dr = s.get("driver") or {}
                tr = h.get("trainer") or {}
                res = s.get("result") or {}
                pools = s.get("pools") or {}
                vdist = (
                    pools.get("V85") or pools.get("V75") or pools.get(gtype) or {}
                ).get("betDistribution")
                market_share = None
                if isinstance(vdist, int):
                    market_share = vdist / 10000.0 if vdist > 100 else vdist / 100.0
                implied = (impi / s_inv) if impi and s_inv else None
                if market_share is None and implied is not None:
                    market_share = implied
                stats = h.get("statistics") or {}
                life = stats.get("life") or {}
                last5 = stats.get("lastFiveStarts") or {}
                records = life.get("records") or []
                dist_rec = [r for r in records if r.get("distance") == dclass]
                places = [r.get("place") for r in dist_rec if r.get("place")]
                dist_form = (1.0 / min(places)) if places else None
                shoes = h.get("shoes") or {}
                sulky = (h.get("sulky") or {}).get("type") or {}
                row = {
                    "date": race.get("date"),
                    "game_type": gtype,
                    "game_id": game_id,
                    "race_id": race.get("id"),
                    "track_id": track.get("id"),
                    "track_name": track.get("name"),
                    "track_condition": track.get("condition"),
                    "distance": dist,
                    "distance_class": dclass,
                    "start_method": race.get("startMethod"),
                    "field_size": len(field),
                    "start_number": s.get("number"),
                    "post_position": s.get("postPosition"),
                    "horse_id": h.get("id"),
                    "horse_name": h.get("name"),
                    "horse_age": h.get("age"),
                    "horse_sex": h.get("sex"),
                    "trainer_id": tr.get("id"),
                    "trainer_name": f"{tr.get('firstName', '')} {tr.get('lastName', '')}".strip(),
                    "driver_id": dr.get("id"),
                    "driver_name": f"{dr.get('firstName', '')} {dr.get('lastName', '')}".strip(),
                    "place": res.get("place"),
                    "won": 1 if res.get("place") == 1 else 0,
                    "final_odds": o,
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
                    "trainer_win_pct_ytd": (
                        (tr.get("statistics") or {}).get("years") or {}
                    )
                    .get("2025", {})
                    .get("winPercentage", 0)
                    / 10000.0,
                    "driver_win_pct_ytd": (
                        (dr.get("statistics") or {}).get("years") or {}
                    )
                    .get("2025", {})
                    .get("winPercentage", 0)
                    / 10000.0,
                }
                rows.append(row)
                history[h.get("id")].append(
                    (race.get("date"), tr.get("id"), dist, dclass, res.get("place"), o)
                )
        time.sleep(0.15)

    history_sorted = {hid: sorted(v, key=lambda x: x[0]) for hid, v in history.items()}

    def prior(hid, date):
        return [x for x in history_sorted.get(hid, []) if x[0] < date]

    for r in rows:
        prev = prior(r["horse_id"], r["date"])
        if prev:
            last_tr = prev[-1][1]
            r["trainer_change"] = (
                1 if last_tr and r["trainer_id"] and last_tr != r["trainer_id"] else 0
            )
            same = [x for x in prev if x[3] == r["distance_class"]][-5:]
            same_places = [x[4] for x in same if x[4]]
            if same_places:
                r["distance_form"] = round(
                    sum(1.0 / p for p in same_places) / len(same_places), 4
                )
                r["distance_starts_prior"] = len(same_places)
            else:
                r["distance_starts_prior"] = 0
            last5p = [x[4] for x in prev[-5:] if x[4]]
            r["recent_form"] = (
                round(sum(1.0 / p for p in last5p) / len(last5p), 4) if last5p else None
            )
        else:
            r["trainer_change"] = 0
            r["distance_starts_prior"] = 0
            r["recent_form"] = None

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} rows={len(rows)}")


if __name__ == "__main__":
    main()
