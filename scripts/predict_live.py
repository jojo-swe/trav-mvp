#!/usr/bin/env python3
"""Hämta nästa V75/V85 och skriv en föreslagen vinnarrad.

Personlig analys — inte finansiell rådgivning. Placera inga spel automatiskt.
ATG förbjuder robotspel (automatiserad kupongoptimering mot andra).

Kör från repots rot:
    python scripts/predict_live.py
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import urllib.error
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fetch_atg_sample import (  # noqa: E402
    HISTORY_CSV,
    LIVE_DIR,
    STOCKHOLM,
    apply_history,
    discover_vgame,
    fetch_game,
    history_from_csv,
    rows_from_game,
    save_live_payloads,
    stockholm_today,
)
from mvp_core import fit_full, load_starters, run_mvp, score_starters  # noqa: E402

DISCLAIMER = (
    "Personlig analys, inte finansiell rådgivning och inte speltips. "
    "Placera inga spel automatiskt. ATG förbjuder robotspel."
)
# Value-häst tas med i garderingen om den är en annan häst och minst så här
# mycket bättre value än spiken. Jämförelsen är relativ inom avdelningen.
VALUE_EDGE = 1.20


def _num(value, digits: int = 6):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, digits)


def _fmt(value, spec: str, missing: str = "–") -> str:
    number = _num(value)
    if number is None:
        return missing
    return format(number, spec)


def _horse_bits(row: pd.Series) -> dict:
    return {
        "start_number": int(row["start_number"]),
        "horse_name": row["horse_name"],
        "trainer_name": row.get("trainer_name") or "",
        "model_p": _num(row["model_p"], 4),
        "market_p": _num(row["market_p"], 4),
        "value": _num(row["value"], 3),
        "trainer_change": None if pd.isna(row.get("trainer_change")) else int(row["trainer_change"]),
        "distance_form": _num(row.get("distance_form"), 3),
        "prior_starts": None if pd.isna(row.get("prior_starts")) else int(row["prior_starts"]),
        "distance_starts_prior": (
            None
            if pd.isna(row.get("distance_starts_prior"))
            else int(row["distance_starts_prior"])
        ),
        "distance_form_source": row.get("distance_form_source") or "",
    }


def _rank(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    work = frame.copy()
    work["_key"] = pd.to_numeric(work[column], errors="coerce")
    return work.sort_values(
        ["_key", "model_p"], ascending=[False, False], na_position="last"
    ).drop(columns="_key")


def _coupon(numbers: list[list[int]]) -> str:
    parts = []
    for group in numbers:
        parts.append("/".join(str(number) for number in group))
    return "-".join(parts)


def build_suggestion(scored: pd.DataFrame) -> dict:
    """Enkelrad, reserver och ett litet favorit+value-system."""
    legs = []
    spik_numbers: list[list[int]] = []
    value_numbers: list[int] = []
    market_numbers: list[int] = []
    system_groups: list[list[int]] = []

    order = (
        scored[["leg", "race_id"]]
        .drop_duplicates()
        .sort_values("leg")
    )
    for _, key in order.iterrows():
        group = scored[scored["race_id"] == key["race_id"]].copy()
        by_model = _rank(group, "model_p")
        by_value = _rank(group, "value")
        by_market = _rank(group, "market_p")
        spik = by_model.iloc[0]
        reserves = by_model.iloc[1:3]
        contenders = by_model.head(3)
        value_pick = _rank(contenders, "value").iloc[0]
        market_pick = by_market.iloc[0]
        spik_value = _num(spik["value"], 6) or 0.0
        value_value = _num(value_pick["value"], 6)
        gardera = (
            int(value_pick["start_number"]) != int(spik["start_number"])
            and value_value is not None
            and spik_value > 0
            and value_value >= spik_value * VALUE_EDGE
        )
        second = by_model.iloc[1] if len(by_model) > 1 else None
        spik_p = float(spik["model_p"])
        second_p = float(second["model_p"]) if second is not None else 0.0
        bank = second is None or (second_p > 0 and spik_p >= second_p * 1.75) or (
            second_p == 0 and spik_p > 0
        )
        group_numbers = [int(spik["start_number"])]
        if gardera:
            group_numbers.append(int(value_pick["start_number"]))
        spik_numbers.append([int(spik["start_number"])])
        value_numbers.append(int(value_pick["start_number"]))
        market_numbers.append(int(market_pick["start_number"]))
        system_groups.append(group_numbers)
        head = group.iloc[0]
        legs.append(
            {
                "leg": int(key["leg"]),
                "race_id": head["race_id"],
                "race_name": head.get("race_name") or "",
                "track_name": head.get("track_name") or "",
                "distance": _num(head.get("distance"), 0),
                "start_method": head.get("start_method") or "",
                "field_size": int(head["field_size"]),
                "bank": bool(bank),
                "gardering": gardera,
                "spik": _horse_bits(spik),
                "reserves": [_horse_bits(row) for _, row in reserves.iterrows()],
                "value_pick": _horse_bits(value_pick),
                "market_pick": _horse_bits(market_pick),
                "by_model_p": [_horse_bits(row) for _, row in by_model.head(5).iterrows()],
                "by_value": [_horse_bits(row) for _, row in by_value.head(3).iterrows()],
            }
        )

    expanded = list(product(*system_groups))
    return {
        "legs": legs,
        "enkelrad": _coupon(spik_numbers),
        "value_rad": "-".join(str(number) for number in value_numbers),
        "marknad_rad": "-".join(str(number) for number in market_numbers),
        "system_notation": _coupon(system_groups),
        "system_rows": ["-".join(str(number) for number in row) for row in expanded],
        "n_system_rows": len(expanded),
    }


def rank_candidate_rows(scored: pd.DataFrame, max_rows: int = 50, per_leg: int = 4) -> dict:
    """Full ranked lista över kandidatrader, bäst till sämst — inte bara en kollapsad rad.

    Tar de `per_leg` högst rankade (på model_p) hästarna per avdelning och
    rankar ALLA kombinationer av dem bäst-till-sämst på produkten av
    model_p över avdelningarna (via summa av log-sannolikheter, vilket ger
    samma ordning utan att behöva räkna ut varje produkt separat).

    Att räkna ut *varenda* kombination av samtliga startande (utan att
    begränsa per avdelning) är kombinatoriskt omöjligt för ett V85 — redan
    11*12*11*15*12*12*... blir fort hundratals miljoner rader. Vi begränsar
    därför till de `per_leg` mest sannolika per avdelning (konfigurerbart)
    och rankar den fulla korsprodukten av DEM, bäst till sämst, med en
    heap-baserad k-bästa-sökning (samma princip som "k smallest sums från
    N sorterade listor") så att vi slipper bygga hela korsprodukten i
    minnet när den är stor.
    """
    order = scored[["leg", "race_id"]].drop_duplicates().sort_values("leg")
    leg_candidates: list[list[tuple[int, float, str]]] = []
    for _, key in order.iterrows():
        group = scored[scored["race_id"] == key["race_id"]].copy()
        ranked_group = _rank(group, "model_p").head(max(per_leg, 1))
        cands = [
            (int(row["start_number"]), float(row["model_p"]), str(row["horse_name"]))
            for _, row in ranked_group.iterrows()
        ]
        leg_candidates.append(cands)

    n_legs = len(leg_candidates)
    total_space = 1
    for cands in leg_candidates:
        total_space *= max(len(cands), 1)

    if n_legs == 0 or any(not cands for cands in leg_candidates):
        return {
            "rows": [],
            "candidates_per_leg": per_leg,
            "total_candidate_space": 0,
            "n_rows_returned": 0,
        }

    logs = [[math.log(max(p, 1e-9)) for _, p, _ in cands] for cands in leg_candidates]
    start = tuple([0] * n_legs)
    visited = {start}
    heap: list[tuple[float, tuple[int, ...]]] = [
        (-sum(logs[i][0] for i in range(n_legs)), start)
    ]
    want = min(max_rows, total_space)
    results = []
    while heap and len(results) < want:
        neg_score, idxs = heapq.heappop(heap)
        score = -neg_score
        numbers = [leg_candidates[i][idxs[i]][0] for i in range(n_legs)]
        names = [leg_candidates[i][idxs[i]][2] for i in range(n_legs)]
        results.append(
            {
                "numbers": numbers,
                "coupon": "-".join(str(n) for n in numbers),
                "horse_names": names,
                "combined_probability": math.exp(score),
                "log_score": score,
            }
        )
        for leg in range(n_legs):
            if idxs[leg] + 1 < len(leg_candidates[leg]):
                nxt = list(idxs)
                nxt[leg] += 1
                nxt_t = tuple(nxt)
                if nxt_t not in visited:
                    visited.add(nxt_t)
                    s = sum(logs[i][nxt_t[i]] for i in range(n_legs))
                    heapq.heappush(heap, (-s, nxt_t))
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank
    return {
        "rows": results,
        "candidates_per_leg": per_leg,
        "total_candidate_space": total_space,
        "n_rows_returned": len(results),
    }


def render_report(
    *,
    hit_date: str,
    game_type: str,
    game_id: str,
    status: str,
    start_time: str | None,
    track: str,
    metrics: dict,
    n_full: int,
    n_hist_horses: int,
    n_distance_form: int,
    n_starters: int,
    suggestion: dict,
    ranked: dict,
    paths: dict[str, Path],
) -> str:
    lines = [
        DISCLAIMER,
        "",
        f"Spel: {game_type}  {game_id}",
        f"Dag: {hit_date}   Status: {status or '–'}   Bana: {track or '–'}",
        f"Start: {start_time or '–'}   Avdelningar: {len(suggestion['legs'])}",
        (
            f"Modell: HistGradientBoosting tränad på {n_full} historiska starter "
            f"({HISTORY_CSV.name})."
        ),
        (
            f"Tidsplit-test (före {metrics.get('cut_date')} / från {metrics.get('cut_date')}): "
            f"log-loss {_fmt(metrics.get('log_loss'), '.4f')}, "
            f"Brier {_fmt(metrics.get('brier'), '.4f')}, "
            f"AUC {_fmt(metrics.get('roc_auc'), '.4f')}, "
            f"top-1 modell {metrics.get('top1_model_count')}/{metrics.get('n_races')} "
            f"= {_fmt(metrics.get('top1_model'), '.1%')}, "
            f"marknadsfavorit {metrics.get('top1_market_count')}/{metrics.get('n_races')} "
            f"= {_fmt(metrics.get('top1_market'), '.1%')}."
        ),
        (
            f"Historik mot live-startlistan: {n_hist_horses}/{n_starters} "
            f"({(n_hist_horses / n_starters) if n_starters else 0:.1%}) har minst en tidigare start. "
            f"Distansform från tidigare starter i samma klass: {n_distance_form}/{n_starters} "
            f"({(n_distance_form / n_starters) if n_starters else 0:.1%}). "
            "Saknas tidigare start blir tränarbyte=0. Saknas samma distansklass används rekordet i klassen."
        ),
        "marknad = normaliserad 1/vinnarodds (historiskt stängningsodds, live är aktuellt odds). value = model_p / marknad.",
        "Personlig analys. Inga kuponger skickas och inga spel läggs.",
        "",
        "════════════════════════════════════════",
        "ENKEL VINNARRAD  (högst model_p per avdelning)",
        "════════════════════════════════════════",
        suggestion["enkelrad"],
    ]
    for leg in suggestion["legs"]:
        spik = leg["spik"]
        role = "spik" if leg["bank"] else "öppen"
        lines.append(
            f"Avd {leg['leg']}  #{spik['start_number']} {spik['horse_name']}  "
            f"model_p={spik['model_p']:.3f}  marknad={_fmt(spik['market_p'], '.3f')}  "
            f"value={_fmt(spik['value'], '.2f')}  "
            f"tränarbyte={spik['trainer_change']}  "
            f"distansform={_fmt(spik['distance_form'], '.3f')}  "
            f"tidigare={spik.get('prior_starts')}  "
            f"distansstarter={spik.get('distance_starts_prior')}  ({role})"
        )

    lines += [
        "",
        "════════════════════════════════════════",
        "PER AVDELNING",
        "════════════════════════════════════════",
    ]
    for leg in suggestion["legs"]:
        distance = leg["distance"]
        distance_s = str(int(distance)) if distance is not None else "–"
        lines.append(
            f"Avd {leg['leg']}  {leg['track_name']}  {distance_s} m  "
            f"{leg['start_method']}  {leg['field_size']} startande"
        )
        if leg["race_name"]:
            lines.append(f"  {leg['race_name']}")
        lines.append("  Rankad på model_p")
        for rank, horse in enumerate(leg["by_model_p"], start=1):
            lines.append(_line_from_bits(rank, horse))
        lines.append("  Rankad på value")
        for rank, horse in enumerate(leg["by_value"], start=1):
            lines.append(_line_from_bits(rank, horse))
        reserves = leg["reserves"]
        if reserves:
            names = ", ".join(
                f"#{horse['start_number']} {horse['horse_name']}" for horse in reserves
            )
            lines.append(f"  Reserv: {names}")
        lines.append("")

    lines += [
        "════════════════════════════════════════",
        "RANKADE RADER  (bäst till sämst, inte bara en kollapsad rad)",
        "════════════════════════════════════════",
        (
            f"Topp {ranked['candidates_per_leg']} hästar (på model_p) per avdelning, "
            f"alla kombinationer av dem rankade på kombinerad sannolikhet. "
            f"Fullt korsutrymme med den begränsningen: {ranked['total_candidate_space']:,} rader. "
            f"Visar {ranked['n_rows_returned']} högst rankade."
        ),
    ]
    for row in ranked["rows"][:30]:
        lines.append(
            f"  {row['rank']:>3}. {row['coupon']:<24} "
            f"P(hela raden)~={row['combined_probability']:.6f}"
        )
    if ranked["n_rows_returned"] > 30:
        lines.append(f"  … {ranked['n_rows_returned'] - 30} fler rader i suggested_rows.json (ranked_rows).")

    lines += [
        "",
        "════════════════════════════════════════",
        "SYSTEM  (favoriter + value, legacy — se RANKADE RADER ovan för huvudresultatet)",
        "════════════════════════════════════════",
        f"Rad A  modellfavorit:  {suggestion['enkelrad']}",
        (
            f"Rad B  value (bästa value bland topp-3 model_p):  "
            f"{suggestion['value_rad']}"
        ),
        f"Rad C  spelfavorit (marknad):  {suggestion['marknad_rad']}",
        (
            f"Gardering: value-hästen läggs till där den skiljer sig och value är "
            f">= {VALUE_EDGE:.2f} × spikens value."
        ),
        f"System: {suggestion['system_notation']}   ({suggestion['n_system_rows']} rader)",
    ]
    rows = suggestion["system_rows"]
    if suggestion["n_system_rows"] <= 16:
        for index, row in enumerate(rows, start=1):
            lines.append(f"  {index:>2}. {row}")
    else:
        lines.append("  För många rader att lista var för sig — använd notationen ovan.")
    lines += [
        "",
        "Filer:",
        f"  {paths['starters']}",
        f"  {paths['game']}",
        f"  {paths['calendar']}",
        f"  {paths['text']}",
        f"  {paths['json']}",
        "",
        DISCLAIMER,
    ]
    return "\n".join(lines) + "\n"


def _line_from_bits(rank: int, horse: dict) -> str:
    change = horse["trainer_change"]
    change_s = "–" if change is None else str(change)
    return (
        f"  {rank:>2}. #{horse['start_number']:<2} {str(horse['horse_name']):<22} "
        f"model_p={_fmt(horse['model_p'], '.3f')}  "
        f"marknad={_fmt(horse['market_p'], '.3f')}  "
        f"value={_fmt(horse['value'], '.2f')}  "
        f"tränarbyte={change_s}  "
        f"distansform={_fmt(horse['distance_form'], '.3f')}  "
        f"tidigare={horse.get('prior_starts')}  "
        f"distansstarter={horse.get('distance_starts_prior')}"
    )


def render_no_game(start: str, days: int, discovered) -> str:
    lines = [
        DISCLAIMER,
        "",
        f"Ingen spelbar V75/V85 med publicerad startlista mellan {start} och {days - 1} dagar framåt.",
    ]
    if discovered.placeholders:
        first = discovered.placeholders[0]
        lines.append(
            f"Närmaste platshållare: {first['game_type']} {first['date']} "
            f"(status {first['status']}, inget spel-id ännu)."
        )
    if discovered.errors:
        lines.append("Kalenderfel:")
        lines.extend(f"  {err}" for err in discovered.errors)
    lines += ["", DISCLAIMER]
    return "\n".join(lines) + "\n"


def _parse_date(text: str):
    return datetime.strptime(text, "%Y-%m-%d").date()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hämta nästa V75/V85 och skriv en föreslagen vinnarrad. Lägger inga spel."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Antal dagar framåt från startdatum, inklusive idag (standard 14).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Startdatum YYYY-MM-DD. Standard är idag i Europe/Stockholm.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help="Max antal rankade rader att beräkna/visa (standard 50).",
    )
    parser.add_argument(
        "--candidates-per-leg",
        type=int,
        default=4,
        help="Hur många topphästar (på model_p) per avdelning som ingår i rankningen (standard 4).",
    )
    args = parser.parse_args(argv)
    if args.days < 1:
        print("days måste vara minst 1", file=sys.stderr)
        return 2

    start = _parse_date(args.date) if args.date else stockholm_today()
    print(
        f"Söker V75/V85 från {start.isoformat()} och {args.days - 1} dagar framåt "
        "(Europe/Stockholm). Hämtar kalender med paus mellan anrop…"
    )
    discovered = discover_vgame(start, days=args.days)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    text_path = LIVE_DIR / "suggested_rows.txt"
    json_path = LIVE_DIR / "suggested_rows.json"

    if discovered.hit is None:
        report = render_no_game(start.isoformat(), args.days, discovered)
        text_path.write_text(report, encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {
                    "disclaimer": DISCLAIMER,
                    "found": False,
                    "start": start.isoformat(),
                    "days": args.days,
                    "placeholders": discovered.placeholders,
                    "errors": discovered.errors,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(report, end="")
        return 0

    hit = discovered.hit
    print(f"Hittade {hit.game_type} {hit.game_id} ({hit.status}). Hämtar spelomgång…")
    try:
        game = fetch_game(hit.game_id)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        print(f"Kunde inte hämta {hit.game_id}: {exc}", file=sys.stderr)
        return 1

    rows = rows_from_game(game, hit.game_type, require_results=False)
    if not rows:
        print(f"{hit.game_id} saknar starter.", file=sys.stderr)
        return 1
    if not HISTORY_CSV.exists():
        print(
            f"Saknar historik: {HISTORY_CSV}. Kör först "
            "python scripts/fetch_atg_sample.py --expand --from-date 2025-10-10 "
            "--to-date 2026-10-09 --game-id V85_2026-10-10_5_6",
            file=sys.stderr,
        )
        return 1

    history = history_from_csv(HISTORY_CSV)
    apply_history(rows, extra_events=history)
    paths = save_live_payloads(hit, game, rows)
    paths["text"] = text_path
    paths["json"] = json_path

    print("Tränar modell på historiska starter…")
    evaluation = run_mvp(HISTORY_CSV)
    historical = load_starters(HISTORY_CSV)
    fitted = fit_full(historical)
    scored = score_starters(fitted, pd.DataFrame(rows))
    suggestion = build_suggestion(scored)
    ranked = rank_candidate_rows(scored, max_rows=args.max_rows, per_leg=args.candidates_per_leg)
    track = next((row.get("track_name") for row in rows if row.get("track_name")), "")
    n_hist = sum(1 for row in rows if (row.get("prior_starts") or 0) > 0)
    n_dist = sum(1 for row in rows if (row.get("distance_starts_prior") or 0) > 0)
    report = render_report(
        hit_date=hit.date,
        game_type=hit.game_type,
        game_id=hit.game_id,
        status=game.get("status") or hit.status,
        start_time=hit.start_time,
        track=track,
        metrics=evaluation.metrics,
        n_full=fitted.n_rows,
        n_hist_horses=n_hist,
        n_distance_form=n_dist,
        n_starters=len(rows),
        suggestion=suggestion,
        ranked=ranked,
        paths=paths,
    )
    text_path.write_text(report, encoding="utf-8")
    payload = {
        "disclaimer": DISCLAIMER,
        "found": True,
        "generated_at": datetime.now(STOCKHOLM).isoformat(timespec="seconds"),
        "game_type": hit.game_type,
        "game_id": hit.game_id,
        "date": hit.date,
        "status": game.get("status") or hit.status,
        "start_time": hit.start_time,
        "track_name": track,
        "trained_on_rows": fitted.n_rows,
        "evaluation": evaluation.metrics,
        "starters_with_history": n_hist,
        "starters_with_distance_form": n_dist,
        "starters": len(rows),
        "history_coverage": (n_hist / len(rows)) if rows else None,
        "distance_form_coverage": (n_dist / len(rows)) if rows else None,
        "history_csv": str(HISTORY_CSV),
        "enkelrad": suggestion["enkelrad"],
        "value_rad": suggestion["value_rad"],
        "marknad_rad": suggestion["marknad_rad"],
        "system_notation": suggestion["system_notation"],
        "n_system_rows": suggestion["n_system_rows"],
        "system_rows": suggestion["system_rows"],
        "ranked_rows": ranked["rows"],
        "ranked_rows_meta": {
            "candidates_per_leg": ranked["candidates_per_leg"],
            "total_candidate_space": ranked["total_candidate_space"],
            "n_rows_returned": ranked["n_rows_returned"],
        },
        "legs": suggestion["legs"],
        "files": {name: str(path) for name, path in paths.items()},
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
