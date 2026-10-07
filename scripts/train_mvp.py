#!/usr/bin/env python3
"""MVP: Gradient boosting win-probability for Swedish harness (V75/V85).

Obligatoriska faktorer:
  1) trainer_change  – tränarbyte vs senaste starter i stickprovet
  2) distance_form   – form betingad på distansklass (short/medium/long)
  3) market_*        – marknadspris (implicerad sannolikhet / poolandel)

Datakälla: data/atg_starters.csv (ATG racinginfo, alla avslutade svenska travlopp
i hämtningsfönstret). Bygg om med:
  python scripts/fetch_atg_sample.py --expand --from-date 2025-10-10 --to-date 2026-10-09 --game-id V85_2026-10-10_5_6
"""
from __future__ import annotations

from mvp_core import FEATURES, run_mvp


def main():
    result = run_mvp()
    m = result.metrics
    print("=== Trav MVP (HistGradientBoosting) ===")
    print(f"Data: {m['data_path']}")
    print(f"Starter totalt: {m['n_total']} | train: {m['n_train']} | test: {m['n_test']}")
    if m.get("prior_start_rate") is not None:
        print(
            f"Rader med tidigare start: {m.get('rows_with_prior')} "
            f"({m['prior_start_rate']:.1%}) | "
            f"distansform från starter: {m.get('distance_form_starts_rate', 0):.1%} | "
            f"tränarbyte: {m.get('trainer_change_rate', 0):.1%}"
        )
    print(f"Train-datum < {m['cut_date']} | test >= {m['cut_date']}")
    print(f"Features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print()
    print(f"Log-loss:     {m['log_loss']:.4f}")
    print(f"Brier:        {m['brier']:.4f}")
    if m["roc_auc"] is not None:
        print(f"ROC-AUC:      {m['roc_auc']:.4f}")
    else:
        print("ROC-AUC:      n/a")
    print(
        f"Accuracy@0.5: {m['accuracy_0_5']:.4f}  "
        "(svagt meningsfullt vid ~9% basrate)"
    )

    if m["n_races"]:
        print(
            f"Top-1 win acc (per lopp): "
            f"{m['top1_model_count']}/{m['n_races']} = {m['top1_model']:.3f}"
        )
        print(
            f"Market favorite top-1:    "
            f"{m['top1_market_count']}/{m['n_races']} = {m['top1_market']:.3f}"
        )

    last_race = result.df_te["race_id"].iloc[-1]
    g = result.df_te[result.df_te["race_id"] == last_race].copy()
    g = g.sort_values("model_p", ascending=False)
    print()
    print(
        f"--- Exempel toppickar: {last_race} "
        f"({g['track_name'].iloc[0]}, {g['date'].iloc[0].date()}) ---"
    )
    cols = [
        "start_number",
        "horse_name",
        "trainer_change",
        "distance_form",
        "market_implied_prob",
        "model_p",
        "place",
        "final_odds",
    ]
    print(g[cols].head(5).to_string(index=False))
    print()
    print(
        "Tak för MVP: trav är brusigt; sikta på kalibrerade sannolikheter + value vs pool,"
    )
    print(
        "inte 'tipsa vinnaren varje gång'. Produktion: mer historik, Benter-blend, V-system-scoring."
    )


if __name__ == "__main__":
    main()
