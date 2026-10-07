#!/usr/bin/env python3
"""MVP: Gradient boosting win-probability for Swedish harness (V75/V85).

Obligatoriska faktorer:
  1) trainer_change  – tränarbyte vs senaste starter i stickprovet
  2) distance_form   – form betingad på distansklass (short/medium/long)
  3) market_*        – marknadspris (implicerad sannolikhet / poolandel)

Datakälla: data/atg_starters_sample.csv (ATG racinginfo, offentligt JSON-API).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "atg_starters_sample.csv"

FEATURES = [
    # 3 fokusfaktorer
    "trainer_change",
    "distance_form",
    "market_implied_prob",
    "market_share_pool",
    # övriga officiella/härledda
    "post_position",
    "field_size",
    "distance",
    "life_win_pct",
    "life_place_pct",
    "life_starts",
    "recent_form",
    "avg_odds_last5",
    "trainer_win_pct_ytd",
    "driver_win_pct_ytd",
    "horse_age",
    "shoe_changed",
]


def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = df.sort_values(["date", "race_id", "start_number"]).reset_index(drop=True)

    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Median-impute inom treningsfönster undviks enkelt: fyll med 0 / median global
    # (OK för MVP; produktion ska beräkna features utan leakage.)
    med = df[FEATURES].median(numeric_only=True)
    X = df[FEATURES].fillna(med).fillna(0.0)
    y = df["won"].astype(int)

    # Tidsbaserad split: sista ~25% av starterna = test (ingen framtidsleakage)
    cut = int(len(df) * 0.75)
    # flytta cut till race-gräns
    cut_date = df.iloc[cut]["date"]
    train_mask = df["date"] < cut_date
    test_mask = df["date"] >= cut_date
    if train_mask.sum() < 50 or test_mask.sum() < 20:
        train_mask = df.index < cut
        test_mask = df.index >= cut

    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    df_te = df[test_mask]

    clf = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=200,
        l2_regularization=0.1,
        random_state=42,
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("=== Trav MVP (HistGradientBoosting) ===")
    print(f"Data: {DATA}")
    print(f"Starter totalt: {len(df)} | train: {len(Xtr)} | test: {len(Xte)}")
    print(f"Train-datum < {cut_date.date()} | test >= {cut_date.date()}")
    print(f"Features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print()
    print(f"Log-loss:     {log_loss(yte, proba):.4f}")
    print(f"Brier:        {brier_score_loss(yte, proba):.4f}")
    try:
        print(f"ROC-AUC:      {roc_auc_score(yte, proba):.4f}")
    except ValueError:
        print("ROC-AUC:      n/a")
    print(f"Accuracy@0.5: {accuracy_score(yte, pred):.4f}  (svagt meningsfullt vid ~9% basrate)")

    # Per-lopp: topp-1 accuracy (högst modellprob vs faktisk vinnare)
    top1_ok = 0
    n_races = 0
    for rid, g in df_te.groupby("race_id"):
        idx = g.index
        p = pd.Series(proba, index=Xte.index).loc[idx]
        pick = p.idxmax()
        top1_ok += int(df.loc[pick, "won"] == 1)
        n_races += 1
    if n_races:
        print(f"Top-1 win acc (per lopp): {top1_ok}/{n_races} = {top1_ok/n_races:.3f}")

    # Marknadsbaseline: högst market_implied_prob
    m_ok = 0
    for rid, g in df_te.groupby("race_id"):
        pick = g["market_implied_prob"].astype(float).idxmax()
        m_ok += int(df.loc[pick, "won"] == 1)
    if n_races:
        print(f"Market favorite top-1:    {m_ok}/{n_races} = {m_ok/n_races:.3f}")

    # Visa toppickar för senaste testlopp
    last_race = df_te["race_id"].iloc[-1]
    g = df_te[df_te["race_id"] == last_race].copy()
    g["model_p"] = pd.Series(proba, index=Xte.index).loc[g.index].values
    g = g.sort_values("model_p", ascending=False)
    print()
    print(f"--- Exempel toppickar: {last_race} ({g['track_name'].iloc[0]}, {g['date'].iloc[0].date()}) ---")
    cols = [
        "start_number", "horse_name", "trainer_change", "distance_form",
        "market_implied_prob", "model_p", "place", "final_odds",
    ]
    print(g[cols].head(5).to_string(index=False))
    print()
    print("Tak för MVP: trav är brusigt; sikta på kalibrerade sannolikheter + value vs pool,")
    print("inte 'tipsa vinnaren varje gång'. Produktion: mer historik, Benter-blend, V-system-scoring.")


if __name__ == "__main__":
    main()
