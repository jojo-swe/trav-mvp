#!/usr/bin/env python3
"""Delad kärna för Trav MVP — samma modell som train_mvp.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    "trainer_change",
    "distance_form",
    "market_implied_prob",
    "market_share_pool",
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


@dataclass
class MvpResult:
    df: pd.DataFrame
    df_te: pd.DataFrame
    proba: np.ndarray
    yte: pd.Series
    cut_date: Any
    n_train: int
    n_test: int
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    race_picks: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_starters(path: Path | None = None) -> pd.DataFrame:
    data_path = Path(path) if path else DATA
    df = pd.read_csv(data_path, parse_dates=["date"])
    return df.sort_values(["date", "race_id", "start_number"]).reset_index(drop=True)


def new_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=200,
        l2_regularization=0.1,
        random_state=42,
    )


def feature_matrix(
    df: pd.DataFrame, medians: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    for c in FEATURES:
        if c not in work.columns:
            work[c] = np.nan
        work[c] = pd.to_numeric(work[c], errors="coerce")
    if medians is None:
        medians = work[FEATURES].median(numeric_only=True)
    X = work[FEATURES].fillna(medians).fillna(0.0)
    return X, medians


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X, _medians = feature_matrix(df)
    y = df["won"].astype(int)
    return X, y


def time_split(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, Any]:
    cut = int(len(df) * 0.75)
    cut_date = df.iloc[cut]["date"]
    train_mask = df["date"] < cut_date
    test_mask = df["date"] >= cut_date
    if train_mask.sum() < 50 or test_mask.sum() < 20:
        train_mask = df.index < cut
        test_mask = df.index >= cut
    return train_mask, test_mask, cut_date


def run_mvp(path: Path | None = None) -> MvpResult:
    """Träna HistGradientBoosting med samma hyperparametrar som CLI-MVP:t."""
    data_path = Path(path) if path else DATA
    df = load_starters(data_path)
    X, y = prepare_xy(df)
    train_mask, test_mask, cut_date = time_split(df)

    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    df_te = df[test_mask].copy()

    clf = new_classifier()
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics: dict[str, Any] = {
        "log_loss": float(log_loss(yte, proba)),
        "brier": float(brier_score_loss(yte, proba)),
        "accuracy_0_5": float(accuracy_score(yte, pred)),
        "n_total": int(len(df)),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "cut_date": str(pd.Timestamp(cut_date).date()),
        "data_path": str(data_path),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(yte, proba))
    except ValueError:
        metrics["roc_auc"] = None

    proba_s = pd.Series(proba, index=Xte.index)
    top1_ok = 0
    m_ok = 0
    n_races = 0
    pick_rows = []
    for rid, g in df_te.groupby("race_id", sort=False):
        idx = g.index
        p = proba_s.loc[idx]
        model_pick = p.idxmax()
        market_pick = g["market_implied_prob"].astype(float).idxmax()
        top1_ok += int(df.loc[model_pick, "won"] == 1)
        m_ok += int(df.loc[market_pick, "won"] == 1)
        n_races += 1
        pick_rows.append(
            {
                "race_id": rid,
                "date": g["date"].iloc[0],
                "track_name": g["track_name"].iloc[0],
                "game_type": g["game_type"].iloc[0] if "game_type" in g else "",
                "distance": g["distance"].iloc[0],
                "field_size": int(g["field_size"].iloc[0]),
                "model_pick": df.loc[model_pick, "horse_name"],
                "model_start": int(df.loc[model_pick, "start_number"]),
                "model_p": float(p.loc[model_pick]),
                "model_won": int(df.loc[model_pick, "won"] == 1),
                "market_pick": df.loc[market_pick, "horse_name"],
                "market_start": int(df.loc[market_pick, "start_number"]),
                "market_p": float(g.loc[market_pick, "market_implied_prob"]),
                "market_won": int(df.loc[market_pick, "won"] == 1),
            }
        )

    metrics["n_races"] = n_races
    metrics["top1_model"] = top1_ok / n_races if n_races else None
    metrics["top1_market"] = m_ok / n_races if n_races else None
    metrics["top1_model_count"] = top1_ok
    metrics["top1_market_count"] = m_ok

    df_te = df_te.copy()
    df_te["model_p"] = proba_s.loc[df_te.index].values
    market = df_te["market_implied_prob"].astype(float).replace(0, np.nan)
    df_te["value"] = df_te["model_p"] / market

    fi = pd.DataFrame({"feature": FEATURES, "importance": np.nan})

    race_picks = pd.DataFrame(pick_rows)
    if not race_picks.empty:
        race_picks = race_picks.sort_values("date").reset_index(drop=True)

    return MvpResult(
        df=df,
        df_te=df_te,
        proba=proba,
        yte=yte,
        cut_date=cut_date,
        n_train=int(len(Xtr)),
        n_test=int(len(Xte)),
        metrics=metrics,
        feature_importance=fi,
        race_picks=race_picks,
    )


@dataclass
class FittedMvp:
    """Modell tränad på hela historiken, för att poängsätta en kommande omgång."""

    clf: HistGradientBoostingClassifier
    medians: pd.Series
    n_rows: int


def fit_full(df: pd.DataFrame) -> FittedMvp:
    """Träna på alla rader med känd vinst (live-scoring, inte tidsplit)."""
    trainable = df.dropna(subset=["won"]).copy()
    trainable["won"] = trainable["won"].astype(int)
    X, medians = feature_matrix(trainable)
    clf = new_classifier()
    clf.fit(X, trainable["won"].astype(int))
    return FittedMvp(clf=clf, medians=medians, n_rows=int(len(trainable)))


def score_starters(fitted: FittedMvp, df: pd.DataFrame) -> pd.DataFrame:
    """Lägg på model_p, market_p och value = model_p / market_p.

    market_p är normaliserad 1/vinnarodds. Saknas den används poolandelen.
    """
    out = df.copy()
    X, _medians = feature_matrix(out, fitted.medians)
    out["model_p"] = fitted.clf.predict_proba(X)[:, 1]
    if "market_implied_prob" in out.columns:
        market = pd.to_numeric(out["market_implied_prob"], errors="coerce")
    else:
        market = pd.Series(np.nan, index=out.index)
    if "market_share_pool" in out.columns:
        pool = pd.to_numeric(out["market_share_pool"], errors="coerce")
        market = market.where(market.notna() & (market > 0), pool)
    else:
        market = market.where(market > 0)
    market = market.replace(0, np.nan)
    out["market_p"] = market
    out["value"] = out["model_p"] / market
    return out


def race_detail(result: MvpResult, race_id: str) -> pd.DataFrame:
    g = result.df_te[result.df_te["race_id"] == race_id].copy()
    cols = [
        c
        for c in [
            "start_number",
            "horse_name",
            "driver_name",
            "trainer_name",
            "trainer_change",
            "distance_form",
            "market_implied_prob",
            "market_share_pool",
            "model_p",
            "value",
            "place",
            "final_odds",
            "won",
        ]
        if c in g.columns
    ]
    return g[cols].sort_values("model_p", ascending=False).reset_index(drop=True)
