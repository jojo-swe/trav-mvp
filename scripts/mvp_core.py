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
DATA = ROOT / "data" / "atg_starters.csv"

FEATURES = [
    "trainer_change",
    "distance_form",
    "market_implied_prob",
    "market_share_pool",
    "post_position",
    "post_position_share",
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
    "sulky_changed",
    "driver_change",
    "track_condition_code",
    "sex_mare",
    "sex_stallion",
    "sex_gelding",
]

# Fast = hårt underlag, Heavy/Soft = tungt. Siffran är en grov ordinal skala
# (lägre = snabbare bana) så trädmodellen får ett meningsfullt numeriskt
# samband snarare än en godtycklig kategorikod. Okänt värde -> median vid
# fit-time (samma fallback som övriga features), INTE en ny kategori, så
# kodningen är identisk mellan träning och live-scoring.
TRACK_CONDITION_CODE = {
    "fast": 0,
    "firm": 0,
    "good": 1,
    "normal": 1,
    "light": 1,
    "yielding": 2,
    "soft": 2,
    "heavy": 3,
    "muddy": 3,
    "slow": 3,
}

_BOOL_TRUE = {"true", "1", "1.0", "yes", "ja"}
_BOOL_FALSE = {"false", "0", "0.0", "no", "nej"}


def _coerce_bool(series: pd.Series) -> pd.Series:
    """CSV skriver Python-bools som strängarna "True"/"False".

    pd.to_numeric(..., errors="coerce") gör NaN av dem (bugg som gjorde att
    shoe_changed/sulky_changed i praktiken alltid blev 0 tidigare). Mappa
    explicit till 1/0 innan vi faller tillbaka på to_numeric för riktiga
    bool-/numeriska värden.
    """

    def _one(value):
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if pd.isna(value):
            return np.nan
        text = str(value).strip().lower()
        if text in _BOOL_TRUE:
            return 1.0
        if text in _BOOL_FALSE:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    return series.map(_one)


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


def _derive_engineered_columns(work: pd.DataFrame) -> pd.DataFrame:
    """Härled kolumner som inte finns rakt av i CSV:n, deterministiskt och
    identiskt vid träning och live-scoring (inga per-batch kategorikoder)."""
    if "post_position" in work.columns and "field_size" in work.columns:
        pos = pd.to_numeric(work["post_position"], errors="coerce")
        size = pd.to_numeric(work["field_size"], errors="coerce").replace(0, np.nan)
        work["post_position_share"] = pos / size
    else:
        work["post_position_share"] = np.nan

    for col in ("shoe_changed", "sulky_changed"):
        if col in work.columns:
            work[col] = _coerce_bool(work[col])
        else:
            work[col] = np.nan

    if "driver_change" not in work.columns:
        work["driver_change"] = np.nan

    if "track_condition" in work.columns:
        work["track_condition_code"] = (
            work["track_condition"].astype(str).str.strip().str.lower().map(TRACK_CONDITION_CODE)
        )
    else:
        work["track_condition_code"] = np.nan

    if "horse_sex" in work.columns:
        sex = work["horse_sex"].astype(str).str.strip().str.lower()
        work["sex_mare"] = (sex == "mare").astype(float)
        work["sex_stallion"] = (sex == "stallion").astype(float)
        work["sex_gelding"] = (sex == "gelding").astype(float)
        unknown = ~sex.isin({"mare", "stallion", "gelding"})
        for col in ("sex_mare", "sex_stallion", "sex_gelding"):
            work.loc[unknown, col] = np.nan
    else:
        work["sex_mare"] = np.nan
        work["sex_stallion"] = np.nan
        work["sex_gelding"] = np.nan
    return work


def feature_matrix(
    df: pd.DataFrame, medians: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    work = _derive_engineered_columns(work)
    for c in FEATURES:
        if c not in work.columns:
            work[c] = np.nan
        if c not in ("shoe_changed", "sulky_changed", "sex_mare", "sex_stallion", "sex_gelding"):
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

    if "prior_starts" in df.columns:
        prior = pd.to_numeric(df["prior_starts"], errors="coerce").fillna(0)
        metrics["prior_start_rate"] = float((prior > 0).mean())
        metrics["rows_with_prior"] = int((prior > 0).sum())
    if "distance_form_source" in df.columns:
        metrics["distance_form_starts_rate"] = float((df["distance_form_source"] == "starts").mean())
    if "trainer_change" in df.columns:
        changed = pd.to_numeric(df["trainer_change"], errors="coerce").fillna(0)
        metrics["trainer_change_rate"] = float((changed == 1).mean())

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
