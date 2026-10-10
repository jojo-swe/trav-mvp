#!/usr/bin/env python3
"""Delad kärna för Trav MVP — samma modell som train_mvp.py."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

# Halveringstid för recency-vikt vid träning: en start för ~RECENCY_HALF_LIFE_DAYS
# dagar sedan referenspunkten väger hälften av en start på referenspunkten.
# Vald som en rimlig kompromiss för trav (säsonger, form, stall-ombyten) utan
# att vara ett eget forskningsprojekt — lätt att justera om kalibreringshistoriken
# (se calibration_history.jsonl) visar att modellen glömmer för snabbt/långsamt.
RECENCY_HALF_LIFE_DAYS = 180.0

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
    calibration_curve: list[dict[str, Any]] = field(default_factory=list)
    calibrator: IsotonicRegression | None = None


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


def recency_weights(
    dates: pd.Series, reference_date, half_life_days: float = RECENCY_HALF_LIFE_DAYS
) -> np.ndarray:
    """Exponentiellt avtagande träningsvikt: nyare starter väger mer.

    weight = 0.5 ** (days_before_reference / half_life_days), golvat vid en
    liten positiv konstant så ingen rad får vikt exakt 0 (HistGradientBoosting
    kräver vikt > 0 om sample_weight ges).
    """
    ref = pd.Timestamp(reference_date)
    days_before = (ref - pd.to_datetime(dates)).dt.days.clip(lower=0).astype(float)
    weights = 0.5 ** (days_before / max(half_life_days, 1e-6))
    return weights.clip(lower=1e-3).to_numpy()


def calibration_curve_ece(
    proba: np.ndarray, y: pd.Series, n_bins: int = 10
) -> tuple[list[dict[str, Any]], float]:
    """Binned reliability-kurva + Expected Calibration Error.

    Jämför förutsagd sannolikhet mot faktisk utfallsfrekvens inom bins av
    predikterad sannolikhet (kvantilbins, så varje bin har ungefär lika
    många observationer även med den kraftigt sneda vinst-basraten i trav).
    ECE = sum över bins av (andel av N i binnen) * |mean_pred - mean_actual|.
    """
    frame = pd.DataFrame({"p": np.asarray(proba, dtype=float), "y": pd.Series(y).astype(int).to_numpy()})
    if frame.empty:
        return [], 0.0
    try:
        frame["bin"] = pd.qcut(frame["p"], q=min(n_bins, frame["p"].nunique()), duplicates="drop")
    except ValueError:
        frame["bin"] = 0
    curve = []
    ece = 0.0
    n_total = len(frame)
    for _, group in frame.groupby("bin", observed=True):
        mean_pred = float(group["p"].mean())
        mean_actual = float(group["y"].mean())
        n = len(group)
        curve.append({"n": n, "mean_predicted": mean_pred, "mean_actual": mean_actual})
        ece += (n / n_total) * abs(mean_pred - mean_actual)
    curve.sort(key=lambda row: row["mean_predicted"])
    return curve, float(ece)


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

    # Recency-vikt relativt cut_date (INTE "idag") så att tidsplit-utvärderingen
    # är reproducerbar oavsett när den körs: träningsraderna strax före
    # cut_date väger mer än de äldsta, som om vi faktiskt stod vid cut_date.
    train_weights = recency_weights(df.loc[train_mask, "date"], reference_date=cut_date)

    clf = new_classifier()
    clf.fit(Xtr, ytr, sample_weight=train_weights)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Kalibrera på SAMMA tidsplit-holdout: det är de mest nyligen avslutade
    # loppen modellen ännu inte tränats på, dvs. precis vad en
    # prediktion-mot-utfall-feedbackloop ska jämföra mot. Isotonic är monoton
    # och parameterfri nog för de datamängder vi har (tiotusental rader).
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(proba, yte.to_numpy())
    calibration_curve, calibration_ece = calibration_curve_ece(proba, yte)
    calibrated_proba = calibrator.predict(proba)

    metrics: dict[str, Any] = {
        "log_loss": float(log_loss(yte, proba)),
        "brier": float(brier_score_loss(yte, proba)),
        "accuracy_0_5": float(accuracy_score(yte, pred)),
        "calibration_ece": float(calibration_ece),
        "calibrated_log_loss": float(log_loss(yte, calibrated_proba)),
        "calibrated_brier": float(brier_score_loss(yte, calibrated_proba)),
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
        calibration_curve=calibration_curve,
        calibrator=calibrator,
    )


@dataclass
class FittedMvp:
    """Modell tränad på hela historiken, för att poängsätta en kommande omgång."""

    clf: HistGradientBoostingClassifier
    medians: pd.Series
    n_rows: int
    calibrator: IsotonicRegression | None = None


def fit_full(df: pd.DataFrame, calibrator: IsotonicRegression | None = None) -> FittedMvp:
    """Träna på alla rader med känd vinst (live-scoring, inte tidsplit).

    sample_weight: recency relativt den senaste kända starten i datan (inte
    "idag" via systemklockan, så körningar är reproducerbara givet samma CSV).
    calibrator: isotonic-kalibreraren från run_mvp():s tidsplit-holdout — det
    är löpande "prediktion vs utfall"-återkopplingen. Den är fit på RAW
    model_p från tidsplit-modellen, men den fulla modellen (tränad på allt,
    inklusive holdout-perioden) ger nästan identisk sannolikhetsskala, så vi
    återanvänder samma kalibreringsfunktion istället för att kräva en egen
    holdout ur den fulla träningen (vilket skulle slösa bort färsk data).
    """
    trainable = df.dropna(subset=["won"]).copy()
    trainable["won"] = trainable["won"].astype(int)
    X, medians = feature_matrix(trainable)
    reference_date = trainable["date"].max()
    weights = recency_weights(trainable["date"], reference_date=reference_date)
    clf = new_classifier()
    clf.fit(X, trainable["won"].astype(int), sample_weight=weights)
    return FittedMvp(clf=clf, medians=medians, n_rows=int(len(trainable)), calibrator=calibrator)


def score_starters(fitted: FittedMvp, df: pd.DataFrame) -> pd.DataFrame:
    """Lägg på model_p (kalibrerad), model_p_raw, market_p och value.

    model_p_raw är modellens direkta predict_proba. Om fitted.calibrator
    finns (från run_mvp():s tidsplit-holdout, dvs. de senast avslutade
    loppen) mappas model_p_raw genom den till model_p — det är själva
    återkopplingslänken mellan "vad hände i de senaste loppen" och "hur
    tolkar vi dagens sannolikheter". All nedströms-kod (rankning, system,
    nyhetsjustering, beslutslager) läser model_p och får därmed kalibrerade
    värden automatiskt, utan att behöva ändras.

    market_p är normaliserad 1/vinnarodds. Saknas den används poolandelen.
    """
    out = df.copy()
    X, _medians = feature_matrix(out, fitted.medians)
    raw = fitted.clf.predict_proba(X)[:, 1]
    out["model_p_raw"] = raw
    out["model_p"] = fitted.calibrator.predict(raw) if fitted.calibrator is not None else raw
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
