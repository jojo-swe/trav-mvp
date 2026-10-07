#!/usr/bin/env python3
"""Grafiskt gränssnitt för Trav MVP (Streamlit).

Kör: streamlit run scripts/gui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from mvp_core import DATA, FEATURES, race_detail, run_mvp  # noqa: E402

st.set_page_config(
    page_title="Trav MVP",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Source+Sans+3:wght@400;500;600&display=swap');

:root {
  --ink: #122033;
  --steel: #1c4b6e;
  --fog: #d9e4ef;
  --paper: #eef3f8;
  --accent: #0e8f7a;
  --line: rgba(18, 32, 51, 0.12);
}

html, body, [class*="css"] {
  font-family: "Source Sans 3", sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(900px 380px at 0% 0%, rgba(14, 143, 122, 0.14), transparent 55%),
    radial-gradient(700px 320px at 100% 8%, rgba(28, 75, 110, 0.16), transparent 50%),
    linear-gradient(180deg, #eef3f8 0%, #e2ebf4 55%, #d7e2ee 100%);
}

.block-container {
  padding-top: 1.25rem;
  max-width: 1180px;
}

h1, h2, h3 {
  font-family: "Bricolage Grotesque", sans-serif !important;
  letter-spacing: -0.03em;
  color: var(--steel) !important;
}

.hero {
  display: grid;
  gap: 0.5rem;
  margin-bottom: 1.25rem;
  padding: 1.4rem 1.5rem 1.5rem;
  border-radius: 16px;
  background:
    linear-gradient(120deg, #12324a 0%, #1c4b6e 55%, #166a63 100%),
    repeating-linear-gradient(
      90deg,
      rgba(255,255,255,0.035) 0 1px,
      transparent 1px 18px
    );
  color: #eef6fb;
  box-shadow: 0 16px 36px rgba(18, 32, 51, 0.16);
  animation: rise 0.65s ease-out both;
}

.hero .brand {
  font-family: "Bricolage Grotesque", sans-serif;
  font-size: clamp(2.5rem, 5.2vw, 3.5rem);
  font-weight: 700;
  line-height: 1;
  color: #f4faff;
}

.hero .tag {
  max-width: 40rem;
  font-size: 1.05rem;
  line-height: 1.45;
  color: rgba(238, 246, 251, 0.9);
}

.hero .disclaimer {
  margin-top: 0.3rem;
  font-size: 0.82rem;
  color: rgba(238, 246, 251, 0.62);
}

div[data-testid="stMetric"] {
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.65rem 0.85rem;
  animation: fade-up 0.5s ease-out both;
}

div[data-testid="stMetric"]:nth-child(2) { animation-delay: 0.05s; }
div[data-testid="stMetric"]:nth-child(3) { animation-delay: 0.1s; }
div[data-testid="stMetric"]:nth-child(4) { animation-delay: 0.15s; }

div[data-testid="stMetric"] label {
  color: var(--steel) !important;
}

.stButton > button {
  background: var(--accent);
  color: #f4fffc;
  border: none;
  border-radius: 10px;
  font-weight: 600;
  padding: 0.55rem 1.15rem;
  transition: transform 0.15s ease, filter 0.15s ease;
}

.stButton > button:hover {
  filter: brightness(1.06);
  transform: translateY(-1px);
}

.section-note {
  color: rgba(18, 32, 51, 0.72);
  margin: -0.35rem 0 1rem;
}

@keyframes rise {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes fade-up {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
"""


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.1f}%"


@st.cache_resource(show_spinner="Tränar modellen…")
def cached_run(data_mtime: float, data_path: str):
    return run_mvp(Path(data_path))


def main():
    st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <div class="brand">Trav MVP</div>
          <div class="tag">
            Vinstsannolikheter för svensk trav — tränarbyte, distansform och marknadspris
            i samma vy som CLI-modellen.
          </div>
          <div class="disclaimer">
            Teknisk MVP, inte speltips. Respektera ATG:s villkor. Robotspel är förbjudet.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Körning")
        data_path = st.text_input("CSV", value=str(DATA))
        path = Path(data_path)
        if not path.exists():
            st.error(f"Hittar inte filen:\n{path}")
            st.stop()
        st.caption(f"{path.stat().st_size // 1024} KB · {path.name}")
        retrain = st.button("Träna om", use_container_width=True)
        if retrain:
            cached_run.clear()
        st.markdown("---")
        st.markdown("### Fokusfaktorer")
        for name in FEATURES[:4]:
            st.markdown(f"- `{name}`")
        st.caption(f"+ {len(FEATURES) - 4} övriga features")

    result = cached_run(path.stat().st_mtime, str(path))
    m = result.metrics

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "—")
    c2.metric("Log-loss", f"{m['log_loss']:.4f}")
    c3.metric("Top-1 modell", f"{m['top1_model_count']}/{m['n_races']} ({_fmt_pct(m['top1_model'])})")
    c4.metric("Top-1 marknad", f"{m['top1_market_count']}/{m['n_races']} ({_fmt_pct(m['top1_market'])})")

    st.markdown(
        f'<p class="section-note">Historik: <strong>{m["n_total"]}</strong> starter · '
        f'train &lt; {m["cut_date"]} ({m["n_train"]}) · '
        f'test ≥ {m["cut_date"]} ({m["n_test"]} starter, {m["n_races"]} lopp) · '
        f'Brier {m["brier"]:.4f}</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns((1.15, 1))
    with left:
        st.subheader("Testlopp")
        picks = result.race_picks.copy()
        if picks.empty:
            st.warning("Inga testlopp.")
        else:
            picks["datum"] = pd.to_datetime(picks["date"]).dt.strftime("%Y-%m-%d")
            picks["modell_rätt"] = picks["model_won"].map({1: "✓", 0: ""})
            picks["marknad_rätt"] = picks["market_won"].map({1: "✓", 0: ""})
            show = picks[
                [
                    "datum",
                    "track_name",
                    "game_type",
                    "model_start",
                    "model_pick",
                    "model_p",
                    "modell_rätt",
                    "market_start",
                    "market_pick",
                    "market_p",
                    "marknad_rätt",
                ]
            ].rename(
                columns={
                    "track_name": "bana",
                    "game_type": "spel",
                    "model_start": "nrₘ",
                    "model_pick": "modellpick",
                    "model_p": "p_modell",
                    "market_start": "nrₖ",
                    "market_pick": "marknadsfavorit",
                    "market_p": "p_marknad",
                }
            )
            st.dataframe(
                show.style.format({"p_modell": "{:.3f}", "p_marknad": "{:.3f}"}),
                use_container_width=True,
                height=360,
            )

    with right:
        st.subheader("Loppdetalj")
        race_ids = result.race_picks["race_id"].tolist()
        labels = {
            rid: (
                f"{pd.Timestamp(row.date).date()} · {row.track_name} · "
                f"{row.game_type} · {rid}"
            )
            for rid, row in result.race_picks.set_index("race_id").iterrows()
        }
        default_ix = len(race_ids) - 1 if race_ids else 0
        chosen = st.selectbox(
            "Välj lopp",
            options=race_ids,
            index=default_ix,
            format_func=lambda rid: labels.get(rid, rid),
        )
        if chosen:
            detail = race_detail(result, chosen)
            chart_df = detail.set_index("horse_name")[
                ["model_p", "market_implied_prob"]
            ].rename(
                columns={
                    "model_p": "Modell",
                    "market_implied_prob": "Marknad",
                }
            )
            st.bar_chart(chart_df, height=220)
            table = detail.copy()
            table["value"] = table["value"].round(2)
            st.dataframe(
                table.style.format(
                    {
                        "model_p": "{:.3f}",
                        "market_implied_prob": "{:.3f}",
                        "market_share_pool": "{:.3f}",
                        "distance_form": "{:.3f}",
                        "value": "{:.2f}",
                        "final_odds": "{:.2f}",
                    },
                    na_rep="—",
                ),
                use_container_width=True,
                height=320,
            )
            st.caption("value = model_p / market_implied_prob (>1 = modell ser mer value än poolen)")

    st.subheader("Om MVP:t")
    st.markdown(
        "Samma `HistGradientBoostingClassifier` och tidsbaserade split som "
        "`python scripts/train_mvp.py`. Marknaden är ofta stark baseline — "
        "nästa steg är value mot poolen, inte bara tipsa favoriten."
    )


if __name__ == "__main__":
    main()
