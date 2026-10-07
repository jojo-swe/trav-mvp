"""Parsern ska ge samma marknadsfält som det sparade stickprovet."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_atg_sample import apply_history, rows_from_game  # noqa: E402
from predict_live import build_suggestion  # noqa: E402


class RowsFromGameTest(unittest.TestCase):
    def test_matches_sample_csv_for_saved_v85(self):
        game = json.loads((ROOT / "data" / "game_V85_2025-10-25.json").read_text())
        rows = rows_from_game(game, "V85", require_results=True)
        self.assertGreaterEqual(len(rows), 50)
        parsed = pd.DataFrame(rows)
        sample = pd.read_csv(ROOT / "data" / "atg_starters_sample.csv")
        sample = sample[sample["game_id"] == "V85_2025-10-25_7_3"]
        self.assertFalse(sample.empty)
        merged = sample.merge(
            parsed,
            on=["race_id", "start_number"],
            suffixes=("_csv", "_api"),
        )
        self.assertEqual(len(merged), len(sample))
        for column in (
            "horse_name",
            "final_odds",
            "market_implied_prob",
            "market_share_pool",
            "won",
            "place",
            "field_size",
            "life_win_pct",
            "trainer_win_pct_ytd",
            "driver_win_pct_ytd",
        ):
            left = merged[f"{column}_csv"]
            right = pd.to_numeric(merged[f"{column}_api"], errors="coerce")
            if column == "horse_name":
                self.assertTrue((merged["horse_name_csv"] == merged["horse_name_api"]).all())
                continue
            if column in {"won", "place", "field_size"}:
                self.assertTrue((left.astype(int) == right.astype(int)).all(), column)
                continue
            self.assertTrue(
                (left - right).abs().max() < 1e-5,
                f"{column} max diff {(left - right).abs().max()}",
            )


class HistoryTest(unittest.TestCase):
    def test_trainer_change_and_distance_form_from_prior_starts(self):
        rows = [
            {
                "date": "2025-10-25",
                "horse_id": 10,
                "trainer_id": 1,
                "distance_class": "medium",
                "place": 2,
                "distance": 2140,
                "final_odds": 4.0,
                "distance_form": 1.0,
            },
            {
                "date": "2025-11-01",
                "horse_id": 10,
                "trainer_id": 2,
                "distance_class": "medium",
                "place": 1,
                "distance": 2140,
                "final_odds": 3.0,
                "distance_form": None,
            },
        ]
        apply_history(rows)
        self.assertEqual(rows[0]["trainer_change"], 0)
        self.assertEqual(rows[1]["trainer_change"], 1)
        self.assertAlmostEqual(rows[1]["distance_form"], 0.5)
        self.assertEqual(rows[1]["distance_starts_prior"], 1)


class SuggestionTest(unittest.TestCase):
    def test_system_garderar_bara_tydlig_value(self):
        scored = pd.DataFrame(
            [
                _starter(1, 1, "A", 0.40, 0.50),
                _starter(1, 2, "B", 0.20, 0.10),
                _starter(1, 3, "C", 0.05, 0.20),
                _starter(2, 4, "D", 0.30, 0.30),
                _starter(2, 5, "E", 0.15, 0.20),
            ]
        )
        suggestion = build_suggestion(scored)
        self.assertEqual(suggestion["enkelrad"], "1-4")
        # B has value 2.0 vs A's 0.8, so leg 1 is garderad. Leg 2 is a single.
        self.assertEqual(suggestion["system_notation"], "1/2-4")
        self.assertEqual(suggestion["n_system_rows"], 2)
        self.assertEqual(suggestion["legs"][0]["reserves"][0]["horse_name"], "B")


def _starter(leg, number, name, model_p, market_p):
    return {
        "leg": leg,
        "race_id": f"r{leg}",
        "race_name": "",
        "track_name": "Solvalla",
        "distance": 2140,
        "start_method": "auto",
        "field_size": 3,
        "start_number": number,
        "horse_name": name,
        "trainer_name": "",
        "trainer_change": 0,
        "distance_form": 0.5,
        "model_p": model_p,
        "market_p": market_p,
        "value": model_p / market_p,
    }


if __name__ == "__main__":
    unittest.main()
