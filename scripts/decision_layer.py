#!/usr/bin/env python3
"""Beslutslager — typsäker sammanfattning av de rankade raderna.

Terminologitolkning (research gjord 2026-10-10, se commit-meddelande för
detaljer): användaren skrev "jev (typesafe)" och "descisions api
(openai)". Det visade sig INTE vara rena stavfel/transkriptionsfel:

  * "Jev" är ett riktigt, nylanserat (okt 2026) beslutsmotor-API från ett
    bolag som kallar sig "TypeSafe (AI)" — en konkurrent till OpenAIs nya
    Decisions-endpoint. Det kräver egen kontoregistrering och SDK
    (`@typesafe-ai/sdk` / `typesafe-sdk`, miljövariabel TYPESAFE_API_KEY)
    som inte finns konfigurerad i den här miljön.
  * OpenAI lanserade samtidigt en riktig `POST /v1/decisions`-endpoint
    (modell gpt-6-luna, publik beta) för typade choice/predicate/score-
    frågor — snabbare än vanliga Responses-anrop.

Ingen av dessa två betaprodukter är åtkomliga här (ingen separat
TypeSafe-nyckel, och den dedikerade /v1/decisions-endpointen kräver
beta-access utöver en vanlig OPENAI_API_KEY). Vi implementerar därför den
närmaste praktiska motsvarigheten med det som faktiskt är tillgängligt:

  1. "Typesafe/Jev"  -> frozen dataclasses med fullständiga typannoteringar
     (RowCandidate, DecisionVerdict) istället för lösa dict:ar — Pythons
     närmaste motsvarighet till Zod/TypeSafe-stilens körtidstypade data.
  2. "Decisions API"  -> samma MÖNSTER (en smal, typad choice/score-fråga
     om "hur säker är topp-raden") byggt på OpenAIs vanliga Chat
     Completions med structured outputs (response_format=json_schema),
     eftersom den riktiga /v1/decisions-endpointen inte är nåbar här.
     Om OPENAI_API_KEY saknas (vilket den gör i den här miljön) faller vi
     tillbaka på en deterministisk regelbaserad bedömning, så att
     pipelinen fortsätter fungera helt offline/utan nyckel.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

Confidence = Literal["low", "medium", "high"]
Source = Literal["openai_decisions_pattern", "heuristic_fallback"]


@dataclass(frozen=True)
class RowCandidate:
    rank: int
    coupon: str
    combined_probability: float


@dataclass(frozen=True)
class DecisionVerdict:
    confidence: Confidence
    headline: str
    rationale: str
    risk_flags: tuple[str, ...]
    source: Source


def _heuristic_verdict(
    rows: list[RowCandidate], legs_open: int, news_flags: tuple[str, ...]
) -> DecisionVerdict:
    if not rows:
        return DecisionVerdict(
            confidence="low",
            headline="Inga rader",
            rationale="Ingen giltig rankad rad kunde byggas.",
            risk_flags=news_flags,
            source="heuristic_fallback",
        )
    best = rows[0]
    gap = (
        best.combined_probability / rows[1].combined_probability
        if len(rows) > 1 and rows[1].combined_probability > 0
        else None
    )
    if gap is not None and gap > 1.5 and legs_open <= 2:
        confidence: Confidence = "high"
    elif gap is not None and gap > 1.15:
        confidence = "medium"
    else:
        confidence = "low"
    if news_flags:
        # Nyhetsflaggor sänker aldrig under "low" men gör oss inte mer
        # säkra än modellen redan var — ren försiktighetsregel.
        confidence = "low" if confidence != "high" else "medium"
    rationale = (
        f"Bästa raden ({best.coupon}) har {gap:.2f}x högre kombinerad sannolikhet än rad 2."
        if gap is not None
        else f"Bästa raden ({best.coupon}) saknar en nära rad 2 att jämföra mot."
    )
    if news_flags:
        rationale += " Nyhetsflaggor sänker säkerheten — se risk_flags."
    return DecisionVerdict(
        confidence=confidence,
        headline=f"Topprad: {best.coupon}",
        rationale=rationale,
        risk_flags=news_flags,
        source="heuristic_fallback",
    )


_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "headline": {"type": "string"},
        "rationale": {"type": "string"},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["confidence", "headline", "rationale", "risk_flags"],
    "additionalProperties": False,
}


def _openai_verdict(
    rows: list[RowCandidate], legs_open: int, news_flags: tuple[str, ...]
) -> DecisionVerdict | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not rows:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None

    model = os.environ.get("OPENAI_DECISION_MODEL", "gpt-4o-mini")
    top = rows[:5]
    prompt = (
        "Du bedömer en rankad lista av V75/V85-kupongrader för svenskt trav. "
        "Ge ett strukturerat beslut (på svenska) om hur stark/osäker topp-raden är "
        "jämfört med övriga. Var konservativ — det här är personlig analys, inte "
        "speltips, och hög säkerhet ska bara ges vid tydligt tydligt glapp till rad 2.\n\n"
        f"Öppna avdelningar (bank=False): {legs_open}\n"
        f"Rader: {json.dumps([r.__dict__ for r in top], ensure_ascii=False)}\n"
        f"Nyhetsflaggor: {json.dumps(news_flags, ensure_ascii=False)}"
    )
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "decision_verdict",
                    "schema": _DECISION_SCHEMA,
                    "strict": True,
                },
            },
        )
        data = json.loads(resp.choices[0].message.content)
        return DecisionVerdict(
            confidence=data["confidence"],
            headline=data["headline"],
            rationale=data["rationale"],
            risk_flags=tuple(data.get("risk_flags") or news_flags),
            source="openai_decisions_pattern",
        )
    except Exception:
        # Decisions-API-mönstret är ett tillägg, inte en kritisk väg — om
        # anropet fallerar (nätverk, kvot, schema-fel) ska pipen aldrig
        # krascha, bara falla tillbaka till den deterministiska regeln.
        return None


def decide(
    rows: list[RowCandidate], legs_open: int, news_flags: tuple[str, ...] = ()
) -> DecisionVerdict:
    verdict = _openai_verdict(rows, legs_open, news_flags)
    if verdict is not None:
        return verdict
    return _heuristic_verdict(rows, legs_open, news_flags)
