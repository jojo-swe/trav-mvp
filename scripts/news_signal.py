#!/usr/bin/env python3
"""Nyhetsinhämtning innan scoring — läs trav-nyheter om dagens startande.

Sök efter nyheter om hästar, kuskar, tränare och bana innan vi poängsätter
loppet, så att t.ex. en sen strykning, skada eller formrapport kan påverka
bedömningen — inte bara historiska features från ATG.

Ingen nyckel/API krävs i grundutförandet: vi försöker DuckDuckGos
HTML-sökning (samma enkla HTTP GET-mönster som fetch_atg_sample.py
använder mot ATG, ingen inloggning). I praktiken svarar DuckDuckGo här
med en anti-bot-utmaningssida (HTTP 202, ingen riktig resultat-HTML) för
oautentiserade skript-anrop — det är en medveten skyddsmekanism hos dem,
och vi kringgår den INTE (inga headless-browsers, inga captcha-lösare).
Resultatet är att nyhetssteget just nu nästan alltid faller tillbaka till
no-op i den här miljön, vilket loggas tydligt i NewsSignal.note. Byt ut
`_search()` mot ett riktigt sök-API (Bing Web Search, Brave Search,
SerpAPI, …) med en API-nyckel för att faktiskt få träffar — resten av
modulen (nyckelordsflaggning, justeringar, typsäkra dataclasses) fungerar
oförändrat med vilken sökkälla som helst. Oavsett sökkälla: om nätet inte
går att nå, eller formatet ändras, faller vi tillbaka på en tom, no-op-
signal så att hela pipen fortsätter fungera offline — nyhetssteget får
ALDRIG krascha körningen.

OBS: det här är en enkel nyckelordsheuristik, inte NLP/sentiment på
riktigt. Den flaggar troliga strykningar/skador/formrapporter för manuell
koll, den ersätter inte att faktiskt läsa ATG:s startlista.
"""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

UA = {"User-Agent": "trav-mvp/0.1 (personal research)"}
SEARCH_URL = "https://html.duckduckgo.com/html/"
TIMEOUT_S = 8
SLEEP_BETWEEN_QUERIES_S = 1.0

NEGATIVE_KEYWORDS = (
    "struken",
    "struket",
    "stryks",
    "skadad",
    "skada",
    "galopperade",
    "diskad",
    "sjuk",
    "avstår",
    "scratched",
    "injured",
    "withdrawn",
)
POSITIVE_KEYWORDS = (
    "fin form",
    "stark form",
    "vinnarform",
    "toppform",
    "het favorit",
    "storfavorit",
)

# Multiplikativ justering av model_p. Litet och konservativt med flit —
# det här är en grov nyhetsflagga, inte en ny modellterm, så den ska inte
# kunna dominera en annars välkalibrerad sannolikhet.
NEGATIVE_ADJUSTMENT = 0.35
POSITIVE_ADJUSTMENT = 1.10


@dataclass(frozen=True)
class NewsItem:
    query: str
    title: str
    snippet: str


@dataclass(frozen=True)
class NewsSignal:
    queried: bool
    items: tuple[NewsItem, ...]
    adjustments: dict[str, float] = field(default_factory=dict)
    flags: tuple[str, ...] = ()
    note: str = ""


def _search(query: str) -> list[NewsItem]:
    url = SEARCH_URL + "?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, TimeoutError, OSError):
        return []
    items: list[NewsItem] = []
    pattern = re.compile(
        r'class="result__a"[^>]*>(?P<title>.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        re.S,
    )
    for match in pattern.finditer(html):
        title = re.sub(r"<[^>]+>", "", match.group("title")).strip()
        snippet = re.sub(r"<[^>]+>", "", match.group("snippet")).strip()
        if title or snippet:
            items.append(NewsItem(query=query, title=title, snippet=snippet))
    return items[:6]


def _keyword_hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [kw for kw in keywords if kw in lowered]


def gather_news(
    *,
    track_name: str,
    horse_names: list[str],
    driver_names: list[str] | None = None,
    trainer_names: list[str] | None = None,
    max_entity_queries: int = 12,
    enabled: bool = True,
) -> NewsSignal:
    """Sök trav-nyheter för dagens startande innan scoring.

    Begränsar antal sökningar (max_entity_queries) så att ett stort V85
    inte gör 90+ nätanrop. Prioriterar hästnamn (mest relevant för
    skada/strykning), sedan tränare/kusk om budget finns kvar.
    """
    if not enabled:
        return NewsSignal(queried=False, items=(), note="nyhetssteg avstängt (--no-news)")

    driver_names = driver_names or []
    trainer_names = trainer_names or []
    queries: list[str] = []
    if track_name:
        queries.append(f"{track_name} trav nyheter strykning skada")
    for name in horse_names:
        if name:
            queries.append(f'"{name}" trav skadad struken')
    for name in trainer_names[:3]:
        if name:
            queries.append(f'"{name}" tränare trav nyheter')
    for name in driver_names[:3]:
        if name:
            queries.append(f'"{name}" kusk trav nyheter')
    queries = queries[:max_entity_queries]

    all_items: list[NewsItem] = []
    failures = 0
    for i, query in enumerate(queries):
        items = _search(query)
        if not items:
            failures += 1
        all_items.extend(items)
        if i + 1 < len(queries):
            time.sleep(SLEEP_BETWEEN_QUERIES_S)

    if failures == len(queries) and queries:
        return NewsSignal(
            queried=True,
            items=(),
            note=(
                "nätsökning misslyckades för alla frågor (ingen uppkoppling, "
                "DuckDuckGo blockerad, eller HTML-formatet ändrat) — "
                "fortsätter utan nyhetsjustering."
            ),
        )

    adjustments: dict[str, float] = {}
    flags: list[str] = []
    for name in horse_names:
        if not name:
            continue
        mentions = [
            item
            for item in all_items
            if name.lower() in item.title.lower() or name.lower() in item.snippet.lower()
        ]
        if not mentions:
            continue
        text_blob = " ".join(f"{m.title} {m.snippet}" for m in mentions)
        neg_hits = _keyword_hits(text_blob, NEGATIVE_KEYWORDS)
        pos_hits = _keyword_hits(text_blob, POSITIVE_KEYWORDS)
        if neg_hits:
            adjustments[name] = NEGATIVE_ADJUSTMENT
            flags.append(f"{name}: möjlig strykning/skada nämnd ({', '.join(neg_hits)}) — kontrollera ATG innan spel")
        elif pos_hits:
            adjustments[name] = POSITIVE_ADJUSTMENT
            flags.append(f"{name}: positiv formrapport nämnd ({', '.join(pos_hits)})")

    note = f"{len(queries)} sökningar, {len(all_items)} träffar, {len(adjustments)} hästjusteringar."
    return NewsSignal(queried=True, items=tuple(all_items), adjustments=adjustments, flags=tuple(flags), note=note)
