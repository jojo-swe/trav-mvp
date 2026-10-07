# Trav MVP — V75/V85-prediktion (privat)

Personligt forskningsprojekt: historisk svensk travdata → vinstsannolikheter per häst, med fokus på **tränarbyte**, **distansbetingad form** och **marknadsprissättning**.

> **Ansvar — läs detta.** Detta är ett personligt analysverktyg, inte finansiell rådgivning och inte speltips. Travspel är slumpmässigt och riskfyllt. **Placera inga spel automatiskt.** ATG förbjuder **robotspel** (automatiserad kupongoptimering mot andra spelares kuponger). Skripten hämtar bara publik racinginfo-JSON och skriver ett textförslag.

## Snabbstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# utökad historik: alla avslutade svenska travlopp (återupptagbart, ≥0,4 s mellan anrop)
python scripts/fetch_atg_sample.py --expand --from-date 2025-10-10 --to-date 2026-10-09 --game-id V85_2026-10-10_5_6
python scripts/train_mvp.py
# nästa V75/V85: hämta live-data, träna, skriv vinnarrad
python scripts/predict_live.py
# grafiskt gränssnitt
streamlit run scripts/gui.py
# valfritt: litet V-omgångsstickprov (inte träningsfilen)
# python scripts/fetch_atg_sample.py
```

`predict_live.py` tittar upp till 14 dagar framåt från idag (Europe/Stockholm), tar dagens V75/V85 om den fortfarande är spelbar, annars nästa publicerade omgång. Den sparar startlista och rå-JSON under `data/live/`, tränar på `data/atg_starters.csv` och skriver enkelrad plus ett litet system till stdout och `data/live/suggested_rows.txt`. Minst 0,4 s mellan varje HTTP-anrop. Inga kuponger skickas till ATG.

Historiken byggs av `fetch_atg_sample.py --expand`: kalenderdag → avslutade travlopp på svenska banor → starter. Rå-JSON och checkpoint (`data/history/`) är gitignorerade. Tränings-CSV:n `data/atg_starters.csv` är den fil `train_mvp` / `predict_live` läser. `--game-id` hämtar dessutom `horses/{id}` för startande som saknar tidigare start eller distansform, och fyller på med de loppen. Avbruten hämtning fortsätter från `data/history/done_races.txt`.

## V75 vs V85 (struktur, okt 2026)

| Spel | Avdelningar | Utdelning | Status |
|------|-------------|-----------|--------|
| **V75** | 7 vinnare | 7 / 6 / 5 rätt | Tillbaka som lördagsspel från **28 nov 2026** (radpris 60 öre). Källa: [omatg.se](https://omatg.se/nyheter/v75-tillbaka-pa-lordagarna-premiar-28-november/) |
| **V85** | 8 vinnare | 8 / 7 / 6 / 5 rätt | Premiär 25 okt 2025; **pausas** när V75 återkommer. Källa: samma + [ATG kundservice](https://atg-extern.kb.kundo.se/guide/vilka-spelformer-finns-och-hur-hogt-ar-radpriset) |

MVP:t modellerar **per-lopp vinstsannolikhet**; V75/V85-system = produkt/kombination av avdelningsprobabilitet + value vs pool.

---

## 1) Datakällor — maximal officiell täckning

### A. ATG — publikt JSON (`racinginfo`) ★ primär för MVP

Odokumenterat men **live utan inloggning** (verifierat okt 2026):

| Endpoint | URL-mönster | Innehåll |
|----------|-------------|----------|
| Dagkalender | `https://www.atg.se/services/racinginfo/v1/api/calendar/day/YYYY-MM-DD` | Banor, lopp, `games` (V75/V85/V64…) |
| Spelomgång | `…/games/{GameType_YYYY-MM-DD_track_race}` | Alla avdelningar, starter, **pooler**, resultat, jackpot |
| Lopp | `…/races/{YYYY-MM-DD_track_n}` | Distans, startmetod, banförhållande, starter |
| Start | `…/races/{id}/start/{n}` | Djup häst/kusk |
| Häst | `…/horses/{id}` | Statistik, stamtavla, pengar |

**Fält som API:t faktiskt levererar (ur stickprov):**

- Identitet: häst, kusk, tränare (id + namn), ägare, uppfödare, hemmahana  
- Lopp: distans, autostart/volt, bana, `track.condition` (t.ex. `light`)  
- Resultat: placering, km-tid, prispengar, `finalOdds`  
- **Marknad:** `pools.vinnare.odds` / `finalOdds` (×100), `pools.plats`, `pools.V85.betDistribution` (poolandel)  
- Form/utrustning: skor (`shoes`), vagn/sulky (`sulky`), rekord per distansklass, `lastFiveStarts.averageOdds`, års- och livsstatistik (seger%/plats%)  
- Media-id för video  

**Djup:** historiska dagar svarar (stickprov 2024–2025). **Machine-readable:** ja (JSON). **Gratis:** ja för läsning i webbläsare/API-stil. **ToS:** ej explicit “öppen data”; ATG förbjuder **robotspel** ([guide](https://atg-extern.kb.kundo.se/guide/vad-ar-robotspel)). Bulk-scraping kan bryta villkor — håll dig till rimlig rate, gärna partneravtal för produktion.

### B. ATG — officiella partnerflöden (AIS m.m.)

- [swedishhorseracing.com — Offer and solution](https://www.swedishhorseracing.com/for-partners/offer-and-solution): **AIS** (SOAP/XML), XML race cards, PDF-program, tips/handicapping/ranking från tredjepart, **bulk historik som textfiler** för certifierade partners.  
- Inte publikt för hobby-API-nyckel.

### C. ATG startlista / “robot-” & automatiserad info (UI + underliggande data)

Ingen separat publik produkt som heter exakt **“robotinformation”** hittades. Närmaste **officiella** automatiserade / rika startlistefält:

| Källa | URL | Vad |
|-------|-----|-----|
| Tolka startlista | https://atg-extern.kb.kundo.se/guide/hur-tolkar-jag-informationen-i-startlistan | Spel%, formrader, tipskommentar (TR Media), Travfakta-loppkommentar, vagn, skor, trend |
| Vagninformation | https://atg-extern.kb.kundo.se/guide/vad-ar-vagninfo-och-hur-fungerar-det | Sulkytyp från **Svensk Travsport** via ATG |
| Förkortningar | https://atg-extern.kb.kundo.se/guide/forkortningar-i-startlistan → travsport.se PDF | Officiella koder |
| Harry Boy | https://atg-extern.kb.kundo.se/guide/harry-boy-vad-ar-det-och-hur-fungerar-det | ATG:s **algoritmiska** systembyggare (Original/Favorit/Chans) — inte öppet API |
| Elli (AI) | https://atg-extern.kb.kundo.se/guide/spelassistenten-elli-hast-fragor-kuponger-och-statistik | Inloggad AI över ATG-data — inte API |
| Robotspel (förbud) | https://atg-extern.kb.kundo.se/guide/vad-ar-robotspel | Viktigt: inte samma sak som datamodell |

I vårt CSV mappas “robot-/startliste-automation” till fält som redan finns i racinginfo: **sko, sulky, odds, betDistribution, statistik**.

### D. Svensk Travsport / travsport.se / Sportinfo

| Resurs | URL | Machine-readable | Kommentar |
|--------|-----|------------------|-----------|
| Sportinfo hästsök | https://sportapp.travsport.se/sportinfo/horse | Web UI (+ ev. underliggande XHR) | Karriärresultat, stamtavla, ägare — **gratis att titta** |
| Startlista tävlingsdag | https://sportapp.travsport.se/race/raceday/…/startlist/all | HTML (+ “Exportera”) | Propositioner, priser, spår |
| Resultat | https://sportapp.travsport.se/race/raceday/…/results/… | HTML | Placering, tid, **banförhållande**, bestraffningar |
| travsport.se | https://www.travsport.se/ | Portal | Kalender, nyheter, Pegasus-rapporter |
| Sportdata-licens | t.ex. Unibet–ST-avtal (press) | — | Officiell sportdata är **licensierad**, inte öppet API |

### E. Tredjepart (inte officiellt, men transparanta)

| Källa | URL | Pris | Kommentar |
|-------|-----|------|-----------|
| Travmaskinen API | https://travmaskinen.se/api-access | Prognos ~1499 kr/mån, databas ~2499 kr/mån | JSON sedan 2012; XGBoost+Benter, AUC ~0.83 enligt deras [metodologi](https://travmaskinen.se/metodologi) — de anger att rådata kommer från ATG:s publika API |
| Parse “ATG Horse Racing API” | https://parse.bot/… | Betald wrapper | Wrapper runt atg.se |

### F. Forskning

- DiVA: *Prediction of Swedish Harness Racing* — historiskt scraped travsport.se (ej rekommenderat som legal väg idag).

---

## 2) Modellrekommendation (MVP → nästa steg)

**MVP:** `HistGradientBoostingClassifier` (sklearn) — binär **vinst** per start.

**Varför:** kategoriska/glesa travfeatures + oddsmarknad fungerar bra med gradient boosting; snabb att träna; kalibrerbara probabiliteter (log-loss/Brier). Alternativ senare: XGBoost/LightGBM + **Benter log-odds blend** med pool (som Travmaskinen beskriver), eller ranking-loss (LambdaMART) per lopp.

### De tre obligatoriska faktorerna

1. **`trainer_change`** — 1 om `trainer_id` skiljer sig från hästens senaste tidigare start i historiken.  
2. **`distance_form`** — medel av `1/placering` på samma distansklass (short &lt;1800, medium &lt;2300, long) i upp till 5 tidigare starter; fallback: rekord i klassen.  
3. **`market_implied_prob` / `market_share_pool`** — från ATG `vinnare`-odds (normaliserad 1/odds) respektive V-spel `betDistribution`.

Övriga features: spår, fältstorlek, livs-/årsstatistik, kusk/tränar-seger%, skor/sulky-byte, ålder.

**Split:** strikt **tidsbaserad** (träna på äldre datum, testa på nyare) — ingen läckage från framtida lopp.

**V75/V85-scoring:** per avdelning modell-p → systemet = kombinationer; jämför med poolfavoriter för **value** (`model_p / market_p`). Favorit-tunga vs value-system är produktval (jämför Harry Favorit/Chans).

**Tak:** brusigt; realistiskt tak liknande publicerade ~AUC 0.80–0.85 / top-1 ~25–30% — inte “alla rätt varje lördag”.

---

## 3) Vad som finns i repot

```
trav-mvp/
  README.md
  requirements.txt
  .gitignore
  data/atg_starters.csv          # primär träning: svenska travstarter (fetch --expand)
  data/atg_starters_sample.csv   # litet V75/V85-stickprov, används av parsertestet
  data/history/                  # jsonl-checkpoint från --expand (gitignorerad)
  data/*.json                    # exempel-payloads (gitignorerade)
  data/live/                     # skapas av predict_live.py (gitignorerad)
  scripts/fetch_atg_sample.py    # racinginfo-hämtning, delas med live
  scripts/mvp_core.py            # delad träningskärna
  scripts/train_mvp.py           # CLI: träna + metriker + toppickar
  scripts/predict_live.py        # nästa V75/V85 → föreslagen rad
  scripts/gui.py                 # Streamlit-gränssnitt
```

### Körresultat (körning 2026-10-07, Europe/Stockholm)

Historik: **68 156 starter**, 7 079 svenska travlopp, 10 335 hästar, 2025-10-10 … 2026-10-07. Tidsplit train &lt; 2026-07-20 / test ≥ 2026-07-20 (50 932 / 17 224 starter).

| Metrik | Värde |
|--------|-------|
| Log-loss | **0.2479** |
| Brier | **0.0734** |
| ROC-AUC | **0.8513** |
| Top-1 (modell) | 768/1779 = **43.2%** |
| Top-1 (marknadsfavorit) | 765/1779 = **43.0%** |

Det tidigare V-omgångsstickprovet (1 574 starter) gav AUC 0.8247 men täckte bara ~33/99 hästar på live-V85. På den utökade historiken ligger modellens top-1 i linje med marknaden.

Live **V85 Solvalla 2026-10-10** (`V85_2026-10-10_5_6`, spelbar):

| | |
|--|--|
| Historik | **99/99 (100%)** har minst en tidigare start |
| Distansform från starter | **98/99 (99%)** |
| Enkelrad (model_p) | **1-9-11-1-2-6-4-2** |
| Value (topp-3) | 1-8-7-14-2-6-5-2 |
| Marknad | 1-9-11-1-2-6-4-2 |

Tränarbyte är beräknat mot senaste tidigare start (ett byte på kortet: avd 2 #1 Charron). Ace Wibb (avd 4 #14) har 14 tidigare starter men ingen på lång distans, så distansform lämnas tom i stället för ett påhittat värde. `market_share_pool` i historikfilen faller tillbaka till implicerad vinnarsannolikhet när lopp-JSON saknar V-poolens `betDistribution`; live-raden använder publicerad poolandel. Historiska `finalOdds` är stängningsodds. Inga spel läggs.

### Nästa steg mot produktion

1. Ännu längre fönster (flera år) om en live-omgång fortfarande har hästar utan svensk start.
2. Snapshotta **pre-race** odds (inte bara finalOdds) för äkta marknadsfeatures.
3. Benter-blend + V-system simulator (kostnad, 5–8 rätt-fördelning).
4. Undvik allt som liknar förbjudet robotspel.

## Licens / privat

Privat repo. Data från ATG/ST tillhör respektive rättighetsinnehavare — använd för personlig analys enligt deras villkor.
