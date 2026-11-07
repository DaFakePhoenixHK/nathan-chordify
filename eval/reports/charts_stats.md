# Chord-chart harvest: statistics

Generated 2026-11-01 by `train/harvest_charts.py index` from `data/charts/*.chart.txt` (primary chart per song; the second Simply Worship charts in `data/charts/alt/` are not counted below).
Chords only: no lyrics were stored. Test songs excluded (立志擺上, 活出愛, Here I Bow, 10,000 Reasons, 回家 Eternity Girls).
Bar lengths are **inferred** from the syllables under each chord (charts have no bar lines), so durations and
"changes per bar" are approximate; chord order, chord types and progressions are as the charts give them.
Numbers (1, 4, 5, 6m, 5/7 ...) are Nashville numbers / 級數 relative to each chart's key, so songs in different keys add up.

## Coverage

- Songs with a chart: **123** of 123 candidates (test songs excluded); parse OK: **123/123** (100%). Extra Simply Worship second charts: 41 (parse OK 41).
- Per language: english 62, cantonese 40, mandarin 21
- Per source: CantonHymn 58, WorshipTogether 48, UltimateGuitar 14, SimplyWorship 3
- Keys: G 37, D 23, A 19, C 15, E 8, F 8, Ab 4, B 3, Eb 2, Dm 1, F# 1, Am 1, Bb 1
- Meters: 4/4 121, 3/4 2 (4/4 is assumed when a page gives no time signature)
- Total: 5693 bars, 5897 chord changes, 176 distinct chord symbols; 1.04 changes per bar (inferred).
- Fetch log (`data/charts/harvest_log.csv`): 186 page requests; all answered HTTP 200, no source blocked us.
- Pages with no chords on them (another source was used): WorshipTogether 5, CantonHymn 3

## How often each kind of chord appears (share of chord changes / songs that use it)

| Kind | Share of chord changes | Songs using it |
|---|---:|---:|
| sus (sus2, sus4, 7sus4, 9sus4, 13sus4) | 5.8% | 46/123 |
| add9 / madd9 / 6/9 | 0.5% | 7/123 |
| maj7 / maj9 | 2.0% | 20/123 |
| minor 7th family (m7, m9, m11) | 5.8% | 59/123 |
| dominant 7th family (7, 9, 13, 7b9 ...) | 3.5% | 51/123 |
| slash chords (any bass other than the root) | 12.1% | 91/123 |
| plain major / minor triads | 82.2% | 123/123 |

## How long chords last (inferred, so approximate)

| Length (beats; 4/4 bar = 4) | Share of chord changes |
|---|---:|
| half a bar or less | 35.0% |
| up to one bar | 53.6% |
| up to two bars | 9.6% |
| longer | 1.9% |

## Chord qualities (Harte names)

| Quality | Changes | Share | Share of time |
|---|---:|---:|---:|
| maj | 3772 | 64.0% | 65.0% |
| min | 1073 | 18.2% | 17.5% |
| m7 | 336 | 5.7% | 5.7% |
| 7 | 203 | 3.4% | 2.6% |
| sus2 | 198 | 3.4% | 3.8% |
| sus4 | 144 | 2.4% | 2.2% |
| maj7 | 102 | 1.7% | 2.0% |
| add9 | 23 | 0.4% | 0.5% |
| maj9 | 16 | 0.3% | 0.2% |
| m7b5 | 6 | 0.1% | 0.1% |
| 6 | 6 | 0.1% | 0.1% |
| madd9 | 4 | 0.1% | 0.1% |
| m9 | 4 | 0.1% | 0.1% |
| dim | 3 | 0.1% | 0.0% |
| aug | 3 | 0.1% | 0.0% |
| 7b9 | 2 | 0.0% | 0.0% |
| 7sus4 | 2 | 0.0% | 0.0% |

## Most common chords as numbers (full detail)

`1` 1099, `4` 970, `5` 965, `6m` 605, `1/3` 241, `2m` 241, `5/7` 186, `6m7` 183, `3m` 178, `4sus2` 167, `5(7)` 128, `2m7` 116, `5sus4` 116, `4maj7` 70, `4/5` 48, `1/5` 45, `b7` 41, `3m7` 31, `2` 26, `1(7)` 22, `3(7)` 21, `5/4` 21, `1sus4` 19, `3` 18, `4/6` 17, `1maj7` 17, `4/1` 16, `5/1` 16, `4m` 15, `4maj9` 14

Simplified to triads (simple level): `1` 1510, `5` 1456, `4` 1320, `6m` 798, `2m` 371, `3m` 216, `3` 53, `b7` 52, `2` 50, `4m` 20, `6` 19, `5m` 9, `b3` 8, `#4°` 7, `7` 2

## Most common slash chords (as numbers)

`1/3` 241, `5/7` 186, `4/5` 48, `1/5` 45, `5/4` 21, `4/6` 17, `4/1` 16, `5/1` 16, `1maj7/5` 14, `1/4` 11, `2/#4` 10, `1/7` 9, `6m/5` 9, `3(7)/#5` 7, `2m/1` 6, `5sus4/7` 6, `3m/5` 5, `1sus2/7` 4, `b7/2` 4, `4add9/6` 4

## Most common chord symbols (absolute, as written)

`G` 701, `D` 671, `C` 553, `A` 381, `E` 258, `Em` 238, `F` 216, `Am` 194, `Bm` 172, `F#m` 152, `G/B` 104, `B` 98, `Am7` 90, `D/F#` 87, `E/G#` 76, `Bb` 75, `A/C#` 72, `Em7` 68, `Csus2` 63, `F#m7` 54, `Eb` 54, `C#m` 53, `Dm` 52, `Ab` 43, `G#m` 42, `F#` 41, `Dsus2` 39, `Bm7` 38, `Cm` 37, `Dsus4` 37

## Progressions: most common chord-to-chord moves (bigrams)

Simple level (triads, repeats merged):

| Move | Count |
|---|---:|
| 5 → 1 | 574 |
| 4 → 1 | 546 |
| 1 → 5 | 476 |
| 4 → 5 | 473 |
| 1 → 4 | 460 |
| 5 → 6m | 320 |
| 6m → 4 | 319 |
| 5 → 4 | 294 |
| 6m → 5 | 229 |
| 2m → 5 | 170 |
| 1 → 6m | 165 |
| 4 → 6m | 160 |
| 1 → 2m | 134 |
| 3m → 4 | 99 |
| 5 → 3m | 84 |
| 3m → 6m | 82 |
| 2m → 4 | 76 |
| 6m → 2m | 74 |
| 6m → 1 | 73 |
| 5 → 2m | 68 |
| 4 → 2m | 65 |
| 6m → 3m | 64 |
| 2m → 1 | 61 |
| 1 → 3m | 40 |
| 2 → 5 | 32 |

Full detail (as written, with sus / 7ths / slash basses):

| Move | Count |
|---|---:|
| 5 → 1 | 329 |
| 4 → 1 | 294 |
| 4 → 5 | 279 |
| 1 → 5 | 254 |
| 6m → 4 | 239 |
| 5 → 4 | 194 |
| 5 → 6m | 161 |
| 1 → 4 | 160 |
| 6m → 5 | 137 |
| 1 → 5/7 | 123 |
| 4 → 1/3 | 104 |
| 4 → 6m | 95 |
| 1 → 6m | 93 |
| 5(7) → 1 | 89 |
| 1/3 → 4 | 85 |
| 3m → 4 | 81 |
| 2m → 5 | 77 |
| 5/7 → 6m | 75 |
| 3m → 6m | 62 |
| 5 → 3m | 56 |
| 1 → 6m7 | 56 |
| 1 → 4sus2 | 52 |
| 6m → 3m | 49 |
| 6m7 → 5 | 48 |
| 1/3 → 2m | 47 |

## Four-chord patterns (simple level)

| Pattern | Count |
|---|---:|
| 1 – 5 – 6m – 4 | 119 |
| 5 – 1 – 4 – 5 | 109 |
| 4 – 1 – 5 – 6m | 99 |
| 1 – 4 – 5 – 1 | 98 |
| 4 – 5 – 1 – 4 | 96 |
| 5 – 1 – 5 – 6m | 94 |
| 5 – 6m – 4 – 1 | 89 |
| 6m – 4 – 1 – 5 | 89 |
| 6m – 4 – 5 – 1 | 77 |
| 4 – 5 – 1 – 5 | 66 |
| 6m – 5 – 4 – 1 | 64 |
| 1 – 5 – 4 – 1 | 64 |
| 4 – 1 – 4 – 1 | 63 |
| 5 – 4 – 1 – 5 | 63 |
| 1 – 2m – 5 – 1 | 62 |

## Per language

| Language | Songs | Changes | sus | add9 | maj7 | slash |
|---|---:|---:|---:|---:|---:|---:|
| english | 62 | 3373 | 8.5% | 0.2% | 2.2% | 9.8% |
| cantonese | 40 | 1824 | 2.6% | 0.8% | 1.8% | 12.9% |
| mandarin | 21 | 700 | 1.1% | 0.9% | 1.9% | 21.0% |

## Suggested training weights (from these counts)

- Sample chord *qualities* roughly in proportion to the quality table above (plain triads dominate; sus, add9, 7ths and slash chords are the next tier), but keep a floor for rare ones so the model still sees them.
- Sample *progressions* from these charts (transposed to all 12 keys) rather than random chord pairs: the bigram tables show the moves real worship songs use (1–5–6m–4 family, 4→5, 5/7 and 1/3 passing basses).
- Slash chords are common enough that the bass note matters: render the written bass in the left hand.
- "2" chords (C2, G2) are read as sus2, except a 2 chord over its own third (D2/F#), read as add9.

