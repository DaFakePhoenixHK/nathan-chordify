# D1-draft — first test on Nathan's 5 priority songs (2026-11-01)

**Status: DRAFT answer keys, not checked by ear.** The chord *names* come from published sheets / charts; only
their *timing* was found by computer (aligned to the recording). Numbers below will move once Nathan (or a musician
friend) checks the "listen here" spots at the end.

## In plain words (for Nathan)

On these five worship songs the app gets the chord's **root note (the letter name) right about 89 % of the time**,
and right including major/minor about 85–89 % of the time. That is roughly the same as the plain open-source
model it is built on (B1), not better yet. It is very good on simple, steady songs (10,000 Reasons ≈ 97 %, Here I
Bow and 立志擺上 ≈ 94 %) and much weaker on 回家 (≈ 65–70 %), which is full of colourful chords (B+7, Em6, slash
chords) and whose "answer key" is itself the least certain. The extra colour in the names (7ths, maj7, add9) is right
only about 70 % of the time. Short passing chords (one beat or less) are the main weakness: the app catches about
half of them, and at "Detailed" most of the short chords it adds are wrong. In the 12/8 song 活出愛 the app loses
track of the beat, which makes it flicker between chords (about 10 wrong flashes per minute at Detailed; the
Simple view is much calmer). So: **use Simple view for playing**, and treat short chords at Detailed as hints.

## How the answer keys were made

| song | chord source | key (recording) | meter / beat | covered | how sure is the chart |
|---|---|---|---|---|---|
| 立志擺上 `yt-WLtqi4s29LQ` | 建道神學院 **編曲版主領譜** PDF (same arrangement as the video; image-only, read from the pages) | E♭ major | 4/4, ♩ ≈ 66 | whole song (92 bars) | high |
| 活出愛 `yt-lfP-3JLVvaw` | Clay Music official lead sheet PDF (text layer + images) | C major → D major (from 165 s) | 12/8, beat = dotted quarter ≈ 63 | all but 26 s (turnaround bar, ending tag) | high for sung parts; intro inferred (L) |
| Here I Bow `yt-hvHi7N8kc8s` | WorshipTogether free chord chart | D major | 4/4, ≈ 69 | all but 44 s (intro, outro) | medium (chart = chord order per lyric line) |
| 10,000 Reasons `yt-XtwIT8JjddM` | WorshipTogether free chord chart (header says C, chords are in G) | G major | 4/4, 72.5 | all but 33 s (intro, final-chorus ending + tag) | medium |
| 回家 `yt-Y7boUFbM76k` | existing hand-made draft chart (consensus of published charts) | B major | 4/4, ≈ 65 | whole song | medium–low (many bars only give the chord order) |

- **Key check**: the engine's key detector agrees with every sheet (E♭, C, D, G, B major), so nothing was transposed.
- **Order of sections**: written from the sheets' repeat signs plus a "section search" (`python -m eval.build_gt
  <key> --explore`) that matches the chart's sections to the engine's chord evidence. It confirmed: 立志擺上 plays
  the sheet straight through; 10,000 Reasons = intro, chorus, (verse, chorus) ×3, chorus, tag; Here I Bow = intro,
  V1, Ch, V2, Ch, Bridge ×2, Ch, outro; 活出愛 = intro, V, Ch, V, Ch (2nd ending), modulation, V, Ch, Ch (in D), tag.
- **Timing**: a DTW lays the chart's beats on the recording's beats, using the engine's main-layer chord evidence
  (lv-chordia + consonance-ACE, scored by chord family so Eb / Ebadd9 / Ebmaj7 count alike). GT beats = the engine's
  beats, except 活出愛 (the raw tracker's half-bar downbeats split in two, because the engine's own grid jumps
  between eighth notes and dotted quarters there).
- **Unknown parts**: stretches the published chart does not cover are labelled `X` and left out of every score
  (they are not guessed from the engine).
- Files: `eval/gt/<key>.lab` / `.beats` / `.chart.txt` (回家: `.chart.aligned.txt`; its hand-made chart is kept),
  `eval/songs.yaml` (all `reviewed: false`, `status: draft`), sources in `data/sheets/<key>/`.

**Alignment spot-check** — share of half-second samples where the chart chord and the engine's Simple chord have
the same root (and same major/minor): 立志擺上 93 % (92 %), 活出愛 92 % (88 %), Here I Bow 94 % (93 %), 10,000 Reasons
98 % (98 %), 回家 67 % (66 %). At every section start the chart chord matches the engine's root, except the start of
立志擺上 Chorus 3 (243.7 s, engine hears no chord) and the 回家 tag (222.6 s, quiet, engine hears no chord). For 回家
the computed section starts match the old hand-made times within 0.3 s except the tag (222.6 s vs 224.0 s).

**Caveat**: because the timing was aligned with the same evidence the engine uses, chord-*change* timing scores
(boundary F) are flattering; chord-*name* scores are fair (names come from the charts).

## Results

Metrics from `eval/chord_metrics.py` (mir_eval levels; "short" = reference chords ≤ 1 beat, majmin_inv level;
flicker = wrong chords shorter than ½ beat per minute; change F = chord-change F-measure, ¼-beat window).
Engine Detailed = Detail 0.8, Engine Simple = Detail 0 (main chords only), B1 = lv-chordia's own decoder.
Raw numbers: `eval/reports/D1-draft.json` (`python -m eval.d1_eval`).

#### 立志擺上 (`yt-WLtqi4s29LQ`) — scored 340 s

| prediction | root | majmin | majmin_inv | sevenths_inv | short P / R / F1 (≤1 beat) | flicker /min | bass acc | change F (¼ beat) |
|---|---|---|---|---|---|---|---|---|
| Engine Detailed | 0.937 | 0.934 | 0.893 | 0.632 | 0.20 / 0.60 / 0.30 (n=5) | 0.71 | 0.929 | 0.871 |
| Engine Simple | 0.936 | 0.932 | 0.902 | 0.640 | 0.19 / 0.20 / 0.19 (n=5) | 0.00 | 0.935 | 0.893 |
| B1 lv-chordia | 0.917 | 0.914 | 0.878 | 0.653 | 0.60 / 0.40 / 0.48 (n=5) | 0.00 | 0.930 | 0.858 |

#### 10,000 Reasons (`yt-XtwIT8JjddM`) — scored 228 s

| prediction | root | majmin | majmin_inv | sevenths_inv | short P / R / F1 (≤1 beat) | flicker /min | bass acc | change F (¼ beat) |
|---|---|---|---|---|---|---|---|---|
| Engine Detailed | 0.969 | 0.968 | 0.944 | 0.923 | 0.41 / 1.00 / 0.59 (n=4) | 0.92 | 0.948 | 0.870 |
| Engine Simple | 0.975 | 0.981 | 0.966 | 0.951 | 0.75 / 1.00 / 0.86 (n=4) | 0.00 | 0.970 | 0.912 |
| B1 lv-chordia | 0.963 | 0.968 | 0.962 | 0.962 | 0.86 / 1.00 / 0.92 (n=4) | 0.00 | 0.968 | 0.907 |

#### 回家 (`yt-Y7boUFbM76k`) — scored 247 s

| prediction | root | majmin | majmin_inv | sevenths_inv | short P / R / F1 (≤1 beat) | flicker /min | bass acc | change F (¼ beat) |
|---|---|---|---|---|---|---|---|---|
| Engine Detailed | 0.654 | 0.700 | 0.642 | 0.455 | 0.40 / 0.47 / 0.44 (n=17) | 0.00 | 0.712 | 0.705 |
| Engine Simple | 0.661 | 0.708 | 0.643 | 0.460 | 0.52 / 0.47 / 0.49 (n=17) | 0.00 | 0.707 | 0.723 |
| B1 lv-chordia | 0.715 | 0.739 | 0.669 | 0.465 | 0.69 / 0.41 / 0.52 (n=17) | 0.00 | 0.741 | 0.719 |

#### Here I Bow (`yt-hvHi7N8kc8s`) — scored 220 s

| prediction | root | majmin | majmin_inv | sevenths_inv | short P / R / F1 (≤1 beat) | flicker /min | bass acc | change F (¼ beat) |
|---|---|---|---|---|---|---|---|---|
| Engine Detailed | 0.948 | 0.923 | 0.898 | 0.800 | 0.06 / – / – (n=0) | 0.00 | 0.935 | 0.815 |
| Engine Simple | 0.948 | 0.927 | 0.906 | 0.808 | 0.08 / – / – (n=0) | 0.00 | 0.940 | 0.829 |
| B1 lv-chordia | 0.920 | 0.901 | 0.901 | 0.846 | – / – / – (n=0) | 0.00 | 0.931 | 0.812 |

#### 活出愛 (`yt-lfP-3JLVvaw`) — scored 259 s

| prediction | root | majmin | majmin_inv | sevenths_inv | short P / R / F1 (≤1 beat) | flicker /min | bass acc | change F (¼ beat) |
|---|---|---|---|---|---|---|---|---|
| Engine Detailed | 0.938 | 0.890 | 0.812 | 0.747 | 0.00 / 0.00 / 0.00 (n=2) | 10.31 | 0.843 | 0.671 |
| Engine Simple | 0.933 | 0.894 | 0.842 | 0.772 | 0.00 / 0.00 / 0.00 (n=2) | 1.89 | 0.875 | 0.729 |
| B1 lv-chordia | 0.935 | 0.920 | 0.891 | 0.847 | 0.00 / 0.00 / 0.00 (n=2) | 0.00 | 0.913 | 0.911 |

#### Overall (5 songs; standard scores weighted by scored seconds, short chords / flicker / bass pooled)

| prediction | root | majmin | majmin_inv | sevenths_inv | short P / R / F1 (≤1 beat, n=28) | flicker /min | bass acc |
|---|---|---|---|---|---|---|---|
| Engine Detailed | 0.891 | 0.885 | 0.839 | 0.701 | 0.20 / 0.54 / 0.29 | 2.45 | 0.874 |
| Engine Simple | 0.892 | 0.890 | 0.853 | 0.716 | 0.32 / 0.46 / 0.38 | 0.39 | 0.886 |
| B1 lv-chordia | 0.891 | 0.889 | 0.859 | 0.743 | 0.62 / 0.46 / 0.53 | 0.00 | 0.897 |

### What this says

- **Root / major-minor: engine ≈ B1** (0.89 both). The engine wins on 立志擺上 and Here I Bow, ties on 10,000
  Reasons, loses on 回家 and on the major/minor + inversion levels of 活出愛.
- **Detailed does not help yet on real songs**: it finds a few more short chords (recall 0.54 vs 0.46) but most
  of its short additions are wrong (precision 0.20), and it adds 2.45 wrong flashes per minute (almost all in
  活出愛). Only 28 short reference chords exist in these drafts, so short-chord numbers are rough.
- **7th / colour level is weak** (0.70–0.74 for everyone): 立志擺上 is written with maj7/add9/7sus colours that
  the models mostly hear as plain triads, and 回家's chart is full of 7ths, 6ths and slash chords.

### Problems found in the engine (reported, not changed — engine/ is not mine to edit)

1. **12/8 rhythm (活出愛)**: the rhythm grid switches between eighth-note beats (0.32 s) and dotted quarters
   (0.95 s), with ~10 "free time" stretches and bars flipping 2/4 ↔ 4/4 ↔ 3/4 in the first 30 bars
   (`cache/yt-lfP-3JLVvaw/analysis.json` → `meter`, `tempo.free_time`). The raw tracker's downbeats are steady
   (every 1.90 s = half a 12/8 bar). Result: 49 short wrong chords at Detailed (10.3/min) and chord-change F 0.67
   vs B1's 0.91. Needs a compound-meter (12/8 → 4 dotted-quarter beats) rule in `rhythm`.
2. **Here I Bow intro**: the grid runs at half tempo for the first ~14 s (beats 1.6 s apart), then 0.87 s.
3. **"No chord" inside the music**: 立志擺上 243.9–246.4 s (Chorus 3 start) and the quiet 回家 tag (222–245 s)
   come out as N; may be real (band drop-out) — see the listening list.
4. Key confidence is low for 立志擺上 (0.25) and 活出愛 (0.19, it modulates) although the key is right.

## Please check by ear (most useful first)

| song | time (s) | the chart says | the app hears | why it matters |
|---|---|---|---|---|
| Here I Bow | 35.3–42.2 | Em → A ("Beautiful, so beautiful", end of verse 1) | D the whole time | chart vs recording may differ |
| Here I Bow | 136.2–139.7, 164.0–167.5 | Dsus4 (bridge "lifted high") | D | sus4 or plain D? |
| 10,000 Reasons | 168.9–170.5 | C (verse 3 "sing Your praise unending") | D | chart says C; recording may play D |
| 10,000 Reasons | 0–13.3 | intro "C G D/F# Em C G Dsus D" (2 beats each) did not fit → left unknown | G, C G D C D C G | fill in the real intro |
| 10,000 Reasons | 36.5, 92.7, 149.0, 208.6 (last bar of choruses 1–4) | Gsus – G | G (sus only in chorus 1) | sus or not |
| 10,000 Reasons | 235.1–255.1 | unknown (final chorus ending + tag) | Em, then C D Em, C D G | fill in |
| 立志擺上 | 243.7–246.4 | E♭add9 (Chorus 3 "立志今擺上") | no chord | does the band stop here? |
| 立志擺上 | 305.5–311 | G7♭9/B (2 beats) → Cm (held) | B° → Cm | timing of the 2/4 bar + pause |
| 活出愛 | 1–31.6 | intro written as the verse chords (inferred) | same, bar 6 as F/A → C/G | confirm the intro |
| 活出愛 | 92.6–96.4, 258.6–281 | unknown (turnaround bar; ending tag) | Dm7 → G7sus4; Bm, Em A D … | fill in |
| 回家 | 6.8–10.3 | G♯m/D♯ (intro) | D♯m7 | intro is the least sure part of the chart |
| 回家 | 77–80.5, 143.5–147 | C♯m7/E | E / Emaj7 | nearly the same notes; which name? |
| 回家 | 222–245 | tag B B7 \| A9 \| B/F♯ \| C♯m7 \| E/F♯ \| B | mostly "no chord" (quiet) | timing of the tag |
