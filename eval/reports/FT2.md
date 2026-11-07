# FT2 — plain chords, piano near middle C, simple recordings (2026-11-01)

## In plain words (for Nathan)

We tried your idea. The practice songs were rebuilt to be **simpler**:

- about **8 in 10 chords are plain major or minor**; the only "colour" chords left are add, sus, dim, 7 and 9
  chords (exactly the list you gave);
- the piano plays **near middle C**: the right hand stays between the F below middle C and the A two octaves up
  (most of the time within a few notes of middle C), the left hand plays one bass note, mostly in the octave
  just below middle C, never lower than the G below that;
- **almost half of the songs are piano alone**, about a third add a soft background (strings or a singing
  voice), and only a quarter have drums and bass guitar;
- there are more **silent moments** (before the song starts, short rests in the middle, endings), so the model
  learns to say "no chord" when nobody plays.

You can listen to 20 examples in `data/train/preview2/` (the list of chords and the notes of each hand are in
`labels.csv`).

We then retrained the chord-hearing model (ACE) twice:

- **Version FT2-a (retrain most of the model)**: this is the big change compared with last time. On your 5 real
  songs it now gets the **root note right more often (89.7 % → 90.3 %)**, the **bass note more often (88.1 % →
  89.3 %)**, and its short "passing" chords are **right more often (21 % → 24 %)**. Last time this kind of
  retraining made the 7th / 9th names clearly worse; now the plain major/minor and 7th scores are almost exactly
  as before (−0.1 and −0.2 points). It helped 回家 a lot (root +5.5 points) and 10,000 Reasons' 7th chords
  (+2.4), but Here I Bow still got a bit worse (7th chords −6.9 points).
- **Version FT2-b (retrain only the root / bass part)**: almost the same as last time's version B; slightly
  better on major/minor and 7ths, but its short passing chords got a little less accurate.

So your simpler recipe **clearly helped** — the "retrain most" version no longer damages real songs the way it
did before. But neither version beats the original on **every** score we agreed to check, so **the app keeps the
original model** for now. FT2-a is the best candidate so far; it can be switched on with one setting
(`ACE_WEIGHTS=ft2-a`) for trying, and the next step is to make more of these simpler songs and check on more
real recordings (or your reviewed answer keys) before deciding.

(One practical note: the computer's disk got too full during this work — an antivirus scan and a growing
Windows page file, not our files — so the practice songs were kept in memory instead of being saved to disk,
and we made 6,000 of them instead of 12,000.)

## 1. The recipe (the user's idea, `train/song_mix_v2.py`)

Everything is new code; FT1's files, pool, preview, runs and the original checkpoint were not touched
(`train/song_mix_v2.py` reuses FT1's building blocks read-only, `train/ft2.py` drives
`train/finetune_ace.py` as a library).

**Chords.** Plain major + minor ≈ 80 %, colours only from this list:
add9, madd9 · sus2, sus4, 7sus4 · dim, dim7 · 7, maj7, m7 · 9, maj9, m9. Everything else is mapped to the
nearest allowed chord, keeping the bass when the new chord allows it: 6→maj, m6→m, 11 (= 9sus4)→7sus4,
13sus4→7sus4, 13→9, aug→maj, aug7→7, aug-maj7→maj7, m7b5→dim, m(maj7)→m, 6/9→add9, m11→m7,
7b9 / 7#9 / 7b13→7, maj7#11→maj7, 5 (power chord)→maj. No song is dropped.
Sources: harvested charts 58 % (123 songs + 41 alternative versions, random key, 70 % new chord lengths when
the chart's bar lengths were guessed), a worship diatonic generator 35 % (colour table: I = maj 90 %, ii/iii/vi =
min 86 %, V = maj 84 %; a passing chord into 10 % of the changes; slash chords I/3, I/5, V/3, IV/5, vi/5, ii/4),
the 32 idiom progressions mapped to the list 4 %, random allowed chords 3 %.

**Register** (MIDI numbers; middle C = C4 = 60):
- right hand inside 53–81 (F3–A5); its inversion and octave are drawn with weight
  exp(−|mean pitch − target| / 2), target ~ N(61, 3), × 0.25 when no C3–B3 bass note fits underneath;
- left hand = the bass note: C3–B3 (48–59) preferred (weight 1), G2–B2 (43–47) only when needed (weight 0.12),
  never below G2; no octave doublings (a second left-hand note, 20 % of songs, is a different chord tone);
  one-hand playing (20 %) puts the bass at 50–62;
- synth bass (band songs only) E2–G3 (40–55), 6–14 dB below the piano.

**Recordings.** Piano alone 45 %; piano + soft pad (−20…−9 dB) and/or voice 30 %; band 25 % (drums + bass,
pad 50 %, voice 70 %; half of the two-section band songs start with piano alone). Reverb RT60 0.3–2.5 s, noise
−70…−40 dB in 40 %, limiter in 25 % (FT1: up to −30 dB noise, 40 % limiter). Meters as FT1 (4/4 60 %,
3/4 22 %, 12/8 18 %).

**No-chord stretches.** Silent lead-in in 60 % of songs (0.3–2.5 s); 1–2 rests inside the music in 35 % (1 beat
to 1 bar: keys and pedal up, pad / bass / voice stop, drums go on in 40 % of those songs); a real ending in 35 %.
A chord before a rest keeps its label for 0.25 s (its decay), then N.

**Preview first**: 20 clips in `data/train/preview2/` (FLAC + `labels.csv` with every chord, the left- and
right-hand notes, arrangement, meter, tempo) were rendered before the pool, for Nathan to listen to.

**What came out** (`models/ace_ft/ft2_pool_stats.json`):

| | train | val (own seed) |
|---|---|---|
| items / hours | 6,000 / 39.3 h | 700 / 4.6 h |
| chords | 74,201 | 8,644 |
| plain major + minor | **79.6 %** (maj 58.2, min 21.4) | 78.4 % |
| colours | m7 5.8, 7 3.2, sus4 3.1, sus2 2.2, maj7 1.7, add9 1.0, dim7 0.7, dim 0.5, 7sus4 0.5, madd9 0.5, m9 0.4, maj9 0.4, 9 0.4 % | similar |
| outside the allowed list | none | none |
| slash chords | **13.3 %** (bass = 3rd 5,795, 5th 1,179, 9th over the root = IV/5 type 960, b7 883, b3 579, 4th 211) | 10.5 % |
| sources | charts 3,512 · diatonic 2,093 · idioms 220 · random 175 | 418 · 235 · 27 · 20 |
| arrangement | piano alone 2,723 (45 %) · soft 1,792 (30 %) · band 1,485 (25 %) | 292 · 225 · 183 |
| meter | 4/4 3,541 · 3/4 1,346 · 12/8 1,113 | 417 · 158 · 125 |
| no-chord (N) frames | 9.8 % (FT1 val: 4.6 %), 2,610 rests | 9.9 %, 316 rests |

Register actually played (per item, [min, 5 %, median, 95 %, max] over items, MIDI): lowest bass note
[43, 43, 45, 50, 56]; highest bass note [45, 52, 57, 61, 67]; right hand lowest [52, 53, 55, 59, 65], highest
[60, 67, 74, 80, 81], mean [57, 60, 64, 67, 72]; synth bass lowest [40, 40, 41, 46, 51], highest [40, 48, 53, 55,
55]. Over all chords (a 150-song sample of the same recipe): bass in C3–B3 74 %, G2–B2 21 %, C4 and above 5 %
(one-hand playing and high right hands); median right-hand pitch 63 (E♭4), 5–95 % 57.5–68.7. Nothing below G2.
(FT1 for comparison: bass mostly around F2 = 41, right hand up to E6 = 88.)

**Where the pool lives.** The coordinator's floor was "keep ≥ 20 GB free on C:". Between 14:13 and 14:28 C: fell
from 33 GB to ≈ 20 GB and then swung between 3.5 and 19 GB: Windows grew its page file (now 33 GB allocated)
and the antivirus (`endpointprotection.exe`, Surfshark — left running, as the user's rules require) wrote up to
700 MB per 15 s during a scan. My on-disk build stopped itself at the floor after 2 shards; two stray training
processes from that first attempt were stopped too (they had added memory pressure). I then kept the pool **in
RAM** (`train/ft2.py all-in-ram`: 4 worker processes with a smaller piano cache, 1.83 GB train + 0.21 GB val,
rendered in 34 + 5 min; items are deterministic by seed, so the same pool can be re-rendered any time) and cut
it to 6,000 train items. Nothing of the pool is on disk; C: was back at ≈ 23 GB free at the end.

## 2. Training (same safeguards as FT1, BatchNorm fix on)

Both runs start from the original checkpoint with FT1's objective (class-softmax CE with ACE's consonance
smoothing for root and bass, BCE × 2 for chord tones), L2-SP 0.01, batch 8, AdamW with 300-step warm-up and
cosine decay, ±2-semitone CQT shifts, BatchNorm statistics kept (`bn_eval`), early stopping on the FT2 val loss.

| run | trainable | lr | best step / ran | GPU min | FT2 val root / bass / chord tones exact / N recall |
|---|---|---|---|---|---|
| original | – | – | – | – | 0.701 / 0.667 / 0.545 / 0.955 |
| **ft2-a** (input projection + first 2 of 6 layers frozen) | 3.03 M | 7e-5 | 7,000 / 9,000 | 21 | **0.941 / 0.922 / 0.839** / 0.966 |
| **ft2-b** (only the root + bass branches) | 49.6 k | 1e-4 | 5,500 / 6,000 | 7 | 0.867 / 0.836 / 0.545 / 0.828 |

ft2-a's curve: step 500 root 0.929 / chord tones 0.811, 1,000 0.934 / 0.826, 3,000 0.939 / 0.837, then flat. The
original is much weaker on this pool than on FT1's (root 0.70 vs 0.82, bass 0.67 vs 0.87): it is not used to
a piano bass near middle C (piano alone: root 0.63; with a band: 0.81).

**Old FT1 val set** (FT1's recipe; the same 700 items as in FT1.md):

| model | root | bass | chord tones exact | N recall |
|---|---|---|---|---|
| original | 0.820 | 0.873 | 0.536 | 0.959 |
| ft1-b | 0.875 | 0.933 | 0.536 | 0.866 |
| ft2-a | 0.902 | 0.955 | 0.657 | 0.989 |
| ft2-b | 0.876 | 0.934 | 0.536 | 0.843 |

(ft1-a on its own val set, for reference: 0.897 / 0.953 / 0.681.) So ft2-a, trained only on the simple
recipe, also handles FT1's busier, lower, more colourful songs about as well as ft1-a did.

**T_ace** (class-level NLL on 350 FT2 val items, known beats): ft2-a **0.452**, ft2-b **0.602** (the original
would want 1.048 on this pool; shipped 0.5801). ft1-b keeps its FT1 value 0.6267.

## 3. Gate — D1 (third use) and mini-D2

D1 has now been used three times (ft1-a, ft1-b, FT2). This third use is allowed because FT2's design comes from
the user's listening, not from D1 results; nothing in FT2 (recipe, hyper-parameters, checkpoints, temperatures)
was chosen by looking at D1. Both FT2 runs and the two references were decoded in one process from the same
decode states (current rhythm code); only the ACE evidence and T_ace differ.

**D1 overall** (5 songs, DRAFT answer keys; `eval/reports/FT2-d1-<arm>.json`, `models/ace_ft/ft2_gate.json`):

| prediction | T_ace | root | majmin_inv | sevenths_inv | short P / R / F1 (≤ 1 beat, n = 28) | flicker /min | bass acc |
|---|---|---|---|---|---|---|---|
| Detailed, original | 0.5801 | 0.8965 | 0.8461 | 0.7076 | 0.211 / 0.536 / 0.303 | 2.45 | 0.8814 |
| Detailed, ft1-b | 0.6267 | 0.9010 | 0.8491 | 0.7090 | 0.217 / 0.536 / 0.309 | 2.49 | 0.8887 |
| Detailed, **ft2-a** | 0.4518 | **0.9031** | 0.8448 | 0.7054 | **0.237** / 0.536 / **0.329** | 2.53 | **0.8933** |
| Detailed, **ft2-b** | 0.6020 | 0.9004 | **0.8508** | **0.7103** | 0.204 / 0.536 / 0.295 | 2.49 | 0.8911 |
| Simple, original | | 0.8981 | 0.8625 | 0.7248 | 0.386 / 0.464 / 0.422 | 0.04 | 0.8954 |
| Simple, ft1-b | | 0.9022 | 0.8652 | 0.7278 | 0.389 / 0.464 / 0.423 | 0.04 | 0.9031 |
| Simple, ft2-a | | 0.9040 | 0.8607 | 0.7204 | 0.383 / 0.464 / 0.420 | 0.04 | 0.9070 |
| Simple, ft2-b | | 0.9016 | 0.8671 | 0.7288 | 0.368 / 0.464 / 0.411 | 0.04 | 0.9058 |

**Gate** (FT1's rule at Detailed: majmin_inv and sevenths_inv up, root not down, short P not down):

| | vs original | vs ft1-b |
|---|---|---|
| ft2-a | **FAIL**: root +0.0066, majmin_inv −0.0013, sevenths_inv −0.0022, short P +0.0257 | FAIL: root +0.0021, majmin_inv −0.0043, sevenths_inv −0.0036, short P +0.0200 |
| ft2-b | **FAIL**: root +0.0039, majmin_inv +0.0047, sevenths_inv +0.0027, short P −0.0079 | FAIL: root −0.0006, majmin_inv +0.0017, sevenths_inv +0.0013, short P −0.0137 |

At Simple: ft2-a root +0.0059, majmin_inv −0.0018, sevenths_inv −0.0044, short P −0.0032, bass +0.0116;
ft2-b +0.0035 / +0.0046 / +0.0040 / −0.0185 / +0.0103.

Per song, Detailed root / majmin_inv / sevenths_inv / bass acc:

| song | original | ft1-b | ft2-a | ft2-b |
|---|---|---|---|---|
| 立志擺上 | .939 / .894 / .633 / .929 | .940 / .895 / .633 / .933 | .930 / .881 / .618 / .927 | .941 / .896 / .634 / .937 |
| 10,000 Reasons | .969 / .944 / .923 / .948 | .973 / .945 / .925 / .950 | .973 / .947 / **.947** / .952 | .973 / .947 / .927 / .952 |
| 回家 | .654 / .642 / .455 / .712 | .678 / .664 / .477 / .738 | **.709 / .671 / .488 / .761** | .682 / .668 / .479 / .742 |
| Here I Bow | .948 / .898 / .800 / .935 | .936 / .889 / .782 / .940 | .930 / .874 / *.731* / .952 | .933 / .885 / .779 / .940 |
| 活出愛 | .965 / .849 / .779 / .879 | .969 / .848 / .778 / .880 | .968 / .848 / .792 / .878 | .965 / .851 / .782 / .882 |

**ACE alone on D1** (frames with a GT chord; no decoder): root / bass / all chord tones exact —
original .831 / .818 / .555; ft1-b .847 / .837 / .555; **ft2-a .855 / .854 / .558**; ft2-b .851 / .841 / .555.
ft2-a has the best root and bass of all four, and — unlike ft1-a (.476) — its chord-tone head no longer gets
worse on real songs. Root per song for ft2-a: 立志擺上 .855 (=), 10,000 Reasons .932 (+1.9), 回家 .706 (+8.2),
Here I Bow .870 (+2.3), 活出愛 .918 (+0.4).

**mini-D2** (72 synthetic items, additive-synth band; Detailed):

| | majmin_inv | sevenths_inv | root | short F1 mm_inv (P / R) | flicker /min | bass acc | boundary F |
|---|---|---|---|---|---|---|---|
| original | 0.925 | 0.808 | 0.924 | 0.623 (0.82 / 0.50) | 0.28 | 0.981 | 0.877 |
| ft1-b | 0.917 | 0.804 | 0.917 | 0.618 (0.81 / 0.50) | 0.28 | 0.983 | 0.865 |
| ft2-a | 0.919 | **0.841** | 0.919 | **0.656** (0.81 / 0.55) | 0.46 | **0.987** | 0.877 |
| ft2-b | 0.911 | 0.795 | 0.912 | 0.615 (0.81 / 0.50) | 0.32 | 0.982 | 0.861 |

## 4. Decision

- **Neither FT2 run passes the gate**, against the original or against ft1-b. `DEFAULT_WEIGHTS` stays None (not
  changed, as instructed); both runs are opt-in only.
- **Recommended**: keep the original as the default. Treat **ft2-a as the lead candidate** for the next step,
  because the user's recipe fixed the main FT1 failure: retraining most of the network no longer hurts real songs
  much (majmin_inv −0.1 pt, sevenths_inv −0.2 pt vs ft1-a's −0.8 / −1.8), while root (+0.7 pt), bass (+1.2 pt),
  short-chord precision (+2.6 pt at Detailed) and ACE's own root / bass on all real songs improve. Its remaining
  loss is concentrated in Here I Bow (sevenths −6.9 pt) and 立志擺上 (−1.5 pt). ft2-b adds nothing over ft1-b.
- To decide ft2-a for real, without touching D1 again: (1) more real songs or Nathan-reviewed answer keys;
  (2) the full-size pool (12,000+ items) once C: has room again — this run had 6,000; (3) listen to Here I Bow
  with ft2-a's chords to see which chord tones it adds there.
- Not changed: ft2-a raises flicker on mini-D2 (0.28 → 0.46 /min) and slightly at Detailed on D1 (2.45 → 2.53);
  Simple view is unaffected (0.04).

## 5. Files

- New code: `train/song_mix_v2.py` (recipe, preview, in-RAM / on-disk pool), `train/ft2.py` (driver: RAM pool,
  training, calibration, comparisons, gate), tests `train/tests/test_song_mix_v2.py`, `train/tests/test_ft2.py`.
- Preview: `data/train/preview2/` (20 FLAC + labels.csv + labels.json).
- Pool statistics: `models/ace_ft/ft2_pool_stats.json` (the pool itself lived in RAM, see §1); run log
  `models/ace_ft/ft2_all_in_ram.log`; the stopped on-disk attempt's logs in `data/train/pool2/`.
- Runs: `models/ace_ft/ft2-a/`, `models/ace_ft/ft2-b/` (best.pt, last.pt, config.json, log.jsonl, ft2.json,
  val_best.json, calibration.json); comparisons `models/ace_ft/ft2_val_compare.json`, `models/ace_ft/ft2_gate.json`;
  D1 tables `eval/reports/FT2-d1-{original,ft1-b,ft2-a,ft2-b}.json`; manifest: new `"ace_ft"` entries for ft2-a
  and ft2-b (FT1's entries unchanged).
- Use a run: `ACE_WEIGHTS=ft2-a` (or `ft2-b`) + `DecoderParams(temp={..., "ace": <its calibration.json T>})`.
  `models/ace.py` already resolves any run id under `models/ace_ft/`; it was not edited for FT2, and
  `DEFAULT_WEIGHTS` stays None.
