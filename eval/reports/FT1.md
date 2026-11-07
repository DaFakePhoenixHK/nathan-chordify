# FT1 — fine-tuning consonance-ACE on rendered worship-piano songs (2026-11-01)

## In plain words (for Nathan)

We taught one of the app's two "chord-hearing" models (consonance-ACE) with **78 hours of made-up worship
songs** played on a real sampled grand piano (Salamander) with a pretend band: bass, string pad, drums and a
humming "voice". Because we wrote the songs ourselves, we know every chord exactly. The chords came from 123
real worship songs' chord charts (in all 12 keys), plus typical worship patterns and random chords.

We tried two versions.

**Version A (retrain most of the model)**: on **made-up songs it hasn't seen** it got much better (root note
right 82 % → 90 %, bass note 87 % → 95 %, all chord notes right 54 % → 68 %). But on your **5 real
recordings** it was **not better overall**:

- it **helped** the piano-heavy, colourful songs: 回家 (root +4.6 points, bass note +5.8) and the 7th/9th
  colours of 立志擺上 (+2.8);
- it **hurt** the plain band songs Here I Bow and 10,000 Reasons: it started "hearing" an extra note (a 7th,
  9th or sus note) that is not in the chord, so plain chords like D or G got wrong colour names more often.

The reason: our made-up songs had **too many colourful chords** (only about half plain major/minor chords,
while real worship charts are 82 % plain).

**Version B (retrain only the small "which letter is the root / which note is in the bass" part, keep the
rest exactly as it was)**: on your 5 songs it is **a tiny bit better** on every score we check (root +0.5
points, bass note +0.7, major/minor +0.3, 7th colours +0.1) — mostly thanks to 回家 (+2.4 root); Here I Bow got
a little worse (−1.2 root). It passes the test we set, but only just: the differences are about the size of the
uncertainty in our draft answer keys, and on the old synthetic test set it is slightly worse (it says "no
chord" less often at the start and end of a piece). **So the app still uses the original model by default**;
version B can be switched on with one setting (`ACE_WEIGHTS=ft1-b`) and should be switched on for good only
after it is checked on more real songs (or on your reviewed answer keys).

The lesson for next time: make the made-up songs look like real worship charts (mostly plain chords), then
retrain more of the model.

## 1. Training data (`train/song_mix.py`, pool in `data/train/pool/`)

One item = a 15–30 s excerpt of a made-up song, labelled per ACE frame (23.2 ms): root pc or N, bass pc or N,
12-note chroma of the chord (template tones ∪ bass, = `ChordSpec.pcs`).

| | train | val (fixed, own seed) |
|---|---|---|
| items / hours | 12,000 / 78.3 h | 700 / 4.6 h |
| ACE frames | 12.40 M | 0.72 M |
| chords (passing ≤ 1 beat) | 173,142 (25,739) | 9,643 (1,377) |
| progression source | chart 4,638 · diatonic generator 4,053 · idiom 2,090 · random vocabulary 1,219 | 297 · 207 · 119 · 77 |
| meter | 4/4 7,132 · 3/4 2,688 · 12/8 2,180 | 419 · 159 · 122 |
| piano layout | LH+RH 7,435 · two-hand block 3,112 · one hand 1,453 | 422 · 184 · 94 |
| piano style (per section) | arpeggio 5,483 · block 4,553 · pulse 4,109 · bass-chord 3,630 · rolled 2,673 | |
| band | full band (bass+drums+pad+voice) 5,256 · piano only 978 · piano+voice 435 · other mixes 5,331 | 301 · 49 · 20 · 330 |
| size on disk | 3.70 GB | 0.22 GB (total 3.91 GB < 8 GB cap) |

- **Chords**: 123 harvested worship charts + 41 alternative versions (`data/charts/`, the 5 D1 songs excluded),
  transposed to a random key; their bar lengths were inferred from lyrics, so 70 % of chart items get new chord
  lengths (whole / half bars, some 2-bar holds and 1-beat changes; chord ORDER kept). The 32 idiom progressions
  (passing °7, walk-ups, IV/5, sus resolutions, 回家's pre-chorus, …), a worship diatonic generator (I–V–vi–IV with
  add9/maj7/sus colours, slash basses, secondary dominants, °7, borrowed iv, 18 % minor keys) and random chords
  of the whole 1,668-chord vocabulary (all 32 qualities stay represented).
- **Quality mix (train, share of chords)**: maj 38 %, min 13.5 %, m7 8.1 %, 7 6.4 %, sus4 4.5 %, maj7 3.8 %,
  add9 3.1 %, sus2 2.5 %, dim7 2.3 % … rarest +maj7 626 chords. Inversions: root 82 %, chord-tone bass 16 %,
  foreign bass 2 %. **Real charts are 82 % plain triads** (eval/reports/charts_stats.md) — our mix is far more
  colourful (≈ 52 % triads), which turned out to matter (§4).
- **Sound**: Salamander piano at 22.05 kHz (voicings + styles from `train/chords.py`, legato pedal, sometimes
  lifted late so chords bleed), bass / pad / drums / voice synths of `eval/synth/render.py` (the voice sings chord
  tones plus 20 % appoggiaturas), 1–3 arrangement sections (bands come in later), reverb RT60 0.3–3 s, EQ
  tilt / high-pass / low-pass, noise −65…−30 dB, limiter. 2,000 train + 100 val items are "piano-focused"
  (45 % piano alone, 35 % piano + voice).
- **Features**: ACE's own |CQT| (144 bins, hop 512) of the whole excerpt, float16, plus the per-frame peak; a
  training crop of 862 frames is divided by its own peak = ACE's per-window normalisation (unit test checks it
  against the adapter's features: < 2 % mean difference away from the window edges).
- Rendering: ≈ 2–3 s per item per process, 7 processes, ≈ 75 min wall time in total.
- Caveat: the first val shards were rendered before the chart split was made stable (crc32 of the song slug), so
  some chart *progressions* occur in both train and val (different keys, tempi and sounds). This only affects the
  synthetic validation numbers.

## 2. Fine-tuning (`train/finetune_ace.py`)

- Start: the original checkpoint (`conformer_decomposed_smooth.ckpt`, sha256 6ab1a314…). Objective per frame:
  root and bass cross-entropy over the 13 classes with ACE's consonance label smoothing (α 0.075), chord tones
  BCE × 2. (ACE's own `ConsonanceDecomposedLoss` normalises its log-softmax over *time*, not over the 13 classes;
  we train the class softmax our adapter actually uses. ACE's cardinality term has no gradient and is dropped.)
- Against forgetting: input projection + first 2 of 6 Conformer layers frozen (3.03 M of 4.54 M weights
  trainable), L2-SP λ = 0.01 towards the original weights, AdamW lr 7e-5 (300-step warm-up, cosine), batch 8
  (16 needs 2.9 GB VRAM — the GPU was shared with another worker), bf16 autocast, random ±2-semitone CQT shifts.
- Early stopping on the synthetic val loss (patience 6 evaluations of 500 steps).
- Found after ft1-a: ACE's Conformer convolution blocks use BatchNorm, and in train mode their running
  statistics were re-estimated on the synthetic audio — in the "frozen" layers too. So ft1-a also shifted the
  feature normalisation of the whole network towards synthetic sound. The code now keeps frozen layers in eval
  mode and, by default (`bn_eval`), every BatchNorm's original statistics; ft1-b ran that way.

**Run ft1-a** (`models/ace_ft/ft1-a/`, log.jsonl): best at step 4,000 (2.7 epochs), stopped at 7,000;
**54 GPU-minutes**. Most of the gain comes in the first 500 steps, then it plateaus:

| step | val loss | root acc | bass acc | chord-tone exact | chord-tone F1 |
|---|---|---|---|---|---|
| 0 (original) | 3.535 | 0.820 | 0.873 | 0.536 | 0.846 |
| 500 | 3.046 | 0.887 | 0.945 | 0.657 | 0.910 |
| 1,500 | 2.995 | 0.893 | 0.950 | 0.675 | 0.918 |
| **4,000 (best)** | **2.967** | **0.897** | **0.953** | **0.681** | **0.921** |
| 7,000 (last) | 2.972 | 0.896 | 0.952 | 0.681 | 0.921 |

Synthetic val by group (root acc original → ft1-a): piano only 0.69 → 0.84, with band 0.83 → 0.90; 12/8 0.78 →
0.91; chart progressions 0.87 → 0.95, random chords 0.56 → 0.62 (many are sound-alike, e.g. C6 = Am7/C).
N recall 0.96 → 0.98; root NLL 0.87 → 0.62 nats.

**Temperature** (`decode/calibrate.fit`, class-level NLL on 350 synthetic val items, known beat grids):
ft1-a T_ace = **0.481** (NLL 2.29 at T = 1 → 1.84); the original on the same items would want 0.728 (shipped
0.5801, fitted on mini-D2).

**Run ft1-b** (`models/ace_ft/ft1-b/`) — the second and last variant, chosen after ft1-a's D1 diagnosis (its
chord-tone head was the part that failed on real songs, its bass head the part that improved): input projection,
all 6 Conformer layers and the chord-tone head frozen with their BatchNorm statistics, so the chord-tone output is
bit-identical to the original (unit test); only the root and bass branches train (49,562 weights), lr 1e-4,
L2-SP 0.01, batch 8, patience 4. Best at step 4,500, ran to 6,000; **23 GPU-minutes**. (A first ft1-b attempt
that still re-estimated BatchNorm statistics was stopped at step 2,500 and discarded.)

| step | val loss | root acc | bass acc | chord-tone exact | N recall |
|---|---|---|---|---|---|
| 0 (original) | 3.535 | 0.820 | 0.873 | 0.536 | 0.959 |
| 500 | 3.319 | 0.868 | 0.926 | 0.536 | |
| 2,000 | 3.288 | 0.874 | 0.932 | 0.536 | |
| **4,500 (best)** | **3.284** | **0.875** | **0.933** | 0.536 | 0.866 |

T_ace (same 350 val items) = **0.627** (NLL 2.57 → 2.33).

Total GPU time for FT1: ≈ 54 + 23 min training (+ 10 min of the discarded attempt) + ≈ 25 min calibration /
evidence; the GPU was shared with another worker throughout.

## 3. Gate on real songs (D1, current rhythm code) and mini-D2

Both arms decoded in the same process from the same (re-analysed, new-rhythm) decode states; only the ACE
evidence (mix + harmonic stem) and T_ace differ. Original = shipped weights, T 0.5801; ft1-a = its weights,
T 0.481. Raw: `eval/reports/FT1-d1-original.json`, `FT1-d1-ft1-a.json`, `models/ace_ft/ft1-a/gate.json`.

**D1 overall (5 songs, DRAFT answer keys)**

| prediction | root | majmin_inv | sevenths_inv | short P / R / F1 (≤ 1 beat, n = 28) | flicker /min | bass acc |
|---|---|---|---|---|---|---|
| Detailed, original ACE | 0.897 | **0.846** | **0.708** | 0.21 / 0.54 / 0.30 | 2.45 | 0.881 |
| Detailed, ft1-a | **0.904** | 0.838 | 0.690 | 0.21 / 0.50 / 0.30 | 2.45 | **0.900** |
| Simple, original ACE | 0.898 | 0.863 | 0.725 | 0.39 / 0.46 / 0.42 | 0.04 | 0.895 |
| Simple, ft1-a | 0.905 | 0.854 | 0.704 | 0.32 / 0.46 / 0.38 | 0.04 | 0.913 |
| B1 lv-chordia (reference) | 0.891 | 0.859 | 0.743 | 0.62 / 0.46 / 0.53 | 0.00 | 0.897 |

**Gate (Detailed): majmin_inv −0.008, sevenths_inv −0.018, root +0.007, short P −0.001 → FAIL.**

Per song (Detailed, original → ft1-a):

| song | root | majmin_inv | sevenths_inv | bass acc |
|---|---|---|---|---|
| 立志擺上 | 0.939 → 0.940 | 0.894 → 0.892 | 0.633 → **0.661** | 0.929 → 0.944 |
| 10,000 Reasons | 0.969 → 0.969 | 0.944 → 0.917 | 0.923 → *0.882* | 0.948 → 0.965 |
| 回家 | 0.654 → **0.700** | 0.642 → **0.664** | 0.455 → **0.483** | 0.712 → **0.770** |
| Here I Bow | 0.948 → 0.935 | 0.898 → *0.852* | 0.800 → *0.664* | 0.935 → 0.940 |
| 活出愛 | 0.965 → 0.967 | 0.849 → 0.853 | 0.779 → 0.776 | 0.879 → 0.881 |

**Why** (ACE alone, no decoder, frames with a GT chord, `gate.json` → ace_alone_d1_*): root 0.831 → 0.831,
bass **0.818 → 0.845**, all chord tones right 0.555 → **0.476**. The fine-tuned chord-tone head switches on
more notes (3.3–3.6 per frame vs 3.0–3.2; GT 3.0 on the triad songs): false extra tones per frame on Here I
Bow 0.60 → 1.00 and 10,000 Reasons 0.24 → 0.60, but fewer missed tones on the colourful 回家 (1.17 → 0.76)
and 立志擺上 (0.92 → 0.67). That matches the colour-heavy training mix (§1).

**mini-D2** (72 items, the additive-synth band — a timbre the model never trained on; Detailed):

| | majmin_inv | sevenths_inv | root | short F1 mm_inv (P / R) | short F1 sev_inv | flicker /min | bass acc | out-of-chord bass |
|---|---|---|---|---|---|---|---|---|
| original ACE | 0.925 | 0.808 | 0.924 | 0.623 (0.82 / 0.50) | 0.420 | 0.28 | 0.981 | 0.690 |
| ft1-a | 0.925 | **0.859** | 0.923 | **0.673** (0.84 / 0.56) | **0.460** | 0.56 | 0.986 | 0.721 |

So the gain is real on synthetic audio of any timbre, but it does not carry over to real band recordings.

### ft1-b (second and last D1 use)

Original = shipped weights, T 0.5801; ft1-b = its weights, T 0.627. Raw: `eval/reports/FT1-d1-ft1-b.json`,
`models/ace_ft/ft1-b/gate.json`.

| prediction | root | majmin_inv | sevenths_inv | short P / R / F1 (n = 28) | flicker /min | bass acc |
|---|---|---|---|---|---|---|
| Detailed, original ACE | 0.897 | 0.846 | 0.708 | 0.211 / 0.536 / 0.303 | 2.45 | 0.881 |
| **Detailed, ft1-b** | **0.901** | **0.849** | **0.709** | **0.217** / 0.536 / **0.309** | 2.49 | **0.889** |
| Simple, original ACE | 0.898 | 0.863 | 0.725 | 0.386 / 0.464 / 0.422 | 0.04 | 0.895 |
| **Simple, ft1-b** | **0.902** | **0.865** | **0.728** | **0.389** / 0.464 / **0.423** | 0.04 | **0.903** |

**Gate (Detailed): root +0.0045, majmin_inv +0.0030, sevenths_inv +0.0014, short P +0.0057 → PASS (narrowly).**
Simple moves the same way. Per song (Detailed root / majmin_inv / sevenths_inv, points): 回家 +2.4 / +2.2 / +2.2,
10,000 Reasons +0.4 / +0.1 / +0.2, 立志擺上 +0.1 / +0.1 / 0.0, 活出愛 +0.4 / −0.1 / −0.1, **Here I Bow −1.2 /
−0.9 / −1.8**.

ACE alone on D1 (chord frames): root 0.831 → **0.847**, bass 0.818 → **0.837**, better on **all 5 songs** (root:
立志擺上 +1.1, 10,000 Reasons +0.6, 回家 +4.4, Here I Bow +1.4, 活出愛 +0.4); chord tones unchanged (0.555).

mini-D2 (Detailed): majmin_inv 0.925 → 0.917, sevenths_inv 0.808 → 0.804, root 0.924 → 0.917, short F1
0.623 → 0.618, bass acc 0.981 → 0.983, boundary F 0.877 → 0.865. ACE alone on mini-D2 shows why: on chord
frames root 0.875 → 0.949 and bass 0.895 → 0.976, but "no chord" frames (the silent lead-in and the reverb
tail) are recognised as N only 55 % of the time instead of 84 % — the heads-only model learned to say N less
often (our pool has few N frames).

## 4. Decision

- **ft1-a: FAIL** (majmin_inv −0.008, sevenths_inv −0.018 on D1). Not used.
- **ft1-b: the D1 gate PASSES, narrowly** — every gate metric is ≥ 0 at Detailed and at Simple, and ACE's own
  root / bass on real songs improve on all 5 songs. **But I did not switch the app's default**, because:
  1. the engine-level gains (+0.3 pt majmin_inv, +0.1 pt sevenths_inv) are far inside the noise of 5 DRAFT
     answer keys (28 short chords), mostly one song (回家), and Here I Bow gets worse;
  2. mini-D2 drops by about the same amount (−0.8 pt majmin_inv), from weaker N detection at silences;
  3. ft1-b's design was informed by ft1-a's D1 diagnosis (D1 used twice — the allowed maximum), so its pass is
     slightly optimistic;
  4. switching needs changes outside FT1's files: `pipeline.load_state` adds every `ace@*.v1.npz` it finds, so it
     must learn to pick the evidence of the active weights (else re-analysed songs carry both),
     `engine/tests/test_ace.py::test_checkpoint_loads_with_weights_only` asserts that `ace.load()` returns the
     original checkpoint, DecoderParams `temp["ace"]` must become 0.6267, and every cached song's ACE evidence
     must be re-run.
- **How to switch on** (opt-in, today): `ACE_WEIGHTS=ft1-b` (or `Ace(weights="ft1-b")`) and
  `DecoderParams(temp={..., "ace": 0.6267})` (`ace.FT1_RUN`, `ace.FT1_TEMP`). To make it the default later:
  `ace.DEFAULT_WEIGHTS = "ft1-b"`, `DecoderParams.temp["ace"] = 0.6267`, plus the pipeline / test changes in (4).
  Recommended only after it holds up on more real songs (Nathan-reviewed D1 keys or new songs) — ideally
  together with the next pool's model (§5).

## 5. Remaining weaknesses / next steps

1. **Chord-quality mix**: render the next pool with the real charts' statistics (≈ 80 % plain triads, 12 %
   slash chords, colours as in the charts) plus a smaller "rare chords" share, and weight the chord-tone loss by
   how often a quality occurs in real songs. ft1-a's notes head learned "worship songs are colourful".
2. **Real-band realism**: the band layers are simple synths (no guitars, no real drums, no real voices; the
   voice sings mostly chord tones). Real vocals and guitars add non-chord tones the model must ignore.
3. **No real training audio** (by design: YouTube only for testing). A few real multitrack worship recordings
   with the user's own chords would be the best guard against forgetting.
4. **D1 is small and DRAFT** (5 songs, 28 short reference chords): differences under ~1 point are within noise.
   It was used twice here (ft1-a, then ft1-b); do not tune further on it.
5. Short-chord precision at Detailed stays low (0.21) for both models — that is the decoder's passing layer
   (e.g. 活出愛's arpeggiated 12/8 bass), not ACE.
6. **N ("no chord")**: our pool has few N frames (lead-ins, endings); ft1-b says N less often, which costs a little
   at silences (mini-D2) but might help where the engine wrongly hears N inside quiet music (D1 problem 3).
   Next pool: more silent / near-silent stretches and quiet breaks, labelled consistently.
7. **BatchNorm**: fine-tuning ACE must keep its BatchNorm statistics (now the default); ft1-a re-estimated them
   on synthetic audio, which likely contributed to its real-song drop.

## Files

- `train/song_mix.py` (pool builder), `train/finetune_ace.py` (training, calibration, gate), tests in
  `train/tests/test_song_mix.py`, `train/tests/test_finetune_ace.py`.
- Pool: `data/train/pool/{train,val}/shard-*.npz`, stats `data/train/pool/pool.json`, build logs there.
- Runs: `models/ace_ft/ft1-a/`, `models/ace_ft/ft1-b/` (best.pt, last.pt, config.json, log.jsonl, val_best.json,
  calibration.json, gate.json); original on the val split: `models/ace_ft/baseline_val.json`;
  `models/MANIFEST.json` → `"ace_ft"`.
- Adapter: `engine/nathan_chordify/models/ace.py` — `ACE_WEIGHTS=<path to best.pt>` (or `Ace(weights=...)`) uses
  fine-tuned weights; their evidence is cached as `evidence/ace@<input>.v1-ft<sha8>.npz`, never mixed with the
  original's `ace@<input>.v1.npz`. Default: the original checkpoint (`DEFAULT_WEIGHTS = None`); DecoderParams
  unchanged (T_ace 0.5801). To try ft weights in the app, also pass `temp={"ace": <calibration.json T>}`.
  Note for whoever makes a fine-tuned model the default later: `pipeline.load_state` adds every
  `ace@*.v{Ace.adapter_version}.npz` it finds, so it must then be taught the weights tag too.
