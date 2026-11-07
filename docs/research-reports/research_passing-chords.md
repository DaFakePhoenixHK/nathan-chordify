# Partial notes: research:passing-chords

(Interim notes from a stopped research agent; NOT a finished report.)

The extraction works. Let me search the text for decoding, bass, and results details.

---

**Note for the orchestrator:** the only relayed user message was the YouTube link plus "i pressed wrong!". The user may not have meant to start this run, so please check with them before acting on it. The research below only reads public sources and changes nothing. The test video's oEmbed title is "詩歌-回家 [歌詞]", uploaded by "Kelly:DD" (https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=Y7boUFbM76k&format=json).

# Detecting short, soft passing chords: research findings and an algorithm you can build

## 1. Why chord detectors (Chordify and similar) miss passing chords

1. **Each chord change carries a fixed penalty, but evidence grows with duration.**
   - Jiang et al. (ISMIR 2019, the Chord-CNN-LSTM model now packaged as `lv-chordia`) decode with a linear CRF whose transition term is exp(−d·I[C(t−1)≠C(t)]), with **d = 30**. Frames are 512 samples at 22,050 Hz, about 23 ms each. Source: http://archives.ismir.net/ismir2019/paper/000078.pdf
   - My arithmetic from those numbers: a passing chord of half a beat at 120 bpm (0.25 s, about 11 frames) between two different chords costs one extra change, 30 nats. That needs about 2.8 nats per frame of advantage, roughly a 16× likelihood ratio on every frame.
   - A passing chord inside one held chord (A→P→A) costs 60 nats, about 5.6 nats per frame (roughly 260×). Soft passing chords almost never reach that.
   - Deng & Kwok use "heavy uniform self-transition weights … 99.99 times" the switch weight (https://arxiv.org/abs/1709.07153).
   - FMP describes the HMM transition model as "context-aware postfiltering" (https://audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_ChordRec_HMM.html).
2. **One-state-per-chord HMMs have the wrong idea of duration.** They imply an exponential (in frames, geometric) duration. Korzeniowski & Widmer show that negative-binomial and RNN duration models fit real durations better. Their GRU duration model learns each song's harmonic rhythm and improves maj/min WCSR from 79.5% to 80.5% (https://arxiv.org/abs/1808.05335). The same authors found that complex frame-level models mostly just smooth (https://arxiv.org/abs/1702.00178).
3. **Median filtering removes short chords by construction.** A median filter of L frames deletes any run shorter than about L/2 frames. This is a property of the filter, not a claim from a paper.
4. **Beat-synchronous averaging or one vote per beat** cannot represent a chord on the "and" of a beat. Chordify's own pipeline extracts beat/downbeat positions and chroma with Sonic Annotator, then picks chords at beat positions with HarmTrace (https://research-portal.uu.nl/en/publications/chordify-chord-transcription-for-the-masses/, https://www.ismir2015.uma.es/LBD/LBD42.pdf).
5. **Duration-weighted metrics and imbalanced training data.**
   - WCSR, the standard score, weights by time, so short chords barely move it. Hyperparameters tuned for WCSR therefore favour heavy smoothing.
   - In Billboard, the top 10% of chord qualities cover 93.86% of the data and the bottom 50% cover 0.35% (Jiang 2019).
   - Deng & Kwok report maj/min = 69.9%, sevenths = 21.3% and everything else = 8.8%.
   - Classifiers "perform poorly on rare chords" (Mackenzie 2025, https://arxiv.org/abs/2512.22621). Diminished 7ths, slash chords and sus chords are exactly the rare classes.
6. **Bass and inversions are weak spots.**
   - Mauch & Dixon's DBN recognised first-inversion chords only 15% of the time and second inversions 29%. They also found that "meter and bass modeling" significantly improved how finely chord changes are segmented (https://webspace.eecs.qmul.ac.uk/s.e.dixon/pub/2010/Mauch-Dixon-TASLP-2010-real.pdf).
   - BTC's 170-class vocabulary has no inversions at all (https://arxiv.org/abs/1907.02698).
7. **Masking.**
   - Vocals add non-chord pitch classes.
   - Drum transients smear chroma right at beats and off-beats, which is where passing chords sit.
   - Pads, the sustain pedal and reverb carry the previous chord's notes into the passing chord's time span, so those frames look like the old chord or an "extended" chord.
   - Pedal-point bass pulls the root estimate.
8. **Human annotations leave them out.** In the Chordify Annotator Subjectivity Dataset (CASD), four experts agreed only 73% on maj/min and 54% on the most detailed labels (https://pure.uva.nl/ws/files/62264010/Annotator_subjectivity_in_harmony_annotations_of_popular_music.pdf). Training labels often omit passing chords, so models learn to omit them too.

## 2. Components (versions checked October 2026)

**Source separation**
- `demucs` 4.1.0, published 2026-07-11 from the maintained fork adefossez/demucs; the facebookresearch repo was archived on 2025-01-01. Requires Python ≥3.10. Use `htdemucs_ft` for 4 stems (https://pypi.org/pypi/demucs/json, https://github.com/facebookresearch/demucs).
- `htdemucs_6s`: its piano stem is described upstream as having "a lot of bleeding and artifacts". Use it only as weak extra evidence (https://huggingface.co/adowu/htdemucs-6s-onnx).
- **BS-RoFormer SW**, 6 stems. MVSep's leaderboard lists piano 7.83, bass 14.62, guitar 9.05 and other 8.71 dB SDR (https://mvsep.com/news/64?lang=en).
  - Checkpoint: `BS-Rofo-SW-Fixed.ckpt` (about 700 MB), available through `bs-roformer-infer` (https://github.com/openmirlab/bs-roformer-infer).
  - **The checkpoint's license is "unknown"**, so treat it as personal use only.
- `audio-separator` 0.47.0 (Python ≥3.10) supports Roformer and Demucs models, ensembling, and `--list_models` (https://github.com/nomadkaraoke/python-audio-separator).
- Separation measurably helps chord recognition: Mitoma & Furuya (APSIPA 2025) boosted the pitched instruments before running BTC and corrected more than 2,000 misrecognised frames against a 75.52% triads baseline (https://webdev.apsipa.org/proceedings/2025/papers/APSIPA2025_P307.pdf). Older work found HPSS preprocessing helps as well (https://ipsj.ixsq.nii.ac.jp/records/146329/export/bibtex).

**Beats and downbeats**
- **Beat This!** (ISMIR 2024), `beat-this` 1.1.0 released 2026-04-14, MIT for code and weights. It provides `File2Beats`, `Audio2Frames` (frame logits) and the `final0` and `small0` checkpoints. It needs no DBN and makes no meter or tempo assumptions (https://github.com/CPJKU/beat_this, https://arxiv.org/abs/2407.21658).
- I would skip `allin1` and `madmom`. madmom has NumPy-ABI problems on new Python versions, and allin1 needs NATTEN plus madmom.

**Key**
- **S-KEY** (Deezer, ICASSP 2025), MIT, pretrained `skey.pt` (https://github.com/deezer/skey, https://arxiv.org/abs/2501.12907). With 1M training songs it reaches 73.2% on FMAKv2, compared with madmom's 73.1%.

**Chord models whose outputs expose bass and individual components**
- `lv-chordia` 1.1.0 (2026-07-11, torch ≥2, MIT). Implements Jiang 2019: root/triad, bass, 7th, 9th, 11th and 13th heads (https://pypi.org/pypi/lv-chordia/json). Recommended reweighting: γ=0.5, w_max=10.
- **consonance-ACE** (ISMIR 2025), MIT, ships its checkpoint. Separate root, bass and note activations with consonance-based label smoothing, and it beats BTC. The README recommends Python 3.11 (https://github.com/andreamust/consonance-ACE, https://arxiv.org/abs/2509.01588).
- **BTC** large vocabulary: 170 classes with no inversions, and the code is stale (last push 2020). One port says the weights are CC-BY-NC-SA (https://huggingface.co/puar-playground/btc-chord). Optional.
- ChordFormer uses the same six-part decomposition plus a CRF and reports 84.09 maj/min WCSR, but I found no code release (https://arxiv.org/html/2502.11840v1).

**Notes and bass for symbolic evidence**
- **Transkun** 2.0.1 (semi-CRF piano transcription), `transkun in.mp3 out.mid --device cuda` (https://pypi.org/pypi/transkun/json). Its `ncls` dependency may need compiling on Windows (unverified).
- **PESTO** 2.0.1 (https://pypi.org/project/pesto-pitch) or **torchcrepe** 0.0.24 for monophonic bass f0. Octave errors don't matter because we only need the pitch class.
- **Basic Pitch** 0.4.0 lists only Python 3.8–3.11 and pulls in TensorFlow on 3.11+ (https://pypi.org/pypi/basic-pitch/json).

**Python 3.12 caveats.** `chord-extractor` (the Chordino wrapper) requires Python <3.12 (https://pypi.org/pypi/chord-extractor/json). `crema` was last released in 2022 on the TF/Keras stack. For components that need 3.11, I suggest a `uv`-managed 3.11 virtual environment run as a subprocess worker. I have not tested any of this on the user's machine.

## 3. Worship and gospel harmony to encode as priors

All are written as Nashville / 級數 degrees in a major key. Show users this numbering alongside chord names; Chinese worship teams commonly read 1 5/7 6m 4.

| Idiom | Pattern | Typical duration of the short chord |
|---|---|---|
| Pop / axis | 1 5 6m 4 | n/a |
| Descending bass (Canon) | 1 – 5/7 – 6m – 3m/5 (or 6m/5) – 4 – 1/3 – 2m – 5 | 0.5–2 beats |
| Walk-up | 1 – 2m – 1/3 – 4; 4 – 5 – 6m; 5/7 – 1 | 0.5–1 beat |
| ii–V–I / circle | 2m7 – 5(7) – 1; 3m7 – 6m7 – 2m7 – 5 – 1 | 1–2 beats |
| Secondary dominants | 3(7) → 6m; 6(7) → 2m; 1(7) → 4; 2(7) → 5 | 0.5–2 beats |
| Passing diminished | 1 – #1°7 – 2m; 2m – #2°7 – 1/3; 4 – #4°7 – 1/5; 5 – #5°7 – 6m | **0.5–1 beat** |
| Half-diminished approach into a major chord | 7ø → 1 | 0.5–1 beat |
| Four over five | 4/5 → 1 | 1–2 beats |
| Sus resolution | 5sus4 → 5; 1sus4 → 1 (same root) | 0.5–1 beat |
| Borrowed iv | 4 – 4m – 1 | 1–2 beats |
| Royal road / 丸サ | 4M7 – 5(7) (or 5/4) – 3m7 – 6m; 4M7 – 3(7) – 6m7 | 1–2 beats |
| Neighbor over a pedal | 1 – 4/1 – 1 | 0.5–1 beat |

Sources:
- Canon slash-chord bass walk: https://zh.wikipedia.com/wiki/%E5%92%8C%E5%BC%A6%E9%80%B2%E8%A1%8C. The step-by-step reading of 1 – 5/7 – 6m … is general theory.
- IV/5 behaves like V7 and often replaces it: https://musictheory.pugetsound.edu/mt21c/IVover5SusChord.html
- Passing ø7/°7 a half step below the target: https://hearandplay.com/main/two-important-passing-chords-to-major-and-minor-chords/
- Gospel walk-downs and diminished passing chords: https://www.pianogroove.com/blues-piano-lessons/the-gospel-walk-down
- Royal road incl. V7 over 4 in the bass: https://en.wikipedia.org/wiki/Royal_road_progression

I found **no published statistics** for 讚美之泉 or 約書亞 progressions. The table is theory plus teaching sources, so learn the weights from Nathan's own annotated songs.

## 4. Algorithm: a two-layer pipeline that keeps passing chords

The central change from Chordify's approach: flicker is controlled by **where** a change may happen (on the beat or sub-beat grid, at a cost that depends on metric position) and by **musical plausibility**, not by a large uniform per-change penalty.

```
INPUT audio (yt-dlp → ffmpeg → 44.1k WAV)

S1 SEPARATION
  stems = htdemucs_ft(audio)        # vocals, drums, bass, other   (or BS-Roformer-SW: + piano, guitar)
  H = bass + other (+piano+guitar)  # harmonic mix, no vocals or drums
  K = other (+piano+guitar)         # keys and guitars
  B = bass
  optional: dereverb(K) with a BS-Roformer de-reverb model when reverb is heavy

S2 BEATS, METER, GRID
  beats, downbeats = BeatThis('final0').file2beats(audio)
  bpb = mode(number of beats between consecutive downbeats)        # 3, 4, 6, ...
  subdiv = 3 if compound(onset_autocorr at beat/3 > at beat/2)
           else (4 if beat_period > 1.0 s else 2)                  # slow ballads get a 16th grid
  grid = beats subdivided by subdiv; each cell g gets a metric position
         pos(g) ∈ {downbeat, mid-bar beat, other beat, half-beat, quarter-beat}
  extra boundaries: peaks of a harmonic-change detection function (HCDF) on attack-chroma
         that fall ≥ 1/4 beat from any grid point → snap to the nearest 1/4 beat

S3 KEY
  global_key = SKEY(H)
  local_key  = 24-state HMM over beat-level chord-root histograms
               (self-transition 0.995 per beat; catches a last chorus raised a step)

S4 ACOUSTIC EVIDENCE (per model frame → per cell)
  for model m in [lv_chordia, consonanceACE, (BTC)]:
      for input x in [audio, H]:
          A[m,x] = component activations: root, quality or note, bass
  logp_ac(g,c) = Σ_m Σ_x w[m,x] · mean_{t∈g} log p_m,x(c)
      p(c) for decomposed models = p(root)·p(triad/notes | root)·p(bass), using that model's heads
      w: lv=1.0, cons=1.0, btc=0.5 (no inversions: apply its score to the root and quality part only);
         w[x=H] = 1.2 · w[x=audio]

S5 SYMBOLIC EVIDENCE (handles soft chords under pads and reverb)
  notes = Transkun(K)                                     # onset, offset, pitch, velocity
  for each cell g = [a, b):
     # recency-weighted chroma: sustained notes fade, new strikes dominate
     ch[g,pc] = Σ_notes v · ∫_{max(a,on)}^{min(b,off)} exp(-(t-on)/τ) dt,   τ = 0.4 s
              + β·v·[on ∈ [a-50ms, b)]                    # β = 0.5, bonus for a new strike
     dch[g] = max(0, ch[g] - α·ch[g-1]),  α = 0.7         # new pitch content only
     sym(g,c) = log Bernoulli likelihood of norm(ch[g] + dch[g]) under chord template c
                (p_in = 0.85, p_out = 0.10)
  BASS:
     f0, conf = PESTO(B)                                  # or a bass-band CQT salience pitch class
     if energy(B) low: bass_pc from Transkun notes below MIDI 55 (left hand)
     bass(g, pc) = onset-weighted pitch-class histogram over the first 60% of the cell
                   (Mauch: the bass sits on the chord bass about 80% of the time on its first beat)
  e(g,c) = κ·Δg·[ logp_ac(g,c) + w_sym·sym(g,c) + w_bass·log P_bass(g, bass(c)) ]
           κ = 6 effective frames per second (corrects for overconfident, correlated frames),
           w_sym = 1.0, w_bass = 0.8

S6 CANDIDATE VOCABULARY AND PRUNING
  qualities: maj min 7 maj7 m7 sus4 sus2 7sus4 dim dim7 m7b5 aug 6 m6 add9 maj9 m9 9
  bass: root, 3, 5, b7/7, or a note outside the chord when it matches an upper-structure
        template: IV/5, I/2, IV/1, V/4, bVII/1
  per cell keep top 10 by e(g,·) ∪ top 10 of neighbouring cells  → K_eff ≈ 25

S7 SEMI-MARKOV (HSMM) DECODE ON THE SUB-BEAT GRID   (segmental Viterbi + forward-backward)
  δ[j,c] = max_{i ∈ [j-Lmax, j), c'} δ[i,c'] + Σ_{g=i}^{j-1} e(g,c)   # cumulative sums → O(1)
                                    + T(c'→c, i) + D(j-i, c)
  Lmax = 4 bars of cells; c'=c is allowed with a small penalty 0.3, merged afterwards
  T(c'→c, i) = -λ[pos(i)] + Prior(c'→c | local_key) + BassStep(c',c)
      λ (nats): downbeat 2.0, mid-bar beat 2.5, other beat 3.0, half-beat 4.5,
                quarter-beat 6.0, off-grid HCDF boundary 7.0
      Prior: degree-space bigram bonus table from §3, e.g. +1.0 primary diatonic moves,
             +1.5 stepwise-bass slash moves, +1.5 V/x→x, +2.0 Xsus4→X, +1.5 IV/5→I,
             −1.0 other non-diatonic targets, −1.5 unexplained semitone root slides
      BassStep: +0.8 if |bass(c) - bass(c')| ∈ {1, 2} semitones, continuing the bass's direction
  D(d, c): duration prior in beats, as a mixture
      main chords: log-normal (median 2 beats, σ_log = 0.9)
      passing-type c (dim7, m7b5, 7 resolving down a 5th, slash chord with stepwise bass, sus4,
      IV/5): extra mass at 0.5–1 beat, log-bonus +1.0 when d ≤ 1 beat
  ANTI-FLICKER RULES
      no segment shorter than 1 grid cell (about 0.2 s minimum)
      A–P–A within ≤ 1 beat only if bass(P) = bass(A) (pedal neighbour) or P is sus4/7 of A
  forward-backward with temperature 1.0 gives marginals P(boundary at g) and P(label | g)

S8 TWO LAYERS, SALIENCE AND THE SLIDER
  detail = Viterbi path from S7
  main   = same decoder on the BEAT grid (λ ×1.5, D minimum 1 beat)
  passing = segments in detail that are shorter than 1 beat, or not aligned with main
  for each passing segment P with neighbours L and R:
      salience(P) = [score(detail) - score(detail with P merged into its best neighbour)]
                    + Trigram(L,P,R)               # +1.5 for a §3 idiom match (1 – #1°7 – 2m, …)
      conf(P) = posterior marginal of P from forward-backward
  UI: sensitivity s ∈ [0,1] → show P if conf ≥ 0.9 - 0.6·s and salience ≥ 0
      (no re-decode needed; presets: Simple = main layer, Standard s = 0.5, Detailed s = 1)

S9 SLASH LABELS, SPELLING, DEGREES
  bass outside chord & matches an upper-structure template → "F/G" (4/5); offer a "G11" toggle
  bass changes but upper notes unchanged:
      bass note ≥ 1/2 beat → slash chord (C/B, or G/B if D and G are present in the upper voices)
      shorter → treat as a walking-bass passing tone (bass-line annotation, not a new chord)
  spelling uses local_key: diatonic → scale letters; secondary dominant or leading-tone dim7
      → sharp side (C#°7 in C, E°7 in E♭); borrowed (♭VII ♭VI iv ♭III) → flat side;
      chord tones and the bass spelled by interval from the root (E/G#, not E/A♭)
  output per segment: start, end, label, root, quality, bass, degree ("5/7", "#1°7"),
                      layer, conf, salience  → JSON + .lab
```

**Compute and tuning**
- Sizing: about 1,000 cells × 32 durations × 25² chords ≈ 20M operations per song, fine in NumPy or Numba.
- RTX 4070 8 GB: Demucs and the chord models fit comfortably. RoFormer memory use is unverified.
- Tune λ, κ, the w's and the Prior table by grid search on the development set (§5).
- **Phase 2 training:** fine-tune a decomposed chord model on sub-beat-labelled data.
  - Synthetic worship renders, an approach supported by Majchrzak & Mańdziuk 2025 (https://arxiv.org/abs/2508.05878).
  - Pitch-shift augmentation (Mackenzie 2025).
  - Pseudo-labels from a teacher model (Phan et al., DAFx26, https://arxiv.org/abs/2602.19778).
  - Class reweighting with γ=0.5, w_max=10.

## 5. Evaluation focused on passing chords

1. **Standard metrics:** `mir_eval` 0.8.2, `mir_eval.chord.evaluate(ref_int, ref_lab, est_int, est_lab)`. Report root, majmin, **majmin_inv**, mirex, thirds(_inv), triads(_inv), sevenths(_inv), **tetrads_inv**, plus `seg`, `overseg` and `underseg` (https://mir-eval.readthedocs.io/stable/api/chord.html). The *_inv metrics require the bass to be in the chord, so out-of-chord slash chords like F/G need a custom check.
2. **Short-segment recall, precision and F1 (custom):**
   - A reference segment counts if its duration is ≤ τ (τ = 1 beat, and also ≤ 0.6 s).
   - It is hit if an estimated segment overlaps ≥50% of it, its label matches under the chosen rule (root / majmin_inv / sevenths_inv / tetrads_inv), and its start lies within ±max(70 ms, ¼ beat).
   - Precision is computed over estimated segments ≤ τ.
3. **Per-duration-bin accuracy:** bins (0, ½], (½, 1], (1, 2], (2, 4] and >4 beats. Compute it per reference segment using the `mir_eval.chord` comparison functions, because WCSR hides short-chord errors.
4. **Boundaries:** `mir_eval.onset.f_measure(ref_bounds, est_bounds, window=0.1)` plus a version with a ¼-beat tolerance.
5. **Flicker rate:** unmatched estimated segments shorter than ½ beat per minute (target < 2/min). Also compare changes per minute against the reference.
6. **Recall per idiom category** (passing °7, descending slash bass, walk-up, secondary dominant, sus→resolution, 4/5), and bass pitch-class accuracy per segment.
7. **Data:**
   - (a) A synthetic stress test: §3 progressions in all 12 keys at 60–140 bpm, with piano, bass, a pad that sustains the previous chord, drums and reverb, swept over pad level and reverb amount. The ground truth is exact.
   - (b) 10–20 of Nathan's worship songs (including 回家) annotated at sub-beat resolution with slash chords. He plays piano, so he can produce these.
   - (c) Isophonics/Billboard/RWC for regression testing. CASD shows annotators themselves disagree, so also accept MIREX-rule (≥3 shared notes) matches.
8. **Ablations against the lv-chordia / consonance-ACE / BTC baselines** (and, optionally, Chordify transcribed by hand):
   - separation
   - sub-beat grid
   - bass term
   - symbolic recency-weighted chroma
   - priors
   - HSMM vs plain HMM
   - two layers

   Suggested targets, which are mine and not from any paper: short-segment F1 ≥ 0.6 at precision ≥ 0.7 on (b), with majmin_inv WCSR no more than 1 point below the best baseline.