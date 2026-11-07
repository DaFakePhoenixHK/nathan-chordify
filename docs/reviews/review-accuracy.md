# Adversarial review of DESIGN.md v1.0: accuracy, completeness and scope

Reviewed 2026-10-13. This is the raw review; the fixes are folded into DESIGN.md v1.1.

**Verdict:** the architecture fits the passing-chord goal: separation, a sub-beat HSMM, a dedicated bass tracker, two layers and the Detail slider. The numeric core and the bass tracker need fixing before any UI work:
- As specified, the numeric core will either flicker or drop passing chords, and the slider won't behave as described.
- The bass tracker can't resolve sub-beat bass notes.

## A) Issues

### A1 [BLOCKER] Evidence scale and change costs are not calibrated to each other (§5.9–5.11)

**What's wrong**
- E is a weighted *sum* of 7 sources (Σw = 5.9), not an average.
- Each source is itself a sum of several log-probabilities: lv has 6 head terms; ace has 12 unfloored Bernoulli terms.
- The same network on the mix and on the harmonic stem is counted twice, because κ only corrects for correlation over time.

**Numbers at 65 BPM**
- One 16th cell: κΔg = 1.38.
- A mild 0.6-nat advantage gives about 4.9 nats per cell, so about 9.8 nats for a half-beat chord.
- On clear frames the gaps are 20–60 nats per cell.
- Against that, λ is 2–6, the priors ±1–2 and the PT bonus +1. Decoding collapses to per-cell argmax.

**Knock-on effects**
- Posteriors saturate, so conf ≈ 1 and min_detail ≈ 0.05.
- The salience ≥ 3 exemption is always met.
- "Simple" isn't simple.

**The naive fix fails the other way.** With a true weighted mean and κ = 6, break-even needs about 1.1–1.7 nats for a half-beat A–P–B chord, about 2.6–2.9 nats for A–P–A, and about 4.3 nats for a single 16th cell. Soft passing chords would be lost.

**Fix**
1. Merge @mix and @harm per model into one source.
2. Per source, take a log-softmax over C, apply a temperature T_s fitted by NLL, and clip at −6.
3. Make the bracket Σw·L/Σw.
4. Measure evidence in beats: E = κ_b·(Δg/ibi)·bracket, with κ_b ≈ 10/beat as a start. Aim for break-even ≈ 0.7 nats (half-beat A–P–B) and ≈ 1.3 nats (A–P–A).
5. Fix κ_b and tune λ-scale and the priors relative to it.
6. WP7 acceptance: a break-even table, plus a reliability diagram with ECE ≤ 0.1. Gate the slider on it.

### A2 [BLOCKER] Bass CQT at 36 bins/octave (Q ≈ 51) has filters that are too long (§5.6)
- Filter lengths: 1.25 s at 41 Hz, 0.62 s at 82 Hz.
- The "first 60% of a 0.23 s cell" rule therefore reads the previous note.

**Fix**
- Use 12 bins/octave with filter_scale 0.5 (Q ≈ 8.4: about 0.2 s at 41 Hz), or librosa.pyin (fmin 40, frame 2048 @ 22.05 kHz, hop 256).
- Build P_bass from onset-segmented bass notes, weighted by overlap with the cell, and compensate for latency.
- Acceptance: 16th notes at 65 BPM and 8th notes at 130 BPM, E1–G3, with ≥90% pitch-class accuracy per note and onset error ≤ 40 ms.

### A3 [MAJOR] Bass evidence is double-counted, and the dedicated tracker carries only about 11% of the influence
**Fix**
- P_bass comes only from our own tracker plus the AMT left hand.
- Scale the model bass heads by about 0.5.
- Tune w_bas in [1, 3].

### A4 [MAJOR] Symbolic chroma works against slash chords and "RH holds, bass moves" chords (§5.7, §5.9)
**Fix**
- Give the bass pc τ = 0.5 in sym unless it is also an upper tone.
- Split notes at MIDI 55: below goes to the bass term, above to sym.
- Measure presence with saturation, p = 1 − exp(−ch/θ), using pedal-extended offsets.
- Use dch only as an onset feature. Don't max-normalise to the newest strike.

### A5 [MAJOR] Equivalent labels split the posterior
- Examples: dim7 (4 labels), aug (3 rotations), E6 = C#m7/E, F#9sus4 = E/F# = C#m7/F#, Csus2 = Gsus4/C.
- A passing °7 ends up at conf ≤ 0.25 and is dropped.

**Fix:** compute conf, salience and pruning over the equivalence class (pc-set, bass). Pick the display label inside the class afterwards, using the key/idiom prior and the model root head.

### A6 [MAJOR] Transition-blend phantom chords are not filtered
- At a boundary, the old chord's reverb mixes with the new attack (e.g. B→Emaj7 reads as Emaj7/B or Bsus4).

**Fix:** merge P into the neighbour it overlaps most when all of these hold:
- P ≤ ½ beat;
- pcs(P) ⊆ pcs(A) ∪ pcs(B);
- bass(P) ∈ {bass(A), bass(B)};
- there is no new piano onset with a pc outside A ∪ B.

Checked against 回家: B+7, A7, E/D#, B/D# and F#/A# all survive this rule.

### A7 [MAJOR] The A–P–A rule deletes genuine neighbour chords
- Deleted examples: I – V/7 – I, plagal I – IV – I, and I – vii°/I.
- "Everything" can't bring them back.

**Fix**
- Never delete; demote instead (min_detail += 0.3).
- Whitelist a bass within ±1–2 semitones, IV-of-A and V-of-A.
- Keep sus/add9 neighbours only if harm or sym evidence supports the added tone, so the vocal melody can't create them.

### A8 [MAJOR] The < ½-beat bass-passing-tone merge removes fast walk-ups (½ beat is 0.25 s at 120 BPM)
- Bass approach notes can also be relabelled as fake chords (e.g. "C°" over a held B).

**Fix**
- Threshold = min(½ beat, 0.3 s).
- At s = 1.0, draw these events in the grid as slash chords.
- If P's non-bass pcs ⊆ a neighbour's pcs, P is a bass event: it keeps the neighbour's label with a slash and is never relabelled.
- Make §5.11 and §6.2 agree on when bass tones show.

### A9 [MAJOR] Vocabulary gaps
- **Add:**
  - 13sus4 (4maj7/5 = Emaj7/F#)
  - m11
  - 5 (no 3rd)
  - maj7#11
  - 7b13
  - +maj7
- **Extend** the +2/+5/+10 bass intervals to maj7, m7, 6 and add9 (C#m7/F#, Emaj7/F#, B6/C#).
- **Display:** when the 7th sounds only in the bass, show triad/bass (C/B, Am/G#).

### A10 [MAJOR] Priors
- In 回家, A7 → Em6 (♭VII7 → iv) matches no rule and gets −1.0. Fix: borrowed-iv applies to cp ∈ IV-family ∪ ♭VII-family.
- **Add these idioms:**
  - chromatic descending m7 (with a same-direction trigram);
  - tritone substitution ♭II7 → I;
  - ♭VI → ♭VII → I;
  - a descending chromatic bass under the tonic;
  - line clichés.
- Specify whether rules stack, and set w_dur.
- Document the hidden change cost of the log-normal duration prior (about 1–2.5 nats).

### A11 [MAJOR] Pruning
- The top-10 per cell fills with variants of the sounding chord, so once-per-song passing chords (B+7, Em6) get pruned.

**Fix**
- Rank by equivalence class.
- Add idiom completion: insert the bridge candidates (#x°7, V/x, bass-step slash, sus4, aug) for every adjacent pair of main chords.
- Cap at 250.

### A12 [MAJOR] Key and modulation
- The emission is raw cosine similarity: a 0.05 gap is about 0.06 nats per beat, against a switch cost of about 8.6 nats. A last chorus raised a step is usually missed.

**Fix**
- Emission = log-softmax(cos/T) with T ≈ 0.02.
- Treat same-mode +1/+2 shifts as favoured.
- Run a chord-based local key on 16-beat windows, plus an explicit truck-driver test, then re-decode.
- Default spelling = the fewest accidentals.
- Add key-change markers to the JSON/UI and restart Nashville numbering at them.

### A13 [MAJOR] Beat grid
- The triplet test (subdiv 3) was dropped.
- There's no free-time or rubato fallback.
- The octave rule only fires for bpb == 8.

**Fix**
- Run the triplet test per section; allow subdiv ∈ {2, 3, 4, 6}.
- Where beat_conf < 0.5 or the IBI coefficient of variation > 0.15, use a free-time grid: 0.2 s cells snapped to onsets, flat λ = 4, unmetered display.
- Octave check: if ≥80% of main changes fall on even beats, halve the tempo.

### A14 [MAJOR] Time resolution
- The networks' internal smoothing remains even without the CRF.
- ACE and BTC frames are about 93 ms.
- The pooling rule contradicts itself.

**Fix**
- Interpolate log-probs onto an 11.6 ms timeline and integrate over each cell.
- Measure each adapter's time offset (|offset| < 20 ms).
- Add w_short, which gives the fast sources (bass, sym) more weight in cells ≤ ½ beat.

### A15 [MAJOR] The slider uses only a single label's posterior
**Fix**
- min_detail = σ(a·logit(conf_class) + b·salience + c), fitted so that Standard, Detailed and Everything give passing-chord precision of about 0.85, 0.70 and 0.50.
- Add those per-stop precision constraints to the tuning objective.

### A16 [MAJOR] No cross-repetition consistency
- Worship choruses repeat 3–6×.

**Fix**
- Align repeats with self-similarity and add a soft prior (+1 nat) for agreeing with the consensus label. It must stay soft, because 回家's chorus 3 differs.
- This also produces the "auto" sections.

### A17 [MAJOR] Bug in the piano voicing rule
- B/D# gets the RH D#–F# (which sounds like D#m), and D#7/A# gets G–C#.

**Fix**
- For inversions, keep the root in the RH and drop the bass pc.
- For a non-chord bass, the RH plays the full upper chord.
- Rootless voicings only as an option.
- Check the sign of the top-note term.

### A18 [MINOR] Spelling
- Avoid B#, E#, Cb and Fb roots by default (#1°7 in B shows as C°7), with a toggle.
- Show double sharps as their enharmonic in note names.

### A19 [MINOR] Undefined layer logic
- Does same_chord's "extension" cover sus, 6 and add9?
- When R spans several main segments, which one is M?
- The promotion rule compares posteriors from two different lattices.

## Evaluation

### E1 [MAJOR] The draft GT is circular (aligned to our own beat grid) and bar-level only
**Fix**
- On draft GT, report only the checklist and coarse metrics.
- Nathan taps an independent `gt.beats`.
- Intake format: `| B . . B+7 | Emaj7 . A7 Em6 |`.

### E2 [MAJOR] The checklist is recall-only, and its times are inconsistent
**Fix**
- Write each item as [t0, t1, label, tolerance] and score each occurrence.
- Score at Detailed (0.8).
- Add a false-positive gate: ≤3 extra passing chords per minute in H-confidence bars.

### E3 [MAJOR] D1 (10–20 songs at sub-beat resolution, 20–60 h) is unrealistic, and correcting drafts anchors the annotator
**Fix**
- Stage it: 3 songs by M2, 8 by M3.
- Start annotation from the main layer only.
- GT mode needs a bar loop at 70%, synth A/B against the audio, and keyboard-first entry.
- D1 coverage:
  - ≥2 songs at 100–130 BPM
  - ≥1 song in 6/8 or 12/8
  - ≥1 song with a modulation
  - ≥1 song with a free-time intro
  - ≥1 Mandarin song
  - ≥1 solo-piano arrangement
- 回家 never exercises 16th cells.

### E4 [MAJOR] "Beats Chordify" can't be shown without B4
- Make B4 (Chordify's output) mandatory for 回家 plus ≥3 songs.
- Add B5: our main layer at Simple.

### E5 [MINOR] D2
- Use D2 (synthetic) for calibration and unit tests only.
- Weight D1 at least equally in J.
- Render it with a sampled piano (Salamander, CC-BY).

## Work packages

### W1 [MAJOR] Missing contract types
- Segment, Vocab/Candidate, Evidence, DecodeResult, SymParams, DecoderParams, the overrides schema and the edit-op schema.

### W2 [MAJOR] Hidden critical path
- M3 needs D1, D1 needs WP12's GT mode, and that needs a GT import/export route that isn't in §11.

### W3 [MAJOR] Work with no owner
- auto sections
- applying edits and re-spelling (`edits.py`)
- lyrics routes and the LRCLIB proxy
- ingest.search
- the chords-subset query

### W4 [MAJOR] PUT /edits replaces the whole list, so last writer wins
- Fix: append-only operations with IDs and a server-side merge.

### W5 [MINOR] Interface gaps
- hsmm.decode needs region and boundary-state parameters.
- WP6 should expose trigram().
- key.refine creates a WP6↔WP7 loop that WP8 must orchestrate.

### W6 [MAJOR, scope] Drop the Python + TypeScript dual theory implementation
- Instead, the server precomputes labels for 12 transpositions × levels × accidental preference (about 100–200 KB per song), or provides a /respell endpoint.

### W7 [MINOR] WP7's acceptance test must use real adapter outputs, not synthetic evidence

## B) Missing or mis-described features
1. iReal Pro export.
2. Chords & Lyrics parity: chords anchored to specific characters, text size, karaoke highlighting, pinnable diagrams.
3. Auto section labels (no algorithm, no owner).
4. Key changes in the UI (per-segment badge, Nashville renumbering, transpose behaviour across a modulation).
5. Edits flowing into MIDI, PDF and print; tempo in the MIDI filename.
6. Deezer is unsupported; document it.
7. v1 does not reach full parity: MIDI, PDF, setlists, offline and the editor are v1.1, and the Toolkit is v2. Say this plainly.
8. Our phone app needs the laptop awake, except for songs saved offline.
9. "Search by chords" has no route.
10. When bass tones show is inconsistent between §5.11 and §6.2.

## C) MVP proposal (ordered)
1. **Environment and spikes:**
   - venv, torch, ffmpeg, yt-dlp + deno, smoke test;
   - download the test URL;
   - GO/NO-GO on lv-chordia's raw per-head probabilities;
   - demucs + Beat This! on 回家 (bars ≈ 3.67 s);
   - ask Nathan for his chords.
2. **Engine v0 + CLI, one model:**
   - ingest → htdemucs_ft → Beat This! (triplet test + free-time fallback) → lv-chordia (mix and harm merged) → bass (12-bin CQT or pyin) → calibrated evidence → HSMM detail + main → layers (demotion, phantom rule, bass-event rule) → spelling/Nashville → JSON and .lab;
   - `nchord analyze URL --chart`.
3. **Evaluation harness:**
   - mir_eval, short-segment F1, flicker, checklist + false positives, baseline B1, mini-D2 (12 idioms × 3 keys × 2 tempos), calibration table;
   - tune λ-scale, w_bas, the PT bonus and the slider mapping;
   - gate: beat B1 on short-segment recall with flicker ≤ 2/min.
4. **Browser MVP:**
   - FastAPI, one GPU worker, SSE, audio route;
   - React sub-beat grid synced to `<audio>`;
   - tempo 25–200% (preservesPitch), A/B and drag loop, transpose (server-precomputed labels), Detail/Level/Slash/Nashville;
   - NOW/NEXT piano diagram, alternatives, re-scan.
5. **Minimal editor + GT mode:**
   - set chord, delete, accept/reject, undo;
   - append-only edit ops;
   - chart import/align, .lab export;
   - Nathan annotates 回家 + 2 songs.
6. **Accuracy round 2**, each step kept only if LOSO short-segment F1 rises ≥0.02 with no majmin_inv loss:
   - consonance-ACE
   - piano transcription → symbolic chroma
   - repetition prior
   - modulation detector
   - BTC-CL last
7. **Platforms:** pywebview window with .bat and shortcut; PWA over Tailscale.
8. **v1.1 parity:**
   - signalsmith-stretch (pitch shift, gapless loops, count-in, speed trainer), metronome, stem mixes;
   - exports: MIDI, MusicXML, ChordPro, iReal, print;
   - setlists, offline PWA, lyrics, beat-grid editing UI, library search, auto sections.
9. **v2:** Toolkit and the phase-2 models.

**Deferred:**
- two-phase fast preview
- mkcert/LAN pairing
- ytsearch and LRCLIB
- BS-RoFormer, BACHI, S-KEY, ASR
- the TypeScript theory port
- BTC-CL until an ablation shows it helps
