# Partial notes: research:sota-models

(Interim notes from a stopped research agent; NOT a finished report.)

# Chord-recognition models we can run locally (October 2026), with a ranked ensemble for worship/gospel on an 8 GB RTX 4070

## Summary

**Recommended ensemble, best first:**
1. **lv-chordia** (the Jiang et al. 2019 chord model, repackaged).
2. **consonance-ACE** (Poltronieri et al., ISMIR 2025).
3. **ChordMini's improved BTC checkpoint** (2026).
4. **A notes-first branch:** split the song into stems, transcribe the piano and bass stems into notes, run a symbolic chord recognizer (BACHI) on those notes, then apply music-theory rules.
5. **Chordino** with a custom chord list, used to break ties.
6. **Beat This!** for the beat grid.

All parts are combined at the level of root, bass and individual notes, then smoothed with a key-aware decoder.

**Expected runtime** for a 4-minute song: about 1.5–2.5 min for the full pipeline, or about 30–45 s in a fast mode without stem separation. Peak GPU memory should stay under about 5 GB if models are loaded one after another. These timings are my estimates. I did not benchmark anything, and nothing was installed on the machine.

**Facts I checked about the machine's software stack:**
- The latest PyTorch is 2.14.1 (30 Sep 2026). It has Windows wheels for Python 3.12 with CUDA builds cu126, cu130 and cu132.
- librosa 1.0.0 came out on 11 Aug 2026 and needs Python 3.12 or newer. Pin `librosa==0.11.*` for the older model code until it is tested on 1.0.
- `pip install basic-pitch` on Python 3.12 / Windows **fails**. I confirmed this with a dry run: it requires `tensorflow<2.15.1`, which has no Python 3.12 build. The workaround is `pip install --no-deps basic-pitch`, then install onnxruntime yourself.

---

## 1. Chord models with released weights (ready to run)

### 1. Jiang et al. 2019, "Large-vocabulary chord transcription via chord structure decomposition" — use the lv-chordia package
- **Paper:** ISMIR 2019, pp. 644–651. Repos: [music-x-lab original](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition) (MIT, last push April 2024) and [openmirlab/lv-chordia](https://github.com/openmirlab/lv-chordia) (MIT, pushed 10 Oct 2026; PyPI 1.1.0, 11 Jul 2026).
- **Install:** needs Python ≥3.10 and `torch>=2.13`. It installs cleanly for Python 3.12 / Windows (dry run: torch 2.14.1, numpy 2.5.4).
- **Weights (checked):** five ensemble models of about 5.7 MB each in the original repo's `cache_data/`, named `joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{0..4}.best.sdict`. lv-chordia bundles them (about 28 MB). Re-weighted variants are on Google Drive.
- **How it models chords:** it predicts root+triad, **bass**, 7th, 9th, 11th and 13th as separate outputs, then a CRF assembles the label from a chosen chord dictionary.
- **Chord dictionaries (I read the files):**
  - `submission` has 25 qualities × 12 roots + N = **301 classes**. It covers maj, min, aug, dim; **maj/3, maj/5, min/b3, min/5, maj/2, maj/b7, min/2, min/b7**; maj7, 7, min7, dim7, hdim7; maj9, 9, min9, 11, 13; sus2, sus4, sus4(b7).
  - The lv-chordia README says "about 170"; the file itself has 301.
  - `full` has **382 qualities**, including maj(9) (add9), maj6(9), 7(b9), 7(#9), maj/4, 7/3, 7/5, 7/b7, maj7/3, maj7/5, min7/b7, min11, maj(9)/3 and sus4(b7,9).
  - The README calls `full` "untested". The better move is to build a curated worship dictionary from it.
  - Gospel IV/5 (F/G in C) is maj/2, and 9sus4 is sus4(b7,9). Both are expressible.
- **Scores:**
  - [ChordFormer paper re-run](https://arxiv.org/html/2502.11840v1) on 1217 songs: Root 83.39, Thirds 80.04, MajMin 82.62, Triads 75.91, Sevenths 69.78, Tetrads 62.87, MIREX 81.52.
  - Original paper on the 301-class vocabulary: frame accuracy 0.772, class-wise accuracy 0.348. With re-weighting (0.5, 10): 0.746 / 0.402.
  - The paper's confusion analysis shows maj9/9/min9 collapsing to their 7th chords, and 11/13 rarely detected.
- **Time resolution:** CQT hop 512 at 22.05 kHz, about **23 ms**, the finest of all candidates.
- **Short chords:** the CRF switching penalty is d=30. That is the knob that suppresses short passing chords, so lower it, or decode on a beat grid.
- **Speed:** the README FAQ gives 10–20× real time on GPU, 2–5× on CPU (about 10–30 s per song).
- **Bass/inversions:** yes, explicitly modelled.

### 2. consonance-ACE (Poltronieri, Serra, Rocamora — ISMIR 2025)
- **Links:** [arXiv 2509.01588](https://arxiv.org/html/2509.01588v1), repo [andreamust/consonance-ACE](https://github.com/andreamust/consonance-ACE). MIT, pushed 2 Jul 2026.
- **Weights (checked):** `ACE/checkpoints/conformer_decomposed_smooth.ckpt`, 57.6 MB, inside the repo.
- **Install:** pins Python 3.11, torch 2.6.0, lightning 2.5.1, librosa 0.11.0, beat-this 0.1. Relaxing these for Python 3.12 / torch 2.14 is probably fine but untested.
- **How it models chords:** a Conformer that predicts root (13), **bass (13)** and **12 note activations**. Labels are rebuilt with harte-library and no fixed vocabulary, so in principle it can output add9 or other unusual chords.
- **Scores** (trained on Isophonics + Billboard, tested on RWC-Pop + USPop):
  - Root 84.0, MajMin 77.8, Thirds 80.3, Triads 72.7, Tetrads 60.8, Sevenths 66.0, MIREX 79.8.
  - Inversion-aware scores beat BTC in the same setup: MajMin-inv 75.6 vs 71.5, Sevenths-inv 64.4 vs 60.8, Tetrads-inv 59.4 vs 56.2.
- **Time resolution:** hop 2048, about 93 ms.
- **Inference:** `python -m ACE.inference --audio x.wav --out x.lab --chord-min-duration 0.5 [--beat-sync]`. Lower the minimum duration to catch passing chords.
- **Why it matters for fusion:** it outputs exactly what the fusion step needs — bass and pitch-class activations.

### 3. ChordMini BTC-CL / 2E1D (Phan et al., DAFx26, 2026)
- **Links:** [arXiv 2602.19778](https://arxiv.org/abs/2602.19778), [ptnghia-j/ChordMini](https://github.com/ptnghia-j/ChordMini). MIT, pushed June 2026.
- **Weights (checked):** in `checkpoints/`: `btc_model_best.pth` (35.9 MB), `2e1d_model_best.pth` (27.5 MB), plus the original BTC teacher.
- **Vocabulary:** 170 classes, no inversions. Hop 2048 (about 93 ms).
- **Scores** (120-song held-out test, out of 600 songs from Isophonics, Billboard, RWC and USPop):

| Model | Root | Thirds | Triads | 7ths | Tetrads | MajMin | MIREX |
|---|---|---|---|---|---|---|---|
| BTC-CL (full) | 83.03 | 80.17 | 78.33 | 69.38 | 67.00 | 80.24 | 80.16 |
| Original BTC teacher | 81.89 | 78.49 | 76.85 | 66.29 | 63.72 | 79.00 | 78.65 |

- **Rare chords** (vs the plain supervised baseline): maj7 28.2→42.3, dim7 0→45.6, sus 22.9→25.5.
- **Caveat:** the teacher's training data may overlap this test set.

### 4. BTC (Park et al., ISMIR 2019)
- **Links:** [jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19). Code MIT, last push May 2020.
- **Weights (checked):** `test/btc_model_large_voca.pt` and `test/btc_model.pt`, about 12.2 MB each. An HF repackage compatible with NumPy 2 / torch 2: [puar-playground/btc-chord](https://huggingface.co/puar-playground/btc-chord).
- **Vocabulary:** 170 classes = 12 roots × {maj, min, dim, aug, min6, maj6, min7, minmaj7, maj7, 7, dim7, hdim7, sus2, sus4} + X + N. **No bass, no inversions, no 9ths.**
- **Scores** ([paper](https://ar5iv.labs.arxiv.org/html/1907.02698), 5-fold on Isophonics221 + RW65 + USPop185, large vocabulary): Root 83.5, Thirds 80.8, Triads 75.9, Sevenths 71.8, Tetrads 65.5, MajMin 82.3, MIREX 80.8.
- **Time resolution:** about 93 ms. Very fast (about 3M parameters).
- **Licence note:** a third-party page claims the weights are effectively CC BY-NC-SA because of the training data ([cstr GGUF](https://huggingface.co/cstr/btc-chords-GGUF)). This only matters if the app is ever commercialised.

### 5. Chordino / NNLS-Chroma (Mauch & Dixon, ISMIR 2010)
- **Licence:** GPL.
- **Built-in dictionary (from [source](https://github.com/c4dm/nnls-chroma)):** 24-dimensional profiles (bass + treble chroma). Includes maj, min, m7b5, 6, 7, maj7, m7, m6, dim, aug, plus slash forms /3, /5, /2, /b7 and 7/3. You can edit `chord.dict` to add sus, add9 and others.
- **Accuracy:** clearly below the deep-learning models (lowest bars in Jiang 2019 Fig. 4).
- **Windows:**
  - Included in the [Vamp Plugin Pack 2.0 x64](https://www.vamp-plugins.org/download.html).
  - Run it through **sonic-annotator 1.7 win64** (released 28 May 2025, [GitHub release](https://github.com/sonic-visualiser/sonic-annotator/releases)) as a separate process.
  - The PyPI `vamp` host package only has 2015 Python 2.7 wheels, and `chord-extractor` 0.1.3 requires Python <3.12, so neither works directly.

### Usable but not recommended
- **madmom chords** (CNN+CRF, DeepChroma):
  - Major/minor only (25 classes) at 10 fps (checked in [source](https://github.com/CPJKU/madmom)).
  - Models are CC BY-NC-SA.
  - PyPI 0.16.1 (2018) is source-only. The main branch is 0.17.dev0 and lists Python 3.9–3.12 (last commit Aug 2024, "NumPy compatibility updates"). Building it needs Cython and numpy installed first, plus MSVC Build Tools.
- **omnizart chord:** Harmony Transformer, 25 classes, needs TensorFlow. 0.6.3 (31 May 2026) is source-only and depends on madmom, so it does not resolve as pre-built packages on Python 3.12 / Windows.
- **autochord:** Bi-LSTM-CRF, 25 classes, 67.33% accuracy, TensorFlow, needs the NNLS Vamp plugin, Apache-2.0.
- **crema** (McFee & Bello 2017): BSD-2, last release 0.2.0 in 2022. It does estimate inversions from a bass output, but it is built on Keras 2/TensorFlow, so it is a loading risk on Python 3.12 / Keras 3 (untested).
- **Sheet Sage:** produces chords, but needs Docker on Linux, and Jukebox features need 12 GB of GPU memory. Models are CC BY-NC-SA.
- **allin1:** **does not output chords** (beats and sections only). It also needs NATTEN, which has no Python 3.12 Windows build.

### Papers with no released weights (watch list)
- **ChordFormer** (arXiv 2502.11840, 2025): Root 84.69, MajMin 84.09, Sevenths 72.28, MIREX 83.62. No code found.
- **Kim & Park, ICASSP 2026** (arXiv 2604.24386): event-based modelling of non-triad chords. No code.
- **Mackenzie 2025 thesis** (arXiv 2512.22621): found that features from generative models did not help. No released weights.
- **Majchrzak & Mańdziuk 2025** (arXiv 2508.05878): trained on synthetic audio. No weights found.
- **ChordCoT** (arXiv 2509.18700): GPT-4o refinement step, +1 to +2.77 MIREX points. No code.

### Foundation-model probes
- **MERT-v2** ([HF](https://huggingface.co/m-a-p/MERT-v2-30s)): about 0.6B parameters, weights CC BY-NC. Reports Chords1217 frame accuracy of 78.48 with a probe; the chord vocabulary is not stated.
- **MuQ:** about 300M parameters, CC BY-NC weights.
- **Verdict:** no chord heads are released, so using these means training one ourselves, and the audio for the standard chord datasets is not distributed. Not worth it now; maybe later for fine-tuning on Nathan's own annotated songs.

---

## 2. Note-transcription models and separation for the notes-first branch

### Transcription models
- **ByteDance piano transcription** ([repo](https://github.com/qiuqiangkong/piano_transcription_inference)):
  - PyPI 0.0.6 (Jan 2025), MIT. Installs cleanly for Python 3.12 / Windows (dry run).
  - Checkpoint on [Zenodo 4034264](https://zenodo.org/records/4034264): `CRNN_note_F1=0.9677_pedal_F1=0.9186.pth`, 172 MB, CC BY 4.0. MAESTRO note F1 96.77.
  - Trained on solo piano, so it will produce false notes from bleed in separated stems. Filter by velocity and duration.
- **Basic Pitch:** Apache-2.0, instrument-agnostic, ONNX on Windows. Good for the **bass stem** (monophonic). Needs the `--no-deps` workaround on Python 3.12 (see Summary).
- **MuScriptor** (Kyutai/Mirelo, 2026, [repo](https://github.com/muscriptor/muscriptor)):
  - Multi-instrument, 36 instrument groups. Sizes 103M / 307M / 1.4B parameters.
  - Code MIT; **weights CC BY-NC and gated** behind a Hugging Face licence acceptance.
  - Python 3.10–3.12. On Windows it needs `--torch-backend=cu128` to use the GPU. PyPI 0.3.0 (Aug 2026).
  - Optional single-pass alternative to separation plus per-stem transcription.
- **YourMT3+** (2024): code GPL-3.0, weights inside the HF Space files. Heavy and untested on Windows.
- **Transkun** (ISMIR 2024): MIT. Needs `ncls`, which has no Windows build, so a pre-built-only dry run failed; it would need compiling.
- **Aria-AMT:** robust piano model, Python 3.11, installed from source.
- **MT3 and Onsets & Frames:** older JAX / TF1 stacks. Skip.

### Symbolic chord recognizer
- **BACHI** (ICASSP 2026, [arXiv 2510.06528](https://arxiv.org/abs/2510.06528), [repo](https://github.com/AndyWeasley2004/BACHI_Chord_Recognition), MIT):
  - Weights (checked): HF dataset repo `Itsuki-music/BACHI_Chord_Recognition/pop909_film_kdec/best_model.pt` (125 MB, 31.2M parameters).
  - Input: beat-synchronous piano roll (88 keys, 12 frames per beat). Predicts root, quality and **bass**.
  - POP909-CL (clean MIDI): Root 89.85, MajMin 89.87, **MajMinBass 88.52**, Sevenths 86.04, SeventhsBass 84.97.
  - POP909 is Chinese pop piano arrangements ([arXiv 2008.07142](https://arxiv.org/pdf/2008.07142)), a close match for Chinese worship.
  - Its accuracy on *transcribed* MIDI (with transcription errors) has not been measured.

### Separation
- **BS-Roformer-SW** (6 stems: vocals, bass, drums, guitar, piano, other):
  - Listed in audio-separator 0.47.0's model list (`BS-Roformer-SW.ckpt`).
  - [MVSEP](https://mvsep.com/news/64) reports SDR of piano 7.83, bass 14.62, guitar 9.05.
- **htdemucs_6s:** the Demucs README itself warns of "a lot of bleeding and artifacts" on the piano stem.

### Is the notes-first idea worth it? Yes — as a fusion input, not as a replacement
- **Separating helps:** Mitoma & Furuya (APSIPA 2025) separated stems, boosted the pitched instruments, and BTC fixed more than 2,000 misrecognised frames. I could not retrieve the exact size of the gain.
- **Bringing in note-level/symbolic sources helps:** DECIBEL (TISMIR 2021) improved every audio chord system tested by 0.5–13.6 points.
- **Why it fits worship/gospel:**
  - The bass stem directly answers which slash chord it is and catches walk-ups, which are the cases where Chordify fails.
  - Piano-stem notes reveal 9ths, 11ths and sus voicings that audio models trained on Billboard/Beatles almost never predict.
- **Risks:**
  - Transcription errors on dense pads and reverb.
  - Separation artefacts.
  - So: weight the transcription evidence by confidence, and never let it override strong agreement among the audio models.

---

## 3. Ranked ensemble (8 GB RTX 4070 laptop)

1. **lv-chordia**, run with a curated worship dictionary taken from `full`: add maj(9), maj6(9), sus4(b7,9), 7/3, maj7/3, maj/4, min7/b7, dim7, hdim7, 7(b9). Lower the CRF penalty or decode on beats. Provides root, triad, bass and extension estimates (exposing raw probabilities means hooking into the code).
2. **consonance-ACE**: root, bass and 12-note activations.
3. **ChordMini BTC-CL**: quality and root probabilities, strongest on maj7/dim7/sus.
4. **Notes-first branch:**
   - BS-Roformer-SW separation.
   - ByteDance transcription on the piano stem (and on guitar/other via Basic Pitch, or MuScriptor-medium for personal use).
   - Basic Pitch on the bass stem.
   - BACHI (pop model) on the beat-quantised MIDI.
   - A rule-based matcher for extensions and slash chords: lowest sustained bass note becomes the slash; IV/5 maps to maj/2 or sus4(b7,9); passing #idim7; walk-ups.
5. **Chordino** with a custom `chord.dict`: cheap bass/treble chroma for tie-breaking.

**Supporting pieces:**
- **Beat This!** for the beat grid: MIT, PyPI 1.1.0, no madmom needed.
- **ChordSync** (MIT, SMC 2024): optional "align my chord sheet" mode when the user has a chart.

**Fusion step:**
- Use half-beat cells (quarter-beat when bass onsets are dense).
- Combine root, bass and note probabilities with a log-linear weighting.
- Decode with a semi-Markov/HMM, using a key-aware transition prior learned from Chordonomicon (666k symbolic progressions, [arXiv 2410.22046](https://arxiv.org/abs/2410.22046)).
- Allow chords shorter than a beat only when bass onsets and note evidence agree.
- Tune the weights on 10–20 hand-annotated worship songs, scored with mir_eval's `majmin_inv`, `sevenths_inv` and `tetrads_inv`.
- This is the step that actually catches soft passing chords, because two of the strongest audio models use ~93 ms frames and a 0.5 s default minimum duration.

**Runtime per 4-minute song (estimated, not benchmarked):**

| Step | Time |
|---|---|
| Download + decode | 5–15 s |
| Beat This! | 2–5 s |
| lv-chordia | 10–25 s |
| consonance-ACE | 5–10 s |
| BTC-CL | under 5 s |
| Chordino (CPU, parallel) | about 10 s |
| BS-Roformer-SW | 40–90 s |
| Piano transcription | 10–20 s |
| Bass transcription | about 5 s |
| BACHI + fusion | under 5 s |

- **Full pipeline:** about 1.5–2.5 min, or about 1–1.5 min with CPU and GPU work overlapped.
- **Fast mode (no separation):** 30–45 s.
- Cache results by YouTube video ID.

## 4. Cautions
- **Licences:** madmom models, MuScriptor, MERT, MuQ and Sheet Sage weights are non-commercial. That is fine for personal use, not if the app is sold.
- **Loading checkpoints safely:** checkpoints and `.pkl`/`.sdict` files are pickles. Use `torch.load(weights_only=True)` wherever possible, and load only from the repos listed above. Never unpickle BACHI's `vocab.pkl` blindly; inspect it with `pickletools` first.
- **Untrustworthy page:** one search hit, github.com/K1ller-Whale/Chordify/pull/2, was reported to contain instructions aimed at AI assistants. I did not use it, and its benchmark claims are excluded.
- **User's intent is unclear:** the user's only message was the YouTube link plus "i pressed wrong!". The orchestrator should confirm what they meant before relying on that link as the test song.

Sources: [BTC repo](https://github.com/jayg996/BTC-ISMIR19) · [BTC paper](https://ar5iv.labs.arxiv.org/html/1907.02698) · [puar-playground/btc-chord](https://huggingface.co/puar-playground/btc-chord) · [cstr/btc-chords-GGUF](https://huggingface.co/cstr/btc-chords-GGUF) · [Jiang 2019 PDF](https://archives.ismir.net/ismir2019/paper/000078.pdf) · [music-x-lab repo](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition) · [lv-chordia](https://github.com/openmirlab/lv-chordia) · [ChordFormer](https://arxiv.org/html/2502.11840v1) · [consonance-ACE paper](https://arxiv.org/html/2509.01588v1) · [consonance-ACE repo](https://github.com/andreamust/consonance-ACE) · [ChordMini paper](https://arxiv.org/html/2602.19778) · [ChordMini repo](https://github.com/ptnghia-j/ChordMini) · [ChordMiniApp](https://github.com/ptnghia-j/ChordMiniApp) · [Kim & Park ICASSP 2026](https://arxiv.org/abs/2604.24386) · [Mackenzie thesis](https://arxiv.org/html/2512.22621) · [ChordCoT](https://arxiv.org/abs/2509.18700) · [nnls-chroma source](https://github.com/c4dm/nnls-chroma) · [Vamp downloads](https://www.vamp-plugins.org/download.html) · [sonic-annotator releases](https://github.com/sonic-visualiser/sonic-annotator/releases) · [chord-extractor](https://pypi.org/project/chord-extractor/) · [madmom](https://github.com/CPJKU/madmom) · [madmom PyPI](https://pypi.org/project/madmom/) · [omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) · [autochord](https://github.com/cjbayron/autochord) · [crema](https://github.com/bmcfee/crema) · [allin1](https://github.com/mir-aidj/all-in-one) · [Sheet Sage](https://github.com/chrisdonahue/sheetsage) · [MERT-v2](https://huggingface.co/m-a-p/MERT-v2-30s) · [MuQ](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) · [Basic Pitch PyPI](https://pypi.org/project/basic-pitch/) · [piano_transcription_inference](https://github.com/qiuqiangkong/piano_transcription_inference) · [Zenodo 4034264](https://zenodo.org/records/4034264) · [MuScriptor](https://github.com/muscriptor/muscriptor) · [MuScriptor model card](https://huggingface.co/MuScriptor/muscriptor-small/blob/main/README.md) · [YourMT3](https://github.com/mimbres/YourMT3) · [BACHI paper](https://arxiv.org/html/2510.06528v2) · [BACHI site](https://andyweasley2004.github.io/BACHI/) · [BACHI repo](https://github.com/AndyWeasley2004/BACHI_Chord_Recognition) · [Beat This](https://github.com/CPJKU/beat_this) · [ChordSync](https://github.com/andreamust/ChordSync) · [audio-separator](https://pypi.org/project/audio-separator/) · [MVSEP BS-Roformer SW](https://mvsep.com/news/64) · [APSIPA 2025 separation for ACE](https://webdev.apsipa.org/proceedings/2025/papers/APSIPA2025_P307.pdf) · [DECIBEL](https://research-portal.uu.nl/en/publications/improving-audio-chord-estimation-by-alignment-and-integration-of-/) · [Chordonomicon](https://arxiv.org/abs/2410.22046) · [POP909](https://arxiv.org/pdf/2008.07142)