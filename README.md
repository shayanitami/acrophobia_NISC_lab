# VRET Biofeedback Pipeline — Acrophobia Project

Real-time physiological biofeedback for a Virtual Reality Exposure Therapy
(VRET) system targeting acrophobia, developed at the **NISC Lab, University of
Messina**. The pipeline reads ECG and EDA from a PLUX biosignals hub via LSL,
computes a per-subject composite arousal score, classifies the patient's state,
and sends balloon-altitude commands to a Unity VR scene over UDP — closing the
loop so the exposure adapts to the patient's autonomic state.

> **Status:** research prototype, not clinical software. False-positive
> behaviour and aroused/calm discrimination have been validated on multiple
> recordings; absolute threshold calibration against a graded-arousal protocol
> is still open work. See *Validation & limitations* below.

---

## How it works (one paragraph)

A 120 s **baseline phase** captures the subject's resting ECG and EDA. From
that baseline three frozen references are extracted: `avg_hr`, `avg_hrv`
(RMSSD, with Kubios + Malik 20 % artifact gating), and the phasic-EDA mean and
spread (`eda_mean`, `eda_sigma`). The **live phase** then computes, every
~0.5 s, a percentage HR deviation, a percentage HRV deviation, and the current
phasic EDA value; each is z-scored against its own baseline spread and combined
into a weighted composite score:

```
S_t = 0.5 · z(EDA_phasic) + 0.3 · z(HRV) + 0.2 · z(HR)
```

`S_t` is then smoothed and classified into **calm / stressed / ultra-stressed**
using thresholds at the 1.28σ and 2.33σ points of the baseline score
distribution. The corresponding UDP command (`increase` / `neutral` /
`decrease`) is sent to Unity, which raises or lowers the balloon altitude
accordingly.

The detailed mathematical and implementation rationale for each signal is in
`pdfs/`.

---

## Repository layout

```
vret_pipeline/
├── README.md                ← this file
├── .gitignore
├── vret_server.py           ← the live biofeedback server
├── fake_opensignals.py      ← LSL replayer that plays a recording as if it
│                              were a live device (for development/testing)
├── pdfs/
│   ├── hr_calculation.pdf   ← line-by-line walkthrough of the HR signal
│   ├── hrv_calculation.pdf  ← line-by-line walkthrough of the HRV signal
│   └── eda_calculation.pdf  ← line-by-line walkthrough of the EDA signal
└── samples/                 ← recorded OpenSignals .txt sessions used for
                               validation (calm, aroused, etc.)
```

The PDFs in `pdfs/` are self-contained methodology documents — each explains
its signal end-to-end, with the relevant code blocks quoted line-by-line and
academic references attached to every methodological choice.

---

## Running the pipeline

The pipeline can run against the **real PLUX device** or against a **recorded
sample replayed through LSL**. The replay path is what the development work
uses, and the only difference at runtime is who is producing the LSL stream.

### With a recorded sample (development / testing)

In two terminals:

```bash
# Terminal 1 — start the fake LSL stream replaying a recording
python fake_opensignals.py samples/<recording_name>.txt

# Terminal 2 — start the biofeedback server
python vret_server.py
```

The fake stream prints `[wait] holding at sample 0; waiting for 'start'…` and
then waits. In the server terminal, press **Enter** when prompted — this sends
a UDP packet that releases the fake stream, so both processes always begin from
the same sample of the recording. Baseline then proceeds for 120 s, followed by
the live phase.

### With the real PLUX device

Start OpenSignals (or whatever LSL-publishing tool the lab uses) so the device
is broadcasting on LSL, then just run:

```bash
python vret_server.py
```

The "Enter to start" prompt still appears; with the real device there is no
listener for the start packet, so it's harmlessly ignored — the prompt simply
becomes an operator-ready gate.

### What you'll see in the terminal

- A `[SANITY]` line on baseline start: beat-plausibility on each LSL channel.
  Confirms ECG and EDA were mapped correctly (label-based mapping handles
  reversed sensor orders).
- `[SIGMA-FLOOR]` warnings if a baseline sigma is implausibly small and a
  physiological floor had to engage.
- `[RR-GATE]` log on baseline freeze: the gated RMSSD alongside what raw
  `hrv_time` would have given, so a noisy baseline is visible.
- `[EDA-CLAMP]` whenever a phasic EDA value is rejected as a decomposition
  artifact (resampler edge ringing, see `pdfs/eda_calculation.pdf` §4).
- Per-tick `[LIVE]` lines with raw EDA, HR, RMSSD, the three deltas, the
  current `S_t`, the classified state, and the Unity command.

---

## Requirements

- **Python 3.9+**
- The pipeline assumes ECG and EDA are sampled at **1000 Hz** (the PLUX
  default). Recordings at other rates are not currently supported — the loop
  hardcodes the sampling rate throughout.
- Python packages (install into a venv; `vret_env/` is gitignored):

```
pylsl
neurokit2
numpy
scipy
```

A minimal `pip install pylsl neurokit2 numpy scipy` is usually sufficient.

---

## The composite score, briefly

Each signal is z-scored against **its own** baseline mean and spread before the
weighted sum is taken. This matters: without z-scoring, the three signals
(EDA in µS, HR and HRV in percent deviation) live on different numeric scales
and the largest-magnitude one would dominate regardless of the intended
`0.5 / 0.3 / 0.2` weights. After z-scoring, the weights mean what they say.

Weights, thresholds, and bootstrap behaviour are documented signal-by-signal
in the PDFs.

---

## Validation & limitations

What has been checked on real recordings, and what has not:

- **False-positive behaviour:** verified on calm recordings (Paria session, a
  reversed-sensor-order session). The pipeline correctly reads calm data as
  ~89 % calm.
- **Aroused vs calm discrimination:** verified on the VAR session, where the
  subject was genuinely aroused. The pipeline correctly elevates to ~17 %
  ultra-stressed and ~21 % stressed on that session, driven by multi-channel
  agreement (HR rise + HRV suppression + phasic EDA), not artifacts.
- **Noise mirroring:** a 14-minute deliberately-noisy session (cable
  movement) was correctly reflected in the output — the pipeline did not
  hide the noise. Good faithfulness signal.

What is **not yet validated**:

- **Absolute band placement.** The pipeline separates aroused from calm, but
  whether "ultra-stressed" corresponds to a specific clinical arousal level
  needs a controlled graded-arousal recording to calibrate the 1.28σ / 2.33σ
  thresholds.
- **Non-1000 Hz recordings.** A 200 Hz recording silently produced compromised
  output (the time axis was compressed). The pipeline currently has no rate
  detection and trusts the configured 1000 Hz.

Known signal-processing quirks (documented but not all fixed):

- A rare (~0.5 % of ticks) decomposition artifact in phasic EDA: the
  resampler intermittently corrupts its last output sample, causing the
  zero-phase highpass filter to ring. Mitigated by a plausibility clamp at
  1.0 µS (`[EDA-CLAMP]` log); the underlying resampler bug is left
  unrepaired because the repair heuristics tested risked misfiring on real
  data.
- Baseline-quality dependence on HRV: the Malik 20 % gate cleans a few bad
  beats from an otherwise-good baseline, but cannot rescue a globally noisy
  baseline. A poor-quality baseline (low `ecg_quality` score) should be
  re-recorded rather than trusted.

See `pdfs/hr_calculation.pdf`, `pdfs/hrv_calculation.pdf`, and
`pdfs/eda_calculation.pdf` for the full per-signal discussion of these
points and the references behind every methodological choice.

---

## Author & affiliation

Developed by **Shayan Itami** as part of the acrophobia VRET project at the
**Neuroscience Imaging and Stimulation Center (NISC) Lab**, University of
Messina. Pipeline rebuilt May–June 2026.
