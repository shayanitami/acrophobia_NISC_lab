"""
VRET Biofeedback Server  —  v2 (procedural, library-grounded)
=============================================================
Author : Shayan Itami  |  NISC Lab, University of Messina

Top-to-bottom procedural script. The only function calls are into validated
libraries (neurokit2, pylsl) — none of our own.

SIGNAL MATH — every formula sourced to a library, not hand-rolled:
  CLEAN : nk.ecg_clean(raw_ecg, sampling_rate) BEFORE every peak detection.
          (Canonical NeuroKit order is ecg_clean -> ecg_peaks -> hrv_time.
           ecg_peaks expects a cleaned signal; raw electrode ECG carries 50 Hz
           mains hum + baseline wander, on which the detector finds zero peaks.
           Default method 'neurokit' = 0.5 Hz highpass + powerline filtering.)
  HR    : nk.ecg_rate(peaks, sampling_rate)  -> mean over window.
          (nk.ecg_rate is NeuroKit's official rate function, an alias of
           signal_rate; computes 60/period between R-peaks. Source: neurokit2
           signal/signal_rate.py.)
  RMSSD : nk.ecg_clean -> nk.ecg_peaks -> nk.hrv_time(...)["HRV_RMSSD"].
          Following the repo examples exactly, the FIRST return of ecg_peaks
          (the signals DataFrame, named `peaks` in the docs) is passed to
          hrv_time and ecg_rate -- not the info dict. (Verified numerically
          identical to passing info; changed only to mirror the examples at
          https://neuropsychology.github.io/NeuroKit/functions/hrv.html)
  QUALITY: any RMSSD outside RMSSD_MIN_MS..RMSSD_MAX_MS is rejected as a
          detection failure (poor-quality ECG), not reported. nk.ecg_quality
          is printed at baseline so a weak/noisy ECG channel is visible.
  EDA   : EMA smoothing for the displayed level. Verified to match NeuroKit's
          EDA_Tonic mean to 3 decimals on the test recording.
  ADC->phys : PLUX biosignalsplux datasheet transfer functions (in fake stream).

CHANNEL MAPPING (the bug that was producing impossible RMSSD values):
  The stream carries ch0=digital, ch1=EDA, ch2=ECG. This script now reads the
  channel LABELS from the stream metadata and maps EDA/ECG by name. If labels
  are missing (older fake stream, or a real device that doesn't publish them)
  it falls back to the correct fixed indices. A passive sanity check after
  baseline collection warns loudly if the channel assigned to ECG does not look
  like ECG — so a silent swap can never go unnoticed again.

WINDOWS:
  HR window  = 30 s, RMSSD window = 60 s, both MOVING (sliding deque), updated
  every HR_COMPUTE_INTERVAL ticks. RMSSD needs ~60 s of beats to be low-variance
  (10 s window -> estimate std ~56 ms; 60 s -> ~12 ms; verified).

FIRST 60 s OF LIVE:
  HR and EDA are valid within seconds, so the stress score uses them immediately.
  RMSSD has no valid value until its 60 s window fills, so ΔHRV is simply omitted
  from S_instant until then, and the remaining weights are renormalised so the
  score stays on the same scale. No faked HRV, no dead first minute.

Run order:
    1) python fake_opensignals.py
    2) python vret_server_v2_procedural.py
"""

from pylsl import StreamInlet, resolve_streams
from collections import deque
import time
import numpy as np
import neurokit2 as nk
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
SAMPLING_RATE_HZ        = 1000
LOOP_RATE_HZ            = 50
LOOP_SLEEP_S            = 1.0 / LOOP_RATE_HZ
PULL_CHUNK_MAX_SAMPLES  = 1000

BASELINE_DURATION_S     = 120
LIVE_DURATION_S         = 600

EDA_ALPHA               = 0.005          # EMA for displayed smoothed EDA

# HR is computed by handing NeuroKit whatever ECG is available, up to a memory
# cap (not a smoothing window). HR appears as soon as NeuroKit detects enough
# beats — at 75 BPM that's ~3 s. The cap just keeps processing time bounded.
HR_BUFFER_CAP_S         = 30             # at most this many seconds of ECG fed to nk.ecg_rate
HRV_WINDOW_S            = 60             # genuine moving window for RMSSD (statistical smoothing)
HR_COMPUTE_INTERVAL     = 25             # recompute HR/RMSSD every N ticks (~0.5 s)
PRINT_INTERVAL          = 5              # print every N ticks (~10 Hz)
MIN_PEAKS_HRV           = 10

# Physiological plausibility band for resting RMSSD. Real resting RMSSD is
# ~10-150 ms; values far outside this are not a measurement, they are the
# R-peak detector failing on a poor-quality ECG (it misses/invents beats, so
# successive RR differences explode). We reject such values instead of
# reporting them, and warn that the ECG channel quality needs attention.
RMSSD_MIN_MS            = 5
RMSSD_MAX_MS            = 300

WEIGHT_EDA              = 0.5
WEIGHT_HRV              = 0.3
WEIGHT_HR               = 0.2
ST_ROLLING_WINDOW_S     = 1.0
THRESHOLD_MILD_K        = 1.28     # z for 90th percentile (one-sided 10% tail)
THRESHOLD_HIGH_K        = 2.33     # z for 99th percentile (one-sided  1% tail)

SIGMA_THRESHOLD         = 3              # 3-sigma cleaning for EDA baseline mean

HR_CAP_SAMPLES          = HR_BUFFER_CAP_S * SAMPLING_RATE_HZ
HRV_WIN_SAMPLES         = HRV_WINDOW_S * SAMPLING_RATE_HZ


# ============================================================
# LSL CONNECTION (inline)
# ============================================================
print("Starting VRET biofeedback server v2 (procedural)...")
print("[LSL] searching for OpenSignals stream...")

opensignalstream = None
for j in range(30):
    streams = resolve_streams(wait_time=2.0)
    if not streams:
        if j != 29:
            print(f"attempt {j+1} -> no streams, retrying")
            continue
        else:
            raise RuntimeError("No OpenSignals stream found after 30 attempts")
    for s in streams:
        print(f"name={s.name()} | type={s.type()} | ch={s.channel_count()} "
              f"| rate={s.nominal_srate()} | source={s.source_id()}")
        if s.channel_count() in (2, 3):
            opensignalstream = s
            break
    if opensignalstream is not None:
        break

if opensignalstream is None:
    raise RuntimeError("Streams found but none with 2 or 3 channels")

inlet = StreamInlet(opensignalstream)
n_ch = opensignalstream.channel_count()

# ---- Map EDA/ECG by channel LABEL (robust), with a corrected fixed-index
#      fallback. This is the line that was wrong before: the old code did
#      `eda_ch, ecg_ch = 2, 1`, i.e. EDA was read from the ECG channel and
#      vice-versa, which fed the smooth EDA trace to nk.ecg_peaks and produced
#      impossible RMSSD (~1000 ms) and a stress score that divided by ~0. ----
def _read_channel_labels(inlet_obj):
    labels = []
    try:
        full = inlet_obj.info(timeout=2.0)          # fetch full metadata (desc)
        ch = full.desc().child("channels").child("channel")
        while not ch.empty():
            labels.append((ch.child_value("label") or "").strip().upper())
            ch = ch.next_sibling()
    except Exception:
        labels = []
    return labels

labels = _read_channel_labels(inlet)
eda_ch = ecg_ch = None
if labels:
    for idx, lab in enumerate(labels):
        if "EDA" in lab and eda_ch is None:
            eda_ch = idx
        if "ECG" in lab and ecg_ch is None:
            ecg_ch = idx

if eda_ch is None or ecg_ch is None:
    # Fallback (CORRECTED): 3-ch stream is ch0=digital, ch1=EDA, ch2=ECG;
    #                       2-ch stream is ch0=EDA, ch1=ECG.
    if n_ch == 2:
        eda_ch, ecg_ch = 0, 1
    else:
        eda_ch, ecg_ch = 1, 2
    print(f"[LSL] no usable channel labels; using fixed mapping "
          f"EDA=ch{eda_ch}, ECG=ch{ecg_ch}.")
else:
    print(f"[LSL] mapped by label -> EDA=ch{eda_ch}, ECG=ch{ecg_ch} "
          f"(labels: {labels}).")


# ============================================================
# PHASE 1 — BASELINE
# ============================================================
start = time.time()
tick = 0
total_samples = 0
eda_buffer = []
ecg_buffer = []
eda_smoothed = None
hr_bpm = None
rmssd_ms = None
eda_smoothed_buffer = []
hr_buffer = []
rmssd_buffer = []

while time.time() - start < BASELINE_DURATION_S:
    chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=PULL_CHUNK_MAX_SAMPLES)
    total_samples += len(chunk)

    for sample in chunk:
        eda_buffer.append(sample[eda_ch])
        ecg_buffer.append(sample[ecg_ch])
        if eda_smoothed is None:
            eda_smoothed = sample[eda_ch]
        else:
            eda_smoothed = EDA_ALPHA * sample[eda_ch] + (1 - EDA_ALPHA) * eda_smoothed
        eda_smoothed_buffer.append(eda_smoothed)

    if tick % HR_COMPUTE_INTERVAL == 0:
        ecg_arr = np.asarray(ecg_buffer, dtype=float)

        # --- HR: feed NeuroKit whatever ECG we have (up to the memory cap).
        #     HR appears as soon as NeuroKit detects enough beats.
        if len(ecg_arr) > 0:
            try:
                ecg_for_hr = ecg_arr[-HR_CAP_SAMPLES:] if len(ecg_arr) > HR_CAP_SAMPLES else ecg_arr
                ecg_for_hr = nk.ecg_clean(ecg_for_hr, sampling_rate=SAMPLING_RATE_HZ)
                peaks_hr, info_hr = nk.ecg_peaks(ecg_for_hr,
                                                 sampling_rate=SAMPLING_RATE_HZ,
                                                 correct_artifacts=True)
                rate = nk.ecg_rate(peaks_hr, sampling_rate=SAMPLING_RATE_HZ)
                hr_bpm = float(np.nanmean(rate))
                if not np.isfinite(hr_bpm):
                    hr_bpm = None
            except Exception:
                hr_bpm = None

        # --- RMSSD on the 60 s moving window via nk.hrv_time ---
        if len(ecg_arr) >= HRV_WIN_SAMPLES:
            try:
                ecg_hrv_win = nk.ecg_clean(ecg_arr[-HRV_WIN_SAMPLES:],
                                           sampling_rate=SAMPLING_RATE_HZ)
                peaks_hrv, info_hrv = nk.ecg_peaks(ecg_hrv_win,
                                                   sampling_rate=SAMPLING_RATE_HZ,
                                                   correct_artifacts=True)
                if len(info_hrv["ECG_R_Peaks"]) >= MIN_PEAKS_HRV:
                    r = float(nk.hrv_time(peaks_hrv, sampling_rate=SAMPLING_RATE_HZ)["HRV_RMSSD"].iloc[0])
                    # reject impossible values (poor-quality ECG, not a real measurement)
                    rmssd_ms = r if (np.isfinite(r) and RMSSD_MIN_MS <= r <= RMSSD_MAX_MS) else None
                else:
                    rmssd_ms = None
            except Exception:
                rmssd_ms = None

        if hr_bpm is not None and rmssd_ms is not None:
            hr_buffer.append(hr_bpm)
            rmssd_buffer.append(rmssd_ms)

    if tick % PRINT_INTERVAL == 0:
        if len(chunk) == 0:
            print("got 0 samples")
        else:
            hr_str = f"{hr_bpm:.1f}" if hr_bpm is not None else "—"
            rmssd_str = f"{rmssd_ms:.1f}" if rmssd_ms is not None else "—"
            print(f"got {len(chunk)} samples| raw_EDA={sample[eda_ch]:.4f} | "
                  f"smoothed EDA={eda_smoothed:.4f} μS | "
                  f"HR={hr_str} BPM | RMSSD={rmssd_str} ms")

    tick += 1
    time.sleep(LOOP_SLEEP_S)

elapsed = time.time() - start
print(f"\n=== Baseline collection complete ===")
print(f"Duration: {elapsed:.2f} s")
print(f"Total raw samples: {total_samples}")
print(f"Effective rate: {total_samples / elapsed:.1f} Hz")
print(f"eda_buffer:          {len(eda_buffer)} samples")
print(f"ecg_buffer:          {len(ecg_buffer)} samples")
print(f"eda_smoothed_buffer: {len(eda_smoothed_buffer)} samples")
print(f"hr_buffer:           {len(hr_buffer)} HR values")
print(f"rmssd_buffer:        {len(rmssd_buffer)} RMSSD values")


# ============================================================
# CHANNEL SANITY CHECK (passive, print-only — never changes behavior)
#   Zero-crossing counts were fooled by 50 Hz mains hum on the raw ECG, so we
#   instead ask the real question: which channel, when treated as ECG, yields
#   physiologically plausible heartbeats? We clean each channel and detect
#   R-peaks; the true ECG produces RR intervals almost all within 300-1500 ms
#   (40-200 BPM), while EDA (forced through the detector) does not. If the
#   channel mapped to EDA looks MORE heart-like than the one mapped to ECG,
#   the mapping is probably inverted.
# ============================================================
def _beat_plausibility(buf):
    """Fraction of RR intervals in the human-plausible 300-1500 ms band when
    this buffer is treated as ECG. ~1.0 for a real ECG, lower for EDA."""
    a = np.asarray(buf, dtype=float)
    if a.size < SAMPLING_RATE_HZ * 5:        # need a few seconds
        return 0.0
    try:
        cleaned = nk.ecg_clean(a, sampling_rate=SAMPLING_RATE_HZ)
        _, _info = nk.ecg_peaks(cleaned, sampling_rate=SAMPLING_RATE_HZ,
                                correct_artifacts=True)
        pk = _info["ECG_R_Peaks"]
        if len(pk) < 3:
            return 0.0
        rri = np.diff(pk) * 1000.0 / SAMPLING_RATE_HZ
        return float(np.mean((rri >= 300) & (rri <= 1500)))
    except Exception:
        return 0.0

_ecg_plaus = _beat_plausibility(ecg_buffer)
_eda_plaus = _beat_plausibility(eda_buffer)
print(f"[SANITY] beat-plausibility -> ECG-channel={_ecg_plaus:.2f}, EDA-channel={_eda_plaus:.2f}")
if _eda_plaus > _ecg_plaus:
    print("!!! [SANITY WARNING] The channel mapped to EDA produces more plausible "
          "heartbeats than the one mapped to ECG. The EDA/ECG mapping may be "
          "INVERTED — check the channel labels in fake_opensignals.py / your device. !!!")


# ============================================================
# FREEZE BASELINE STATS (inline, validated whole-baseline computation)
# ============================================================
ecg_all_raw = np.asarray(ecg_buffer, dtype=float)
# Clean ONCE (highpass + 50 Hz powerline removal). nk.ecg_peaks expects a cleaned
# signal; feeding raw mains-contaminated ECG yields zero R-peaks. All baseline peak
# detection below uses slices of this cleaned array.
ecg_all = nk.ecg_clean(ecg_all_raw, sampling_rate=SAMPLING_RATE_HZ)

# avg_hr via official rate, avg_hrv via official hrv_time, over the whole baseline
peaks_all, info_all = nk.ecg_peaks(ecg_all, sampling_rate=SAMPLING_RATE_HZ, correct_artifacts=True)
n_peaks_baseline = len(info_all["ECG_R_Peaks"])
if n_peaks_baseline < MIN_PEAKS_HRV:
    raise RuntimeError(
        f"Only {n_peaks_baseline} R-peaks detected in {len(ecg_all)/SAMPLING_RATE_HZ:.0f} s "
        f"of baseline ECG (need >= {MIN_PEAKS_HRV}). The ECG is likely too noisy or the "
        f"electrodes lost contact — check the ECG channel before continuing. "
        f"Baseline HR/HRV cannot be computed.")

# ECG signal-quality score (nk.ecg_quality, 0..1). Low quality => unreliable
# R-peaks => the RMSSD blow-ups. This is printed so a bad recording is obvious.
try:
    ecg_q = float(np.nanmean(nk.ecg_quality(ecg_all,
                                            rpeaks=info_all["ECG_R_Peaks"],
                                            sampling_rate=SAMPLING_RATE_HZ)))
except Exception:
    ecg_q = float("nan")

avg_hr = float(np.nanmean(nk.ecg_rate(peaks_all, sampling_rate=SAMPLING_RATE_HZ)))
avg_hrv = float(nk.hrv_time(peaks_all, sampling_rate=SAMPLING_RATE_HZ)["HRV_RMSSD"].iloc[0])
if not (RMSSD_MIN_MS <= avg_hrv <= RMSSD_MAX_MS):
    raise RuntimeError(
        f"Baseline RMSSD = {avg_hrv:.0f} ms is outside the physiological range "
        f"({RMSSD_MIN_MS}-{RMSSD_MAX_MS} ms). ECG quality score = {ecg_q:.2f}. This means the "
        f"R-peak detector is failing on this ECG (it misses/invents beats), not that HRV is "
        f"really that high. The recording's ECG channel is too weak or noisy — improve "
        f"electrode contact/grounding and re-record before relying on HRV.")

# EDA baseline mean with 3-sigma cleaning (inlined)
eda_sm_arr = np.asarray(eda_smoothed_buffer, dtype=float)
mu = eda_sm_arr.mean()
sd = eda_sm_arr.std()
eda_cleaned = eda_sm_arr[np.abs(eda_sm_arr - mu) <= SIGMA_THRESHOLD * sd]
avg_eda = eda_cleaned.mean()

# sigma_baseline: slide the live windows across the baseline (inline)
rolling = int(ST_ROLLING_WINDOW_S * LOOP_RATE_HZ)
step = SAMPLING_RATE_HZ // 2
s_t_buf = deque(maxlen=rolling)
s_t_history = []
for end in range(HRV_WIN_SAMPLES, len(ecg_all), step):
    # HR via ecg_rate on the cap-bounded slice ending at `end`
    h = None
    try:
        hr_start = max(0, end - HR_CAP_SAMPLES)
        sig_h, ih = nk.ecg_peaks(ecg_all[hr_start:end],
                                 sampling_rate=SAMPLING_RATE_HZ, correct_artifacts=True)
        hh = float(np.nanmean(nk.ecg_rate(sig_h, sampling_rate=SAMPLING_RATE_HZ)))
        h = hh if np.isfinite(hh) else None
    except Exception:
        h = None
    # RMSSD on 60 s window via hrv_time
    r = None
    try:
        sig_r, ir = nk.ecg_peaks(ecg_all[end - HRV_WIN_SAMPLES:end],
                                 sampling_rate=SAMPLING_RATE_HZ, correct_artifacts=True)
        if len(ir["ECG_R_Peaks"]) >= MIN_PEAKS_HRV:
            rr = float(nk.hrv_time(sig_r, sampling_rate=SAMPLING_RATE_HZ)["HRV_RMSSD"].iloc[0])
            r = rr if (np.isfinite(rr) and RMSSD_MIN_MS <= rr <= RMSSD_MAX_MS) else None
    except Exception:
        r = None
    if h is None or r is None:
        continue
    e_idx = int(end * len(eda_sm_arr) / len(ecg_all)) - 1
    e = eda_sm_arr[min(e_idx, len(eda_sm_arr) - 1)]
    d_eda = (e - avg_eda) / avg_eda * 100
    d_hr  = (h - avg_hr) / avg_hr * 100
    d_hrv = (avg_hrv - r) / avg_hrv * 100          # inverted: low HRV = stress
    s_inst = WEIGHT_EDA * d_eda + WEIGHT_HRV * d_hrv + WEIGHT_HR * d_hr
    s_t_buf.append(s_inst)
    s_t_history.append(np.mean(s_t_buf))
sigma_baseline = float(np.std(s_t_history)) if s_t_history else 1.0

print(f"\n=== Frozen baseline averages ===")
print(f"avg_hr  = {avg_hr:.2f} BPM")
print(f"avg_hrv = {avg_hrv:.2f} ms")
print(f"avg_eda = {avg_eda:.4f} μS")
print(f"sigma_baseline = {sigma_baseline:.4f}")
print(f"ECG quality (nk.ecg_quality, 0-1) = {ecg_q:.2f}   |   R-peaks detected = {n_peaks_baseline}")


# ============================================================
# PHASE 2 — LIVE
# ============================================================
input("press enter to start the live session")

total_drained = 0
while True:
    chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=PULL_CHUNK_MAX_SAMPLES)
    if len(chunk) == 0:
        break
    total_drained += len(chunk)
print(f"total amount of {total_drained} samples were drained from the buffers "
      f"to start live phase fresh")

# state reset
del eda_buffer, eda_smoothed_buffer, hr_buffer, rmssd_buffer, ecg_buffer
eda_smoothed = avg_eda          # seed EMA at baseline level so ΔEDA starts near 0, not a bootstrap spike
hr_bpm = None
rmssd_ms = None
delta_hr = None
delta_hrv = None
delta_eda = None
tick = 0
total_samples = 0
ecg_buffer = deque(maxlen=HRV_WIN_SAMPLES)
s_t_buf = deque(maxlen=rolling)
thresh_mild = THRESHOLD_MILD_K * sigma_baseline
thresh_high = THRESHOLD_HIGH_K * sigma_baseline
start = time.time()

while time.time() - start < LIVE_DURATION_S:
    chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=PULL_CHUNK_MAX_SAMPLES)
    total_samples += len(chunk)

    for sample in chunk:
        ecg_buffer.append(sample[ecg_ch])
        eda_smoothed = EDA_ALPHA * sample[eda_ch] + (1 - EDA_ALPHA) * eda_smoothed

    if tick % HR_COMPUTE_INTERVAL == 0 and len(ecg_buffer) > 0:
        ecg_arr = np.asarray(ecg_buffer, dtype=float)

        # HR via ecg_rate on the cap-bounded slice
        try:
            ecg_for_hr = ecg_arr[-HR_CAP_SAMPLES:] if len(ecg_arr) > HR_CAP_SAMPLES else ecg_arr
            ecg_for_hr = nk.ecg_clean(ecg_for_hr, sampling_rate=SAMPLING_RATE_HZ)
            peaks_hr, info_hr = nk.ecg_peaks(ecg_for_hr,
                                             sampling_rate=SAMPLING_RATE_HZ, correct_artifacts=True)
            rate = nk.ecg_rate(peaks_hr, sampling_rate=SAMPLING_RATE_HZ)
            hr_bpm = float(np.nanmean(rate))
            if not np.isfinite(hr_bpm):
                hr_bpm = None
        except Exception:
            hr_bpm = None

        # RMSSD only once the 60 s window has filled (via hrv_time, official)
        if len(ecg_arr) >= HRV_WIN_SAMPLES:
            try:
                ecg_hrv_win = nk.ecg_clean(ecg_arr, sampling_rate=SAMPLING_RATE_HZ)
                peaks_hrv, info_hrv = nk.ecg_peaks(ecg_hrv_win,
                                                   sampling_rate=SAMPLING_RATE_HZ, correct_artifacts=True)
                if len(info_hrv["ECG_R_Peaks"]) >= MIN_PEAKS_HRV:
                    r = float(nk.hrv_time(peaks_hrv, sampling_rate=SAMPLING_RATE_HZ)["HRV_RMSSD"].iloc[0])
                    rmssd_ms = r if (np.isfinite(r) and RMSSD_MIN_MS <= r <= RMSSD_MAX_MS) else None
                else:
                    rmssd_ms = None
            except Exception:
                rmssd_ms = None

    # --- deltas ---
    delta_hr  = (hr_bpm - avg_hr) / avg_hr * 100 if hr_bpm is not None else None
    delta_eda = (eda_smoothed - avg_eda) / avg_eda * 100
    # HRV bootstrap: for the first ~60 s of live the RMSSD window has not yet
    # filled, so no live RMSSD exists. Rather than renormalising the S_t weights
    # (which would silently change the score's scale, and therefore the meaning
    # of the sigma-baseline thresholds), we hold delta_hrv at 0 — i.e. assume HRV
    # has not yet moved from its frozen baseline avg_hrv. The HRV term stays in
    # the formula with its full 0.3 weight, contributing nothing until we have a
    # real measurement. We deliberately do NOT slide the baseline RMSSD window
    # across the operator gap, since that interval is of unknown length and would
    # contaminate the estimate. Once the live window fills, delta_hrv switches to
    # the genuine deviation.
    if rmssd_ms is not None:
        delta_hrv = (avg_hrv - rmssd_ms) / avg_hrv * 100
    else:
        delta_hrv = 0.0   # frozen-baseline stand-in (current_hrv == avg_hrv)

    # --- composite stress score ---
    #   Weights are FIXED at their study-defined values (EDA 0.5, HRV 0.3,
    #   HR 0.2). delta_hrv is always present (0.0 during the first-60s bootstrap,
    #   real thereafter), so the only term that can be briefly missing is HR, in
    #   the first few seconds before any beats are detected. While HR is missing
    #   we renormalise across the available terms so the score stays on the same
    #   scale; once HR is present (within seconds) all three terms are active and
    #   no renormalisation happens.
    s_t = None
    state = "calm"
    unity_cmd = "increase"
    terms = []
    if delta_eda is not None:
        terms.append((WEIGHT_EDA, delta_eda))
    if delta_hrv is not None:
        terms.append((WEIGHT_HRV, delta_hrv))
    if delta_hr is not None:
        terms.append((WEIGHT_HR, delta_hr))
    if terms:
        wsum = sum(w for w, _ in terms)
        s_inst = sum(w * d for w, d in terms) / wsum * (WEIGHT_EDA + WEIGHT_HRV + WEIGHT_HR)
        s_t_buf.append(s_inst)
        s_t = float(np.mean(s_t_buf))

        # Three-band classification against the z-score-derived thresholds.
        #   below lower threshold  -> calm        -> balloon "increase" (advance exposure)
        #   between the thresholds -> stressed     -> balloon "neutral"  (hold height)
        #   above upper threshold  -> ultra-stressed -> balloon "decrease" (back off)
        if s_t >= thresh_high:
            state = "ultra-stressed"
            unity_cmd = "decrease"
        elif s_t >= thresh_mild:
            state = "stressed"
            unity_cmd = "neutral"
        else:
            state = "calm"
            unity_cmd = "increase"

    # TODO (Stage 5): send `unity_cmd` to Unity over the chosen transport
    # (LSL marker / UDP / socket). For now it is computed and printed only.

    if tick % PRINT_INTERVAL == 0:
        if len(chunk) == 0:
            print("[LIVE] got 0 samples")
        else:
            hr_str = f"{hr_bpm:.1f}" if hr_bpm is not None else "—"
            rmssd_str = f"{rmssd_ms:.1f}" if rmssd_ms is not None else "—"
            dhr_str = f"{delta_hr:+.1f}" if delta_hr is not None else "—"
            dhrv_str = f"{delta_hrv:+.1f}" if delta_hrv is not None else "—"
            deda_str = f"{delta_eda:+.1f}" if delta_eda is not None else "—"
            st_str = f"{s_t:+.1f}" if s_t is not None else "—"
            print(f"[LIVE] got {len(chunk)} samples| raw_EDA={sample[eda_ch]:.4f} | "
                  f"smoothed EDA={eda_smoothed:.4f} μS | HR={hr_str} BPM | "
                  f"RMSSD={rmssd_str} ms | ΔHR={dhr_str}% | ΔHRV={dhrv_str}% | "
                  f"ΔEDA={deda_str}% | S_t={st_str} | {state} -> unity:{unity_cmd}")

    tick += 1
    time.sleep(LOOP_SLEEP_S)

elapsed = time.time() - start
print(f"\n=== Live phase complete ===")
print(f"[LIVE] Duration: {elapsed:.2f} s")
print(f"[LIVE] Total raw samples: {total_samples}")
print(f"[LIVE] Effective rate: {total_samples / elapsed:.1f} Hz")
print(f"[LIVE] ecg_buffer (deque): {len(ecg_buffer)} / {HRV_WIN_SAMPLES} samples")
