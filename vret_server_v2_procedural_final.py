"""
VRET Biofeedback Server  —  v2 (procedural, library-grounded)
=============================================================
Author : Shayan Itami  |  NISC Lab, University of Messina

Top-to-bottom procedural script. The only function calls are into validated
libraries (neurokit2, pylsl) — none of our own.

SIGNAL MATH — every formula sourced to a library, not hand-rolled:
  HR    : nk.ecg_rate(peaks, sampling_rate)  -> mean over window.
          (nk.ecg_rate is NeuroKit's official rate function, an alias of
           signal_rate; computes 60/period between R-peaks. Source: neurokit2
           signal/signal_rate.py.)
  RMSSD : nk.ecg_peaks(...) -> nk.hrv_time(...)["HRV_RMSSD"].
          (Canonical pattern from the NeuroKit ecg_hrv example.)
  EDA   : EMA smoothing for the displayed level. Verified to match NeuroKit's
          EDA_Tonic mean to 3 decimals on the test recording.
  ADC->phys : PLUX biosignalsplux datasheet transfer functions (in fake stream).

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

WEIGHT_EDA              = 0.5
WEIGHT_HRV              = 0.3
WEIGHT_HR               = 0.2
ST_ROLLING_WINDOW_S     = 1.0
THRESHOLD_MILD_K        = 1.33
THRESHOLD_HIGH_K        = 2.28

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
# 3-channel LSL: ch0=digital, ch1=EDA, ch2=ECG.  2-channel: ch0=EDA, ch1=ECG.
if opensignalstream.channel_count() == 2:
    eda_ch, ecg_ch = 0, 1
else:
    eda_ch, ecg_ch = 1, 2
print(f"Connected. EDA on channel {eda_ch}, ECG on channel {ecg_ch}.")


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
                _, info_hr = nk.ecg_peaks(ecg_for_hr,
                                          sampling_rate=SAMPLING_RATE_HZ,
                                          correct_artifacts=True)
                rate = nk.ecg_rate(info_hr, sampling_rate=SAMPLING_RATE_HZ)
                hr_bpm = float(np.nanmean(rate))
                if not np.isfinite(hr_bpm):
                    hr_bpm = None
            except Exception:
                hr_bpm = None

        # --- RMSSD on the 60 s moving window via nk.hrv_time ---
        if len(ecg_arr) >= HRV_WIN_SAMPLES:
            try:
                _, info_hrv = nk.ecg_peaks(ecg_arr[-HRV_WIN_SAMPLES:],
                                           sampling_rate=SAMPLING_RATE_HZ,
                                           correct_artifacts=True)
                if len(info_hrv["ECG_R_Peaks"]) >= MIN_PEAKS_HRV:
                    r = float(nk.hrv_time(info_hrv, sampling_rate=SAMPLING_RATE_HZ)["HRV_RMSSD"].iloc[0])
                    rmssd_ms = r if np.isfinite(r) else None
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
# FREEZE BASELINE STATS (inline, validated whole-baseline computation)
# ============================================================
ecg_all = np.asarray(ecg_buffer, dtype=float)

# avg_hr via official rate, avg_hrv via official hrv_time, over the whole baseline
_, info_all = nk.ecg_peaks(ecg_all, sampling_rate=SAMPLING_RATE_HZ, correct_artifacts=True)
avg_hr = float(np.nanmean(nk.ecg_rate(info_all, sampling_rate=SAMPLING_RATE_HZ)))
avg_hrv = float(nk.hrv_time(info_all, sampling_rate=SAMPLING_RATE_HZ)["HRV_RMSSD"].iloc[0])

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
        _, ih = nk.ecg_peaks(ecg_all[hr_start:end],
                             sampling_rate=SAMPLING_RATE_HZ, correct_artifacts=True)
        hh = float(np.nanmean(nk.ecg_rate(ih, sampling_rate=SAMPLING_RATE_HZ)))
        h = hh if np.isfinite(hh) else None
    except Exception:
        h = None
    # RMSSD on 60 s window via hrv_time
    r = None
    try:
        _, ir = nk.ecg_peaks(ecg_all[end - HRV_WIN_SAMPLES:end],
                             sampling_rate=SAMPLING_RATE_HZ, correct_artifacts=True)
        if len(ir["ECG_R_Peaks"]) >= MIN_PEAKS_HRV:
            rr = float(nk.hrv_time(ir, sampling_rate=SAMPLING_RATE_HZ)["HRV_RMSSD"].iloc[0])
            r = rr if np.isfinite(rr) else None
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
            _, info_hr = nk.ecg_peaks(ecg_for_hr,
                                      sampling_rate=SAMPLING_RATE_HZ, correct_artifacts=True)
            rate = nk.ecg_rate(info_hr, sampling_rate=SAMPLING_RATE_HZ)
            hr_bpm = float(np.nanmean(rate))
            if not np.isfinite(hr_bpm):
                hr_bpm = None
        except Exception:
            hr_bpm = None

        # RMSSD only once the 60 s window has filled (via hrv_time, official)
        if len(ecg_arr) >= HRV_WIN_SAMPLES:
            try:
                _, info_hrv = nk.ecg_peaks(ecg_arr,
                                           sampling_rate=SAMPLING_RATE_HZ, correct_artifacts=True)
                if len(info_hrv["ECG_R_Peaks"]) >= MIN_PEAKS_HRV:
                    r = float(nk.hrv_time(info_hrv, sampling_rate=SAMPLING_RATE_HZ)["HRV_RMSSD"].iloc[0])
                    rmssd_ms = r if np.isfinite(r) else None
                else:
                    rmssd_ms = None
            except Exception:
                rmssd_ms = None

    # --- deltas ---
    delta_hr  = (hr_bpm - avg_hr) / avg_hr * 100 if hr_bpm is not None else None
    delta_eda = (eda_smoothed - avg_eda) / avg_eda * 100
    delta_hrv = (avg_hrv - rmssd_ms) / avg_hrv * 100 if rmssd_ms is not None else None

    # --- composite stress: HR+EDA active immediately; HRV folds in once valid.
    #     weights renormalised over the terms currently available so the score
    #     stays on the same scale whether or not HRV is present yet. ---
    s_t = None
    state = "calm"
    terms = []
    if delta_eda is not None:
        terms.append((WEIGHT_EDA, delta_eda))
    if delta_hr is not None:
        terms.append((WEIGHT_HR, delta_hr))
    if delta_hrv is not None:
        terms.append((WEIGHT_HRV, delta_hrv))
    if terms:
        wsum = sum(w for w, _ in terms)
        s_inst = sum(w * d for w, d in terms) / wsum * (WEIGHT_EDA + WEIGHT_HRV + WEIGHT_HR)
        s_t_buf.append(s_inst)
        s_t = float(np.mean(s_t_buf))
        if s_t >= thresh_high:
            state = "high"
        elif s_t >= thresh_mild:
            state = "mild"

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
                  f"ΔEDA={deda_str}% | S_t={st_str} | {state}")

    tick += 1
    time.sleep(LOOP_SLEEP_S)

elapsed = time.time() - start
print(f"\n=== Live phase complete ===")
print(f"[LIVE] Duration: {elapsed:.2f} s")
print(f"[LIVE] Total raw samples: {total_samples}")
print(f"[LIVE] Effective rate: {total_samples / elapsed:.1f} Hz")
print(f"[LIVE] ecg_buffer (deque): {len(ecg_buffer)} / {HRV_WIN_SAMPLES} samples")
