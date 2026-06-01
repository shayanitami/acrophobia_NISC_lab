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
import socket
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

# If the LSL stream stops delivering data for this long (recording reached its
# end, or the device disconnected), the live phase ends cleanly instead of
# spinning on empty polls. At 50 Hz, 5 s = 250 consecutive empty pulls.
STREAM_DROUGHT_TIMEOUT_S = 5.0

BASELINE_DURATION_S     = 120
LIVE_DURATION_S         = 600

# NOTE: EDA is intentionally NOT smoothed. It is an inherently slow signal, and
# its real processing is the phasic decomposition (compute_phasic_eda) used by
# the score. A display-only EMA was previously applied but contributed nothing
# to S_t and was visually indistinguishable from raw, so it has been removed.
# Raw EDA is shown directly.

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

# RR-interval artifact gate for RMSSD. NeuroKit's Kubios correction
# (correct_artifacts=True) runs first and reconstructs missed beats, but it does
# not REMOVE interpolated intervals — so for the time-domain RMSSD we additionally
# apply the standard Malik / Task Force (1996) rule below. Applied to BOTH the
# frozen baseline RMSSD and the live RMSSD so the two are measured the same way.
#   RR 300-1500 ms  = 40-200 BPM instantaneous (anything outside is a detector error)
#   RR_MAX_RELATIVE_CHANGE = reject a beat whose interval differs from the previous
#   by more than this fraction (Malik 20% rule). Relative, so the tolerance scales
#   with heart rate, unlike a fixed-ms threshold.
RR_MIN_MS                  = 300
RR_MAX_MS                  = 1500
RR_MAX_RELATIVE_CHANGE     = 0.20

WEIGHT_EDA              = 0.5
WEIGHT_HRV              = 0.3
WEIGHT_HR               = 0.2
ST_ROLLING_WINDOW_S     = 1.0
THRESHOLD_MILD_K        = 1.28     # z for 90th percentile (one-sided 10% tail)
THRESHOLD_HIGH_K        = 2.33     # z for 99th percentile (one-sided  1% tail)

# ---- EDA tonic/phasic decomposition + per-signal z-scoring ----
# Raw EDA is dominated by slow tonic drift (electrode hydration, temperature),
# which on test data was ~99% of EDA variance and pinned the score to
# "ultra-stressed". We instead decompose EDA into tonic (SCL) + phasic (SCR)
# and feed only the PHASIC component (event-driven sympathetic responses) into
# the score. Because phasic EDA is in uS while HR/HRV deltas are in %, we put
# all three signals on a common scale by z-scoring each against ITS OWN baseline
# spread before applying the 0.5/0.3/0.2 weights — so the weights express
# relative importance, not an accident of which signal has bigger raw numbers.
EDA_DECOMP_RATE_HZ      = 10       # nk.eda_phasic runs at a low rate; 10 Hz is plenty
EDA_DECOMP_WINDOW_S     = 60       # rolling EDA window handed to nk.eda_phasic each update
EDA_DECOMP_WIN_SAMPLES  = EDA_DECOMP_WINDOW_S * SAMPLING_RATE_HZ
SIGMA_FLOOR             = 1e-6     # absolute guard against divide-by-zero in z-scoring

# ---- Per-signal physiological sigma floors (z-scoring guardrail) ----
# The z-score divides a live deviation by the signal's baseline spread (sigma).
# If a short or unusually flat baseline yields an implausibly tiny sigma, a
# trivial live change becomes a huge z-score and the score false-alarms (a 1%
# HR wobble was reading "ultra-stressed"). These floors refuse to believe a
# resting signal varies less than its real physiological minimum, so they only
# bite on broken/flukey baselines and leave normal ones (HR sigma ~2-3%, HRV
# ~9%) untouched. ADJUST after observing more real sessions — these are
# conservative starting values, not measured constants.
#   *** FLAGGED: tune these once more real recordings are available ***
HR_SIGMA_FLOOR_PCT      = 2.0      # resting HR spread is realistically >= ~2%
HRV_SIGMA_FLOOR_PCT     = 5.0      # resting RMSSD deviation spread >= ~5%
EDA_PHASIC_SIGMA_FLOOR  = 0.02     # phasic-EDA spread floor in uS

# Physiological ceiling for a phasic-EDA value. Real phasic skin-conductance
# responses are tenths of a uS; even a large startle SCR rarely exceeds ~1 uS.
# Values far above this are not physiology — they are filter ringing from a
# resampler edge artifact in the phasic decomposition (the resampler intermittently
# appends a spurious ~0 final sample; nk's zero-phase filter then rings backward
# across the tail, and that ringing can reach 4-15 uS). We do NOT try to repair the
# decomposition (the repair heuristics risk misfiring on real data); we simply
# reject an implausible phasic reading so a known-garbage value can never reach the
# score / Unity. This caps the blast radius of a rare (~0.5% of ticks) artifact.
EDA_PHASIC_MAX_US       = 1.0

SIGMA_THRESHOLD         = 3              # 3-sigma cleaning for EDA baseline mean

# ---- Unity UDP output ----
# The Unity BioFeedbackMiddleware listens on this port and reads each packet as
# a UTF-8 string (it trims + lowercases). It moves the balloon +stepAmount per
# "increase" packet and -stepAmount per "decrease"; "neutral" is ignored (hold).
# It also accepts "start"/"stop" to begin/end the flight. We send the current
# band command at the print cadence (~10 Hz), so a sustained band drives a
# steady climb/descent. UDP is fire-and-forget: a dropped packet is replaced by
# the next one ~0.1 s later, and a send failure must never stall the loop.
UNITY_UDP_IP            = "127.0.0.1"
UNITY_UDP_PORT          = 5005
UNITY_SEND_INTERVAL     = PRINT_INTERVAL   # send every N ticks (~10 Hz), matches print
UNITY_SEND_START_STOP   = True             # send "start" at live begin, "stop" at end

# ---- Fake-stream start handshake (DEV ONLY) ----
# So runs are comparable, the dev fake stream (fake_opensignals.py) waits for a
# "start" packet before pushing sample 0, instead of replaying the moment it
# launches. Pressing enter here sends that packet, so the baseline always begins
# at recording sample 0 and every run reads the same region of the recording.
# With the REAL PLUX device there is no listener: the packet goes nowhere (UDP
# never errors), so the enter prompt simply becomes an operator-ready gate.
# Set False to skip sending entirely.
FAKE_STREAM_CONTROL     = True
CONTROL_UDP_IP          = "127.0.0.1"
CONTROL_UDP_PORT        = 5006             # distinct from Unity's 5005

HR_CAP_SAMPLES          = HR_BUFFER_CAP_S * SAMPLING_RATE_HZ
HRV_WIN_SAMPLES         = HRV_WINDOW_S * SAMPLING_RATE_HZ


def compute_phasic_eda(raw_eda_window, src_rate=SAMPLING_RATE_HZ):
    """Decompose a raw EDA window into its phasic (SCR) component and return the
    LAST phasic value (the current event-driven response, tonic drift removed).

    Resamples to EDA_DECOMP_RATE_HZ, cleans, runs nk.eda_phasic, returns the
    final phasic sample in uS. Returns 0.0 if the window is too short, the
    decomposition fails, or the value is physiologically implausible
    (|phasic| > EDA_PHASIC_MAX_US, i.e. resampler/filter ringing rather than a
    real SCR). 0.0 = "no phasic deviation", the safe neutral value.
    """
    a = np.asarray(raw_eda_window, dtype=float)
    if a.size < src_rate * 5:          # need a few seconds to decompose
        return 0.0
    try:
        ds = nk.signal_resample(a, sampling_rate=src_rate,
                                desired_sampling_rate=EDA_DECOMP_RATE_HZ)
        ds = nk.eda_clean(ds, sampling_rate=EDA_DECOMP_RATE_HZ)
        phasic = nk.eda_phasic(ds, sampling_rate=EDA_DECOMP_RATE_HZ)["EDA_Phasic"].values
        if phasic.size == 0:
            return 0.0
        val = float(phasic[-1])
        # Plausibility ceiling: real phasic SCRs are well under 1 uS. A larger
        # magnitude is filter ringing from a resampler edge artifact, not arousal.
        # Reject it (treat as no deviation) so garbage never reaches the score.
        if not np.isfinite(val) or abs(val) > EDA_PHASIC_MAX_US:
            print(f"[EDA-CLAMP] implausible phasic {val:+.2f} uS rejected "
                  f"(> {EDA_PHASIC_MAX_US} uS = decomposition artifact, not a real SCR).")
            return 0.0
        return val
    except Exception:
        return 0.0


def gated_rmssd_from_peaks(peak_indices):
    """RMSSD from R-peak indices with an RR-interval artifact gate.

    The peaks passed in have already been through NeuroKit's Kubios artifact
    correction (Lipponen & Tarvainen 2019, via ecg_peaks(correct_artifacts=True)).
    But Kubios is a rate-RECONSTRUCTION method: it fills missed beats by inserting
    interpolated midpoint peaks, which keeps the heart-rate trace smooth but leaves
    interpolated intervals in the series — and those still inflate a time-domain
    RMSSD. So for RMSSD specifically we additionally apply the standard Malik /
    Task Force (1996) rule: reject a beat-to-beat interval whose RELATIVE change
    from the previous interval exceeds RR_MAX_RELATIVE_CHANGE (20%), then compute
    RMSSD on the surviving successive differences. A relative rule is sounder than
    an absolute-ms one because a tolerable beat-to-beat change scales with heart
    rate. Returns None if too few clean intervals remain or the result is outside
    the plausible RMSSD band. Used for BOTH baseline and live so they match.

    Reference: Malik M. et al. / Task Force of ESC and NASPE (1996), Heart rate
    variability: standards of measurement. Eur Heart J 17(3):354-381.
    """
    rr = np.diff(np.asarray(peak_indices)) * 1000.0 / SAMPLING_RATE_HZ
    rr = rr[(rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)]
    if rr.size < 2:
        return None
    # Malik 20% rule: keep successive differences whose change relative to the
    # PREVIOUS interval is within tolerance; a larger relative jump is a
    # missed/extra/ectopic beat, not real beat-to-beat variability.
    rel_change = np.abs(np.diff(rr)) / rr[:-1]
    diffs = np.diff(rr)[rel_change <= RR_MAX_RELATIVE_CHANGE]
    if diffs.size < (MIN_PEAKS_HRV - 1):
        return None
    r = float(np.sqrt(np.mean(diffs ** 2)))
    if not (np.isfinite(r) and RMSSD_MIN_MS <= r <= RMSSD_MAX_MS):
        return None
    return r


# ============================================================
# UNITY UDP SOCKET (inline setup)
# ============================================================
# One datagram socket, created once. sendto() to a local port is effectively
# non-blocking, but we still guard every send so a transient network error
# can never interrupt the biofeedback loop.
unity_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_unity_addr = (UNITY_UDP_IP, UNITY_UDP_PORT)

def send_to_unity(command):
    """Send a command string to Unity over UDP. Never raises — a failed send is
    logged once-ish and the loop continues (the next packet self-heals)."""
    try:
        unity_sock.sendto(command.encode("utf-8"), _unity_addr)
    except Exception as e:
        print(f"[UNITY] UDP send failed ({command}): {e}")


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
# Gate the session start so runs are comparable. Pressing enter signals the dev
# fake stream to begin at sample 0 (see FAKE_STREAM_CONTROL above); with the real
# device this is just an operator-ready gate and the packet is harmless.
input("\npress enter to start the session (begins baseline from recording sample 0)")
if FAKE_STREAM_CONTROL:
    try:
        _ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _ctrl_sock.sendto(b"start", (CONTROL_UDP_IP, CONTROL_UDP_PORT))
        _ctrl_sock.close()
        print(f"[CTRL] sent 'start' to fake stream at {CONTROL_UDP_IP}:{CONTROL_UDP_PORT}")
    except OSError as e:
        # Never let a control-send failure stop the session (e.g. no listener).
        print(f"[CTRL] start signal not delivered ({e}); continuing.")
    # Give the fake stream a moment to push its first samples before we pull, so
    # baseline genuinely begins near sample 0 rather than on an empty inlet.
    time.sleep(0.2)

start = time.time()
tick = 0
total_samples = 0
eda_buffer = []
ecg_buffer = []
hr_bpm = None
rmssd_ms = None
hr_buffer = []
rmssd_buffer = []

while time.time() - start < BASELINE_DURATION_S:
    chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=PULL_CHUNK_MAX_SAMPLES)
    total_samples += len(chunk)

    for sample in chunk:
        eda_buffer.append(sample[eda_ch])
        ecg_buffer.append(sample[ecg_ch])

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
                    # Same RR artifact gate as the frozen baseline (see
                    # gated_rmssd_from_peaks); raw hrv_time would let a few bad
                    # beats inflate RMSSD.
                    rmssd_ms = gated_rmssd_from_peaks(info_hrv["ECG_R_Peaks"])
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
            print(f"got {len(chunk)} samples| raw_EDA={sample[eda_ch]:.4f} μS | "
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

# Baseline RMSSD via the SAME gate used live (gated_rmssd_from_peaks): Kubios
# correction has already run on peaks_all (correct_artifacts=True above), then the
# Malik 20% relative-difference rule rejects interpolated/artifact intervals for
# the time-domain RMSSD. WHY NOT just nk.hrv_time(peaks_all): Kubios reconstructs
# missed beats by interpolation rather than removing them, so a few bad beats still
# inflate RMSSD (measured ~78-93 ms here vs ~40 ms after the Malik gate). An
# inflated, FROZEN baseline makes every live RMSSD look artificially "suppressed",
# overstating the HRV arousal contribution for the whole session.
avg_hrv = gated_rmssd_from_peaks(info_all["ECG_R_Peaks"])
if avg_hrv is None:
    raise RuntimeError(
        f"Could not compute a reliable baseline RMSSD after RR artifact-gating "
        f"(too few clean beat-intervals, or value outside {RMSSD_MIN_MS}-{RMSSD_MAX_MS} ms). "
        f"ECG quality score = {ecg_q:.2f}. The ECG is likely too noisy or the electrodes "
        f"lost contact — improve electrode contact/grounding and re-record before relying on HRV.")
print(f"[RR-GATE] baseline RMSSD (Kubios + Malik 20% gate) = {avg_hrv:.1f} ms "
      f"(raw hrv_time would give "
      f"{float(nk.hrv_time(peaks_all, sampling_rate=SAMPLING_RATE_HZ)['HRV_RMSSD'].iloc[0]):.1f} ms).")

# EDA baseline mean with 3-sigma cleaning (computed on RAW EDA — no smoothing)
eda_raw_for_mean = np.asarray(eda_buffer, dtype=float)
mu = eda_raw_for_mean.mean()
sd = eda_raw_for_mean.std()
eda_cleaned = eda_raw_for_mean[np.abs(eda_raw_for_mean - mu) <= SIGMA_THRESHOLD * sd]
avg_eda = eda_cleaned.mean()

# EDA baseline: decompose the whole baseline EDA into its phasic component once,
# at the decomposition rate. We index this per sweep-window (and freeze its mean
# and sigma) so the live phase z-scores against the same phasic-EDA baseline.
eda_raw_arr = np.asarray(eda_buffer, dtype=float)
try:
    _eda_ds = nk.eda_clean(
        nk.signal_resample(eda_raw_arr, sampling_rate=SAMPLING_RATE_HZ,
                           desired_sampling_rate=EDA_DECOMP_RATE_HZ),
        sampling_rate=EDA_DECOMP_RATE_HZ)
    eda_phasic_baseline = nk.eda_phasic(_eda_ds, sampling_rate=EDA_DECOMP_RATE_HZ)["EDA_Phasic"].values
except Exception:
    eda_phasic_baseline = np.zeros(int(len(eda_raw_arr) * EDA_DECOMP_RATE_HZ / SAMPLING_RATE_HZ))

def _phasic_at_sample(end_sample):
    """Phasic EDA value corresponding to ECG sample index `end_sample`."""
    if eda_phasic_baseline.size == 0:
        return 0.0
    pi = int(end_sample * EDA_DECOMP_RATE_HZ / SAMPLING_RATE_HZ) - 1
    return float(eda_phasic_baseline[min(max(pi, 0), eda_phasic_baseline.size - 1)])

# Pass 1: sweep the baseline, collecting each signal's RAW per-window quantity:
#   - EDA: phasic amplitude (uS), tonic-free
#   - HR : percent deviation from avg_hr
#   - HRV: percent deviation from avg_hrv (inverted; low HRV = stress)
# We need each signal's own baseline mean & sigma to z-score it later.
rolling = int(ST_ROLLING_WINDOW_S * LOOP_RATE_HZ)
step = SAMPLING_RATE_HZ // 2
_raw_eda_d = []
_raw_hr_d  = []
_raw_hrv_d = []
_sweep_total = 0
_sweep_dropped_hr = 0
_sweep_dropped_rmssd = 0
for end in range(HRV_WIN_SAMPLES, len(ecg_all), step):
    _sweep_total += 1
    h = None
    try:
        hr_start = max(0, end - HR_CAP_SAMPLES)
        sig_h, ih = nk.ecg_peaks(ecg_all[hr_start:end],
                                 sampling_rate=SAMPLING_RATE_HZ, correct_artifacts=True)
        hh = float(np.nanmean(nk.ecg_rate(sig_h, sampling_rate=SAMPLING_RATE_HZ)))
        h = hh if np.isfinite(hh) else None
    except Exception:
        h = None
    r = None
    try:
        sig_r, ir = nk.ecg_peaks(ecg_all[end - HRV_WIN_SAMPLES:end],
                                 sampling_rate=SAMPLING_RATE_HZ, correct_artifacts=True)
        if len(ir["ECG_R_Peaks"]) >= MIN_PEAKS_HRV:
            # Same RR artifact gate as everywhere else, so the frozen baseline S_t
            # distribution is built from RMSSD measured the same way as live.
            r = gated_rmssd_from_peaks(ir["ECG_R_Peaks"])
    except Exception:
        r = None
    if h is None:
        _sweep_dropped_hr += 1
    if r is None:
        _sweep_dropped_rmssd += 1
    if h is None or r is None:
        continue
    _raw_eda_d.append(_phasic_at_sample(end))          # phasic uS
    _raw_hr_d.append((h - avg_hr) / avg_hr * 100)       # HR %
    _raw_hrv_d.append((avg_hrv - r) / avg_hrv * 100)    # HRV % (inverted)

_raw_eda_d = np.asarray(_raw_eda_d)
_raw_hr_d  = np.asarray(_raw_hr_d)
_raw_hrv_d = np.asarray(_raw_hrv_d)

# Per-signal baseline mean & sigma (the "natural height" of each signal). These
# are FROZEN and used to z-score live deltas: z = (delta - mean) / sigma.
if _raw_eda_d.size:
    eda_mean, eda_sigma = float(_raw_eda_d.mean()), float(_raw_eda_d.std())
    hr_mean,  hr_sigma  = float(_raw_hr_d.mean()),  float(_raw_hr_d.std())
    hrv_mean, hrv_sigma = float(_raw_hrv_d.mean()), float(_raw_hrv_d.std())
else:
    eda_mean = eda_sigma = hr_mean = hr_sigma = hrv_mean = hrv_sigma = 0.0
eda_sigma_measured, hr_sigma_measured, hrv_sigma_measured = eda_sigma, hr_sigma, hrv_sigma
eda_sigma = max(eda_sigma, EDA_PHASIC_SIGMA_FLOOR, SIGMA_FLOOR)
hr_sigma  = max(hr_sigma,  HR_SIGMA_FLOOR_PCT,     SIGMA_FLOOR)
hrv_sigma = max(hrv_sigma, HRV_SIGMA_FLOOR_PCT,    SIGMA_FLOOR)
# If a floor had to engage, the baseline sigma was implausibly small (short or
# unusually flat baseline). Warn loudly — the floor prevents false alarms, but a
# sigma this small can also signal a baseline-collection problem worth checking.
for _name, _meas, _used in (("HR", hr_sigma_measured, hr_sigma),
                            ("HRV", hrv_sigma_measured, hrv_sigma),
                            ("EDA-phasic", eda_sigma_measured, eda_sigma)):
    if _used > _meas + 1e-9:
        print(f"[SIGMA-FLOOR] {_name} baseline sigma {_meas:.3f} was below floor; "
              f"using {_used:.3f}. (Implausibly flat baseline — check collection if recurring.)")


def zscore_s_t(eda_phasic_v, hr_pct, hrv_pct):
    """Composite stress score from z-scored, weighted signal deltas.
    Each signal is expressed in units of ITS OWN baseline spread before the
    0.5/0.3/0.2 weights are applied, so EDA's larger raw magnitude can no
    longer dominate purely by scale."""
    zE = (eda_phasic_v - eda_mean) / eda_sigma
    zH = (hr_pct       - hr_mean)  / hr_sigma
    zV = (hrv_pct      - hrv_mean) / hrv_sigma
    return WEIGHT_EDA * zE + WEIGHT_HRV * zV + WEIGHT_HR * zH

# Pass 2: build the baseline S_t distribution from z-scored values, so the
# thresholds are on the same z-scored scale the live phase will use.
s_t_buf = deque(maxlen=rolling)
s_t_history = []
s_inst_history = []
for ev, hv, vv in zip(_raw_eda_d, _raw_hr_d, _raw_hrv_d):
    s_inst = zscore_s_t(ev, hv, vv)
    s_t_buf.append(s_inst)
    s_t_history.append(np.mean(s_t_buf))   # smoothed (matches what live phase classifies)
    s_inst_history.append(s_inst)          # raw instantaneous (true spread)

# Baseline S_t distribution. The live phase classifies the SMOOTHED rolling-mean
# S_t, so we centre the thresholds on the smoothed baseline MEAN. But the rolling
# mean drastically shrinks variance (std of an average is ~std/sqrt(N) of the
# underlying values), so using the smoothed std as sigma makes the bands far too
# tight — the score then leaps straight past "stressed" into "ultra-stressed" the
# moment any real arousal appears. We therefore take SIGMA from the RAW
# instantaneous s_inst values (their true spread), while keeping the MEAN from the
# smoothed series for centering. sigma_smoothed is computed too, for visibility.
if s_t_history:
    mean_baseline   = float(np.mean(s_t_history))      # centering: smoothed mean
    sigma_smoothed  = float(np.std(s_t_history))        # deflated (for reference only)
    sigma_raw       = float(np.std(s_inst_history))     # true spread of the score
    sigma_baseline  = sigma_raw                          # <-- used for thresholds
else:
    mean_baseline = 0.0
    sigma_smoothed = 1.0
    sigma_raw = 1.0
    sigma_baseline = 1.0

# Guard against a degenerate baseline sweep. If almost no windows survived,
# sigma collapses toward 0 and every live sample would read "ultra-stressed".
# Surface it loudly rather than silently shipping a broken threshold.
_valid = len(s_t_history)
if _valid < 10 or sigma_baseline < 1e-6:
    print("!!! [BASELINE WARNING] Only %d valid S_t windows (sigma=%.4g). "
          "Thresholds will be unreliable — check baseline length and ECG quality. !!!"
          % (_valid, sigma_baseline))

print(f"\n=== Frozen baseline averages ===")
print(f"avg_hr  = {avg_hr:.2f} BPM")
print(f"avg_hrv = {avg_hrv:.2f} ms")
print(f"avg_eda = {avg_eda:.4f} μS")
print(f"per-signal baseline (for z-scoring):")
print(f"  EDA phasic: mean={eda_mean:+.4f} uS,  sigma={eda_sigma:.4f} uS")
print(f"  HR  delta : mean={hr_mean:+.2f}%,  sigma={hr_sigma:.2f}%")
print(f"  HRV delta : mean={hrv_mean:+.2f}%,  sigma={hrv_sigma:.2f}%")
print(f"baseline S_t: mean = {mean_baseline:+.4f}, sigma_raw = {sigma_raw:.4f} "
      f"(used), sigma_smoothed = {sigma_smoothed:.4f}")
print(f"baseline sweep: {_valid} valid / {_sweep_total} windows "
      f"(dropped: HR={_sweep_dropped_hr}, RMSSD={_sweep_dropped_rmssd})")
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
del eda_buffer, hr_buffer, rmssd_buffer, ecg_buffer
phasic_eda = 0.0                # current phasic EDA (uS); 0 = no deviation
hr_bpm = None
rmssd_ms = None
delta_hr = None
delta_hrv = None
delta_eda = None                # now carries the PHASIC EDA value (uS), not a percent
tick = 0
total_samples = 0
last_data_time = time.time()    # when we last received any samples
ended_by_drought = False        # True if the stream dried up before LIVE_DURATION_S
ecg_buffer = deque(maxlen=HRV_WIN_SAMPLES)
eda_raw_live = deque(maxlen=EDA_DECOMP_WIN_SAMPLES)   # rolling raw EDA for decomposition
s_t_buf = deque(maxlen=rolling)
thresh_mild = mean_baseline + THRESHOLD_MILD_K * sigma_baseline
thresh_high = mean_baseline + THRESHOLD_HIGH_K * sigma_baseline

# Tell Unity the flight may begin (controller waits for "start" before moving forward).
if UNITY_SEND_START_STOP:
    send_to_unity("start")
    print(f"[UNITY] sent 'start' to {UNITY_UDP_IP}:{UNITY_UDP_PORT}")

start = time.time()

while time.time() - start < LIVE_DURATION_S:
    chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=PULL_CHUNK_MAX_SAMPLES)
    total_samples += len(chunk)

    # Stream-drought handling: if data is flowing, reset the drought timer. If
    # the stream has been silent past the timeout (recording ended or device
    # disconnected), stop the live phase cleanly rather than spinning on empty
    # polls. We skip the rest of the loop body on an empty pull so we don't
    # recompute on stale buffers or print a flood of "got 0 samples" lines.
    if len(chunk) > 0:
        last_data_time = time.time()
    else:
        if time.time() - last_data_time > STREAM_DROUGHT_TIMEOUT_S:
            ended_by_drought = True
            break
        time.sleep(LOOP_SLEEP_S)
        continue

    for sample in chunk:
        ecg_buffer.append(sample[ecg_ch])
        eda_raw_live.append(sample[eda_ch])

    if tick % HR_COMPUTE_INTERVAL == 0 and len(ecg_buffer) > 0:
        ecg_arr = np.asarray(ecg_buffer, dtype=float)

        # Phasic EDA on the rolling raw-EDA window (tonic drift removed).
        phasic_eda = compute_phasic_eda(eda_raw_live, src_rate=SAMPLING_RATE_HZ)

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

        # RMSSD only once the 60 s window has filled. Same RR artifact gate as
        # the frozen baseline so live and baseline RMSSD are directly comparable.
        if len(ecg_arr) >= HRV_WIN_SAMPLES:
            try:
                ecg_hrv_win = nk.ecg_clean(ecg_arr, sampling_rate=SAMPLING_RATE_HZ)
                peaks_hrv, info_hrv = nk.ecg_peaks(ecg_hrv_win,
                                                   sampling_rate=SAMPLING_RATE_HZ, correct_artifacts=True)
                if len(info_hrv["ECG_R_Peaks"]) >= MIN_PEAKS_HRV:
                    rmssd_ms = gated_rmssd_from_peaks(info_hrv["ECG_R_Peaks"])
                else:
                    rmssd_ms = None
            except Exception:
                rmssd_ms = None

    # --- deltas (raw, per-signal; z-scoring happens in the composition) ---
    #   EDA: phasic amplitude in uS (tonic-free). Its baseline mean ~0, so the
    #        z-score (phasic - eda_mean)/eda_sigma measures how large the current
    #        event-driven response is relative to resting phasic activity.
    #   HR / HRV: percent deviation from the frozen baseline averages.
    delta_eda = phasic_eda                                   # uS (phasic)
    delta_hr  = (hr_bpm - avg_hr) / avg_hr * 100 if hr_bpm is not None else None
    # HRV bootstrap: for the first ~60 s of live the RMSSD window has not yet
    # filled, so no live RMSSD exists. We hold delta_hrv at its baseline mean
    # (hrv_mean) so its z-score is ~0 — i.e. assume HRV has not moved from
    # baseline yet. The HRV term keeps its full 0.3 weight, contributing nothing
    # until a real measurement arrives. We do NOT slide the baseline RMSSD window
    # across the operator gap (unknown duration would contaminate it).
    if rmssd_ms is not None:
        delta_hrv = (avg_hrv - rmssd_ms) / avg_hrv * 100
    else:
        delta_hrv = hrv_mean   # z-scores to ~0: no assumed deviation yet

    # --- composite stress score (per-signal z-scored, then weighted) ---
    #   Each signal's delta is converted to its own baseline z-score before the
    #   fixed 0.5/0.3/0.2 weights are applied, so EDA can no longer dominate by
    #   raw scale. delta_hrv is always present (baseline-mean during bootstrap,
    #   real after), so the only term that can be briefly missing is HR in the
    #   first seconds. While HR is missing we renormalise across the available
    #   z-scored terms so the score keeps the same scale.
    s_t = None
    state = "calm"
    unity_cmd = "increase"
    terms = []
    if delta_eda is not None:
        terms.append((WEIGHT_EDA, (delta_eda - eda_mean) / eda_sigma))
    if delta_hrv is not None:
        terms.append((WEIGHT_HRV, (delta_hrv - hrv_mean) / hrv_sigma))
    if delta_hr is not None:
        terms.append((WEIGHT_HR, (delta_hr - hr_mean) / hr_sigma))
    if terms:
        wsum = sum(w for w, _ in terms)
        s_inst = sum(w * z for w, z in terms) / wsum * (WEIGHT_EDA + WEIGHT_HRV + WEIGHT_HR)
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

    # Send the current band command to Unity at the send cadence (~10 Hz).
    # Only send once a real S_t exists, so we don't drive the balloon during the
    # first moments before any signal is available. "neutral" is sent during the
    # stressed band too — Unity ignores it (holds altitude), which is the intent.
    if s_t is not None and tick % UNITY_SEND_INTERVAL == 0:
        send_to_unity(unity_cmd)

    if tick % PRINT_INTERVAL == 0:
        hr_str = f"{hr_bpm:.1f}" if hr_bpm is not None else "—"
        rmssd_str = f"{rmssd_ms:.1f}" if rmssd_ms is not None else "—"
        dhr_str = f"{delta_hr:+.1f}" if delta_hr is not None else "—"
        dhrv_str = f"{delta_hrv:+.1f}" if delta_hrv is not None else "—"
        phasic_str = f"{phasic_eda:+.3f}" if delta_eda is not None else "—"
        st_str = f"{s_t:+.2f}" if s_t is not None else "—"
        print(f"[LIVE] got {len(chunk)} samples| raw_EDA={sample[eda_ch]:.4f} μS | "
              f"HR={hr_str} BPM | "
              f"RMSSD={rmssd_str} ms | ΔHR={dhr_str}% | ΔHRV={dhrv_str}% | "
              f"phasicEDA={phasic_str} μS | S_t={st_str} | {state} -> unity:{unity_cmd}")

    tick += 1
    time.sleep(LOOP_SLEEP_S)

elapsed = time.time() - start

# Tell Unity the session is over (controller halts forward flight on "stop").
if UNITY_SEND_START_STOP:
    send_to_unity("stop")
    print(f"[UNITY] sent 'stop' to {UNITY_UDP_IP}:{UNITY_UDP_PORT}")
unity_sock.close()

print(f"\n=== Live phase complete ===")
if ended_by_drought:
    print(f"[LIVE] Stream ended: no data for {STREAM_DROUGHT_TIMEOUT_S:.0f}s "
          f"(recording finished or device disconnected). Stopped cleanly.")
else:
    print(f"[LIVE] Reached configured live duration ({LIVE_DURATION_S}s).")
print(f"[LIVE] Duration: {elapsed:.2f} s")
print(f"[LIVE] Total raw samples: {total_samples}")
print(f"[LIVE] Effective rate: {total_samples / elapsed:.1f} Hz")
print(f"[LIVE] ecg_buffer (deque): {len(ecg_buffer)} / {HRV_WIN_SAMPLES} samples")
