from pylsl import StreamInlet, resolve_streams
from collections import deque
import time
import neurokit2 as nk
import numpy as np


# ============================================================
# CONFIGURATION — change values here for different uses of our biofeedback
# ============================================================

# --- Hardware / acquisition ---
SAMPLING_RATE_HZ        = 1000              # PLUX hub setting; LSL stream rate; the default on our tool
LOOP_RATE_HZ            = 50                # how often the Python loop ticks
LOOP_SLEEP_S            = 1.0 / LOOP_RATE_HZ # derived: 0.02 s
PULL_CHUNK_MAX_SAMPLES  = 1000              # safety cap per pull (~1 s of buffered data at 1 kHz)

# --- Phase durations ---
BASELINE_DURATION_S     = 120               # bump to 120 for real sessions(for testing just change it to smaller amounts)
LIVE_DURATION_S         = 60               # 1 minute long live phase

# --- EDA smoothing (EMA) ---
EDA_ALPHA               = 0.005             # smaller = heavier smoothing; ~200-sample time constant

# --- ECG / HR / RMSSD ---
ECG_MIN_SAMPLES_FOR_HR  = 5000              # don't try R-peak detection until buffer has this many(it won't be scientific)
ECG_WINDOW_SAMPLES      = 10000             # R-peak detection window (~10 s at 1 kHz)
HR_COMPUTE_INTERVAL     = 10                # compute HR every N ticks (10 → 5 Hz at 50 Hz loop)
PRINT_INTERVAL          = 5                 # print every N ticks (5 → 10 Hz at 50 Hz loop)
RR_MIN_MS               = 300               # filter: ignore RR shorter than this (>200 BPM)
RR_MAX_MS               = 1500              # filter: ignore RR longer than this (<40 BPM)
RMSSD_WINDOW_BEATS      = 60                # how many recent RR intervals to use for RMSSD

# --- Baseline cleaning ---
SIGMA_THRESHOLD         = 3                 # 3σ rule for outlier removal

# --- Live-phase math (Stage 4, defined now for completeness) ---
WEIGHT_EDA              = 0.5               # weight of ΔEDA in S_instant
WEIGHT_HRV              = 0.3               # weight of ΔHRV in S_instant
WEIGHT_HR               = 0.2               # weight of ΔHR in S_instant
ST_ROLLING_WINDOW_S     = 1.0               # rolling-mean window for S_t (in seconds)
THRESHOLD_MILD_K        = 1.33              # thresh_mild = k × σ_baseline
THRESHOLD_HIGH_K        = 2.28              # thresh_high = k × σ_baseline

# --- Balloon mapping (Stage 4) ---
BALLOON_Y_LOW           = 0                 # minimum balloon altitude (units TBD)
BALLOON_Y_HIGH          = 100               # maximum balloon altitude

# ============================================================




def connect_to_opensignals():

    print("[LSL] is searching for opensignals stream...")

    for j in range(30):
        streams=resolve_streams(wait_time=2.0)#lsl is not instant and finding it requires time.

        if streams==[]:

            if j!=29:
                print(f"attempt{j+1}==>no streams found retrying\n")
            else:
                raise RuntimeError("No OpenSignals stream found after 30 attempts")

        else:
            numberofstreams=len(streams)
            opensignalstream=None

            for i in range(numberofstreams):
                print(
                     f"name={streams[i].name()}|"
                     f"type={streams[i].type()}|"
                     f"channels={streams[i].channel_count()}|"
                     f"rate={streams[i].nominal_srate()}|"
                     f"source={streams[i].source_id()}"
                     )
                if streams[i].channel_count()==2 or streams[i].channel_count()==3:
                    opensignalstream=streams[i] #opensignnalstream is stream info. a description. 
                    break

            if opensignalstream is None:
                    raise RuntimeError("Streams found but none with 2 or 3 channels")
            break
        
    inlet=StreamInlet(opensignalstream) #here we make the connection and read the stream. TCP connection made

    if opensignalstream.channel_count()==2:
        eda_channel=0
        ecg_channel=1
    elif opensignalstream.channel_count()==3:
        eda_channel=1
        ecg_channel=2

    return [inlet,eda_channel,ecg_channel]


if __name__ == "__main__":
    print("Starting VRET biofeedback server...")
    result = connect_to_opensignals()
    inlet=result[0]
    eda_ch=result[1]
    ecg_ch=result[2]
    print(f"Connected. EDA on channel {eda_ch}, ECG on channel {ecg_ch}.")
   
    
    start=time.time()
    tick=0
    total_samples = 0
    eda_buffer=[]
    ecg_buffer=[]
    
    eda_smoothed = None
    hr_bpm=None
    rmssd_ms=None

    eda_smoothed_buffer=[]
    hr_buffer=[]
    rmssd_buffer=[]

    while time.time()-start <BASELINE_DURATION_S: 
        chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=PULL_CHUNK_MAX_SAMPLES)
        total_samples = total_samples + len(chunk)

        for sample in chunk:
            eda_buffer.append(sample[eda_ch])
            ecg_buffer.append(sample[ecg_ch])

            if eda_smoothed is None:
                eda_smoothed=sample[eda_ch]
            else:
                eda_smoothed=EDA_ALPHA * sample[eda_ch] + (1 - EDA_ALPHA) * eda_smoothed 
            eda_smoothed_buffer.append(eda_smoothed)

        if tick%HR_COMPUTE_INTERVAL==0:

            if len(ecg_buffer)>=ECG_MIN_SAMPLES_FOR_HR:

                window=np.array(ecg_buffer[-ECG_WINDOW_SAMPLES:])
                peaks,info=nk.ecg_peaks(window,sampling_rate=SAMPLING_RATE_HZ) #peaks we don't need. info is a dictionary with keys.one of them is ECG_R_Peaks.
                r_peak_indices=info['ECG_R_Peaks'] 
                rr_intervals_ms=np.diff(r_peak_indices)
                valid_rr = (rr_intervals_ms >= RR_MIN_MS) & (rr_intervals_ms <=RR_MAX_MS)
                rr_clean = rr_intervals_ms[valid_rr]
               
                if len(rr_clean) > 0:
                    hr_bpm = 60000 / rr_clean[-1]

                else:
                    hr_bpm = None

                #HRV    
                if len(rr_clean)<2:
                    rmssd_ms=None
                else:
                    rr_diffs=np.diff(rr_clean[-RMSSD_WINDOW_BEATS:]) #for hrv
                    squared=rr_diffs**2
                    mean_sq=np.mean(squared)
                    rmssd_ms=np.sqrt(mean_sq)

                if hr_bpm is not None:
                    hr_buffer.append(hr_bpm)
                if rmssd_ms is not None:
                    rmssd_buffer.append(rmssd_ms)
 
        
        if tick%PRINT_INTERVAL==0:

            if len(chunk)==0:
                print("got 0 samples")
            else:
                hr_str = f"{hr_bpm:.1f}" if hr_bpm is not None else "—"
                rmssd_str = f"{rmssd_ms:.1f}" if rmssd_ms is not None else "—"
                print(f"got {len(chunk)} samples| raw_EDA={sample[eda_ch]:.4f} | smoothed EDA={eda_smoothed:.4f} μS | HR={hr_str} BPM | RMSSD={rmssd_str} ms")

        tick=tick+1
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

    def clean_and_mean(buffer):
        arr = np.array(buffer)
        mu, sigma = np.mean(arr), np.std(arr)
        cleaned = arr[np.abs(arr - mu) <= SIGMA_THRESHOLD * sigma] 
        return np.mean(cleaned)

    avg_hr= clean_and_mean(hr_buffer)
    avg_hrv= clean_and_mean(rmssd_buffer)
    avg_eda= clean_and_mean(eda_smoothed_buffer)


    total_ticks = int(BASELINE_DURATION_S * LOOP_RATE_HZ)
    eda_arr = np.array(eda_smoothed_buffer)
    hr_arr = np.array(hr_buffer)
    rmssd_arr = np.array(rmssd_buffer)

    s_instant_history = []
    s_t_buffer = []
    s_t_history = []

    rolling_window_samples = int(ST_ROLLING_WINDOW_S * LOOP_RATE_HZ)
    for n in range(total_ticks):
        eda_idx = int(n * len(eda_arr) / total_ticks)
        hr_idx = int(n * len(hr_arr) / total_ticks)
    
        current_eda = eda_arr[eda_idx]
        current_hr = hr_arr[hr_idx]
        current_rmssd = rmssd_arr[hr_idx]

        delta_eda = (current_eda - avg_eda) / avg_eda * 100
        delta_hr = (current_hr - avg_hr) / avg_hr * 100
        delta_hrv = (avg_hrv - current_rmssd) / avg_hrv * 100

        s_instant = (WEIGHT_EDA * delta_eda + WEIGHT_HRV * delta_hrv + WEIGHT_HR * delta_hr)
        s_instant_history.append(s_instant)

        s_t_buffer.append(s_instant)
        if len(s_t_buffer) > rolling_window_samples:
            s_t_buffer.pop(0)
        s_t = np.mean(s_t_buffer)
        s_t_history.append(s_t)

    sigma_baseline = np.std(s_t_history)


    print(f"\n=== Frozen baseline averages ===")
    print(f"avg_hr  = {avg_hr:.2f} BPM")
    print(f"avg_hrv = {avg_hrv:.2f} ms")
    print(f"avg_eda = {avg_eda:.4f} μS")
    print(f"sigma_baseline = {sigma_baseline:.4f}")


    input("press enter to start the live session")
    total_drained = 0
    while True:
        chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=PULL_CHUNK_MAX_SAMPLES)
        if len(chunk)==0:
            break
        total_drained=len(chunk)+total_drained

    print(f"total amount of {total_drained} samples were drained from the buffers to start live phase fresh")



    # --- State reset for starting the live session ---
    del eda_buffer, eda_smoothed_buffer,hr_buffer, rmssd_buffer,ecg_buffer
    eda_smoothed=None
    hr_bpm=None
    rmssd_ms=None
    tick=0
    total_samples=0
    ecg_buffer = deque(maxlen=ECG_WINDOW_SAMPLES)

    start=time.time()


    #live session start loop
    while time.time()-start<LIVE_DURATION_S:
        chunk,timestamps = inlet.pull_chunk(timeout=0.0, max_samples=PULL_CHUNK_MAX_SAMPLES)
        total_samples=len(chunk)+total_samples
        for sample in chunk:
            ecg_buffer.append(sample[ecg_ch])

            if eda_smoothed is None:
                eda_smoothed=sample[eda_ch]
            else:
                eda_smoothed=EDA_ALPHA * sample[eda_ch] + (1 - EDA_ALPHA) * eda_smoothed 

        
        if tick%HR_COMPUTE_INTERVAL==0:

            if len(ecg_buffer)>=ECG_MIN_SAMPLES_FOR_HR:

                window=np.array(ecg_buffer)
                peaks,info=nk.ecg_peaks(window,sampling_rate=SAMPLING_RATE_HZ) 
                r_peak_indices=info['ECG_R_Peaks'] 
                rr_intervals_ms=np.diff(r_peak_indices)
                valid_rr = (rr_intervals_ms >= RR_MIN_MS) & (rr_intervals_ms <=RR_MAX_MS)
                rr_clean = rr_intervals_ms[valid_rr]
                   
                if len(rr_clean) > 0:
                    hr_bpm = 60000 / rr_clean[-1]

                else:
                    hr_bpm = None

                #HRV    
                if len(rr_clean)<2:
                    rmssd_ms=None
                else:
                    rr_diffs=np.diff(rr_clean[-RMSSD_WINDOW_BEATS:]) #for hrv
                    squared=rr_diffs**2
                    mean_sq=np.mean(squared)
                    rmssd_ms=np.sqrt(mean_sq)

        if tick%PRINT_INTERVAL==0:
            if len(chunk)==0:
                print("[LIVE] got 0 samples")
            else:
                hr_str = f"{hr_bpm:.1f}" if hr_bpm is not None else "—"
                rmssd_str = f"{rmssd_ms:.1f}" if rmssd_ms is not None else "—"
                print(f"[LIVE] got {len(chunk)} samples| raw_EDA={sample[eda_ch]:.4f} | smoothed EDA={eda_smoothed:.4f} μS | HR={hr_str} BPM | RMSSD={rmssd_str} ms")

            
        tick=tick+1
        time.sleep(LOOP_SLEEP_S)

    elapsed = time.time() - start
    print(f"\n=== Live phase complete ===")
    print(f"[LIVE] Duration: {elapsed:.2f} s")
    print(f"[LIVE] Total raw samples: {total_samples}")
    print(f"[LIVE] Effective rate: {total_samples / elapsed:.1f} Hz")
    print(f"[LIVE] ecg_buffer (deque): {len(ecg_buffer)} / {ECG_WINDOW_SAMPLES} samples")

