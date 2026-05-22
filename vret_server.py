from pylsl import StreamInlet, resolve_streams
import time
import neurokit2 as nk
import numpy as np

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
    
    eda_alpha = 0.005
    eda_smoothed = None

    hr_bpm=None
    rmssd_ms=None

    while time.time()-start <20: #20 seconds is just the duration of testing the code. after trial it gets removed. it is so we won't wait forever for the tests to end.
        chunk,timestamps=inlet.pull_chunk(timeout=0.0, max_samples=1000)
        total_samples = total_samples + len(chunk)
        for sample in chunk:
            eda_buffer.append(sample[eda_ch])
            ecg_buffer.append(sample[ecg_ch])
            if eda_smoothed is None:
                eda_smoothed=sample[eda_ch]
            else:
                eda_smoothed=eda_alpha*sample[eda_ch]+(1-eda_alpha)*eda_smoothed
        if tick%10==0:
            if len(ecg_buffer)>=5000:
                window=np.array(ecg_buffer[-10000:])
                peaks,info=nk.ecg_peaks(window,sampling_rate=1000) #peaks we don't need. info is a dictionary with keys.one of them is ECG_R_Peaks.
                r_peak_indices=info['ECG_R_Peaks'] 
                rr_intervals_ms=np.diff(r_peak_indices)
                valid_rr = (rr_intervals_ms >= 300) & (rr_intervals_ms <= 1500)
                rr_clean = rr_intervals_ms[valid_rr]
               
                if len(rr_clean) > 0:
                    hr_bpm = 60000 / rr_clean[-1]
                else:
                    hr_bpm = None
                #HRV    
                if len(rr_clean)<2:
                    rmssd_ms=None
                else:
                    rr_diffs=np.diff(rr_clean[-60:]) #for hrv
                    squared=rr_diffs**2
                    mean_sq=np.mean(squared)
                    rmssd_ms=np.sqrt(mean_sq)
 
        
        if tick%5==0:
            if len(chunk)==0:
                print("got 0 samples")
            else:
                hr_str = f"{hr_bpm:.1f}" if hr_bpm is not None else "—"
                rmssd_str = f"{rmssd_ms:.1f}" if rmssd_ms is not None else "—"
                print(f"got {len(chunk)} samples| raw_EDA={sample[eda_ch]:.4f} | smoothed EDA={eda_smoothed:.4f} μS | HR={hr_str} BPM | RMSSD={rmssd_str} ms")

        tick=tick+1
        time.sleep(0.02)

    elapsed = time.time() - start
    print(f"\nTotal: {total_samples} samples in {elapsed:.2f} s")
    print(f"Effective rate: {total_samples / elapsed:.1f} Hz (expected ~1000 Hz)")
    print(f"the ecg_buffer length is {len(ecg_buffer)}and the last value is {ecg_buffer[-1]}")
    print(f"the eda_buffer length is {len(eda_buffer)}and the last value is {eda_buffer[-1]}")


    

