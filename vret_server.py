from pylsl import StreamInlet, resolve_streams
import time


def connect_to_opensignals():
    print("[LSL] is searching for opensignals stream...")
    for j in range(30):
        streams=resolve_streams(wait_time=2.0)
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
                    opensignalstream=streams[i]
                    break
            if opensignalstream is None:
                    raise RuntimeError("Streams found but none with 2 or 3 channels")
            break
        
    inlet=StreamInlet(opensignalstream)
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

    while time.time()-start <20:
        chunk,timestamps=inlet.pull_chunk(timeout=0.0, max_samples=1000)
        total_samples = total_samples + len(chunk)
        for sample in chunk:
            eda_buffer.append(sample[eda_ch])
            ecg_buffer.append(sample[ecg_ch])

        if tick%5==0:
            if len(chunk)==0:
                print("got 0 samples")
            else:
                print(f"got {len(chunk)} samples. first sample:{chunk[0]}")
        tick=tick+1
        time.sleep(0.02)

    elapsed = time.time() - start
    print(f"\nTotal: {total_samples} samples in {elapsed:.2f} s")
    print(f"Effective rate: {total_samples / elapsed:.1f} Hz (expected ~1000 Hz)")
    print(f"the ecg_buffer length is {len(ecg_buffer)}and the last value is {ecg_buffer[-1]}")
    print(f"the eda_buffer length is {len(eda_buffer)}and the last value is {eda_buffer[-1]}")




