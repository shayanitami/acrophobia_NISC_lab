import numpy as np
import time
from pylsl import StreamInfo,StreamOutlet

info = StreamInfo(
            name='OpenSignals',
            type='00:07:80:0F:31:9C',   # matches real OpenSignals (uses MAC as type)
            channel_count=3,
            nominal_srate=1000.0,
            channel_format='float32',
            source_id='OpenSignals'
            )

data=np.loadtxt(r"C:\Users\Shayan\Desktop\NICS\acrophobia\shayan_code\vret_pipeline\paria_onmylaptop_opensignals_2026-05-25_15-13-39.txt",skiprows=3)
outlet = StreamOutlet(info)
rows,columns=data.shape
print(rows)
print(columns)
eda_uS=(data[:,2]/65536)* 3.0 / 0.132
ecg_mV=((data[:,3]/ 65536) - 0.5) * 3.0 / 1100 * 1000
digital = data[:, 1]
print("Broadcasting on LSL. Run vret_server.py now.")
while True:
    for i in range(rows):
        outlet.push_sample([digital[i],eda_uS[i], ecg_mV[i]])
        if i%5000==0:
            print(f"[{digital[i]},{eda_uS[i]},{ecg_mV[i]}]")
        time.sleep(0.001)

