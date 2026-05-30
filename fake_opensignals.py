import numpy as np
import time
import json
from pylsl import StreamInfo, StreamOutlet

# ------------------------------------------------------------------
# THE BUG THIS FIXES:
# The OpenSignals .txt header lists the physical column order, e.g.
#     "column": ["nSeq", "DI", "CH1", "CH2"], "sensor": ["ECG", "EDA"]
# meaning data column 2 (CH1) = ECG and column 3 (CH2) = EDA.
# The previous version hard-coded col2->EDA, col3->ECG, which is the
# OPPOSITE of this recording. It broadcast the ECG trace labelled "EDA"
# and the EDA trace labelled "ECG", so the server fed EDA into the R-peak
# detector and HR/RMSSD were garbage.
#
# Fix: read the header, find which physical column is ECG and which is
# EDA by NAME, convert each with its own PLUX transfer function, and push
# them on correctly-labelled LSL channels (ch0=digital, ch1=EDA, ch2=ECG).
# ------------------------------------------------------------------

FILE = r"C:\Users\Shayan\Desktop\NICS\acrophobia\shayan_code\vret_pipeline\14_minute_test_of_myself_2026-05-26_16-47-36.txt"

# ---- parse the JSON device block on the 2nd header line ----
with open(FILE, "r") as f:
    header_lines = [next(f) for _ in range(3)]
meta = json.loads(header_lines[1].lstrip("# ").strip())
dev = next(iter(meta.values()))                 # first (only) device block
sensors = dev["sensor"]                          # e.g. ["ECG", "EDA"]
columns = dev["column"]                          # e.g. ["nSeq","DI","CH1","CH2"]

# The data columns after nSeq/DI correspond, in order, to the sensors list.
# Build sensor_name -> physical column index in the loaded array.
ch_cols = [i for i, c in enumerate(columns) if c not in ("nSeq", "DI")]
sensor_to_col = {name: col for name, col in zip(sensors, ch_cols)}
ECG_COL = sensor_to_col["ECG"]
EDA_COL = sensor_to_col["EDA"]
DI_COL  = columns.index("DI")
print(f"[header] sensors={sensors} -> ECG=col{ECG_COL}, EDA=col{EDA_COL}, DI=col{DI_COL}")

data = np.loadtxt(FILE, skiprows=3)
rows, ncol = data.shape
print(f"[data] rows={rows} cols={ncol} duration={rows/1000:.1f}s")

# PLUX biosignalsplux datasheet transfer functions (16-bit, VCC=3 V):
#   EDA(uS) = (ADC/2^16) * VCC / 0.132
#   ECG(mV) = ((ADC/2^16) - 0.5) * VCC / 1100 * 1000
eda_uS  = (data[:, EDA_COL] / 65536) * 3.0 / 0.132
ecg_mV  = ((data[:, ECG_COL] / 65536) - 0.5) * 3.0 / 1100 * 1000
digital = data[:, DI_COL]

# ---- LSL stream: ch0=digital, ch1=EDA, ch2=ECG, with labels ----
info = StreamInfo(
    name='OpenSignals',
    type='00:07:80:0F:31:9C',
    channel_count=3,
    nominal_srate=1000.0,
    channel_format='float32',
    source_id='OpenSignals'
)
_chns = info.desc().append_child("channels")
for _label, _unit in [("digital", ""), ("EDA", "uS"), ("ECG", "mV")]:
    _c = _chns.append_child("channel")
    _c.append_child_value("label", _label)
    _c.append_child_value("unit", _unit)

outlet = StreamOutlet(info)
print("Broadcasting on LSL (ch0=digital, ch1=EDA, ch2=ECG). Run vret_server.py now.")
while True:
    for i in range(rows):
        outlet.push_sample([digital[i], eda_uS[i], ecg_mV[i]])
        if i % 5000 == 0:
            print(f"[{digital[i]:.0f}, EDA={eda_uS[i]:.3f} uS, ECG={ecg_mV[i]:.3f} mV]")
        time.sleep(0.001)
