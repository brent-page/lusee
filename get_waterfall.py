import sys
sys.path.insert(0,'../pygdsm')
from lusee_sky import *
print ("Using pygdsm from :",pygdsm.__path__)
load_kernels()
try:
    freq_start, freq_end, freq_step, NS, EW = map(float, sys.argv[1:])
except:
    print ("Command line: ./get_waterfall.py start_freq, end_Freq, freq_step, NS beamwdith, EW beamwidth")
    sys.exit(1)
outfile = f"wfall_{freq_start}_{freq_end}_{freq_step}_{NS}_{EW}.npz"
print (f"Saving to {outfile}.")
freq=np.arange(freq_start,freq_end,freq_step)
nights = get_lunar_nights(2024)
wfalls = []
for i,n in enumerate(nights):
    print (f"Doing night {i}")
    wfalls.append(time_freq_K(n,freq,NS_20MHz_beam_stdev_degr = NS,EW_20MHz_beam_stdev_degr=EW,
                              verbose=True))
np.savez(outfile,wfalls=wfalls)
