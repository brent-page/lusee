from lusee_sky import *
import sys
load_kernels()
freq_start, freq_end, freq_step, NS, EW = map(float, sys.argv[1:])
outfile = f"wfall_{freq_start}_{freq_end}_{freq_step}_{NS}_{EW}.npz"
print (f"Saving to {outfile}.")
freq=np.arange(freq_start,freq_end,freq_step)
nights = get_lunar_nights(2024)

wfalls = []
for i,n in enumerate(nights):
    print (f"Doing night {i}")
    wfalls.append(time_freq_K(n,freq,NS_20MHz_beam_stdev_degr=np.sin(NS * np.pi / 180),EW_20MHz_beam_stdev_degr=np.sin(EW * np.pi / 180),
                              verbose=True))
np.savez(outfile,wfalls=wfalls)
