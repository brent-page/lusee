import os
os.environ['ARES']='/home/anze/work/BMX/ares'
import ares
import numpy as np

sim = ares.simulations.Global21cm()
sim.run()
nu = 1420/(1+sim.history['z'])
Tsig = sim.history['dTb']/1000 # We work in K
np.savez('waterfalls/signal.npz',nu=nu, Tsig=Tsig)


