import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from copy import deepcopy


df = pd.read_csv('../lab2/data/ecg.dat')

F = int()

time =  df['time'].to_numpy()
data = df['value'].to_numpy()
data_clear = d2.copy()

desired = (1,  1,  0,  0,  1,   1)
bands =   (0, 40, 40, 60, 60, 500)

#fir = signal.firwin(101, [49, 51], fs=F)
#fir = signal.firls(101, bands, desired, fs=F)
fir = signal.firwin2(101, [0, 45, 55, 500], [1, 0, 0, 1], fs=F)

lf = signal.lfilter(fir, [1], d2)

plt.plot(d2o)
plt.plot(lf)
plt.show()
