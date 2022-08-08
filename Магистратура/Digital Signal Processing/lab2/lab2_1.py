import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal

def restored_signal(sample_values, T_samp, t):
    return(sum(sample_values*np.array([np.sinc((t-T_samp*i)/T_samp) for i in range(len(samp_values))])))

signal_time = 60000

omega = lambda f: (2*np.pi)*f


T = 4
F = 1/T
time = [0]

step = 0.001

for i in range(0,signal_time):
    time.append(time[-1]+step)

values = np.array([np.sin(omega(F)*t) for t in time])

sampling_T = 1

sampling_time_line = [0]

while sampling_time_line[-1] <= max(time):
    sampling_time_line.append(sampling_time_line[-1]+sampling_T)

samp_values = np.array([np.sin(omega(F)*t) for t in sampling_time_line])

restored_values = np.array([restored_signal(samp_values, sampling_T, t) for t in time])

plt.plot(time, values, alpha = 0.5)
plt.plot(time, restored_values, alpha = 0.5)
plt.savefig('imgs/Lab_2_1.png')
