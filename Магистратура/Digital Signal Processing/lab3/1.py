import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

fs = 8000.0  # Hz
# desired = (0, 0, 1, 1, 0, 0)
# bands = (0, 1, 2, 3, 4, 5)
desired = (0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0)
bands = (0, 50, 50, 150, 150, 350, 350, 750, 750, 900, 900, 1500, 1500, 4000)

bi = 0
fir = signal.firls(101, bands, desired, fs=fs)

hs = list()

freq, response = signal.freqz(fir)
plt.semilogy(0.5 * fs * freq / np.pi, np.abs(response))

for band, gains in zip(zip(bands[::2], bands[1::2]),
                       zip(desired[::2], desired[1::2])):
    plt.semilogy(band, np.maximum(gains, 1e-7), 'k--', linewidth=2)


plt.grid(True)
plt.ylabel('Magnitude')
plt.xlabel('Frequency')

plt.tight_layout()
plt.show()
