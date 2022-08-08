import pandas as pd
from scipy.fft import fft, ifft, rfft, irfft, rfftfreq
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as save_wav
from scipy.io.wavfile import read as read_wav


def main():
    root_folder = 'data/{}'
    image_folder = 'imgs/{}'

    frame = pd.read_csv(root_folder.format('ecg.dat'), sep=' ')
    
    duration = frame.iloc[-1]['time'] - frame.iloc[0]['time']

    sample_rate = int(frame.shape[0]/duration)
    print(sample_rate)
    spectrum = rfft(frame['value'].to_numpy())
    freqs = rfftfreq(frame.shape[0], 1/sample_rate)

    for index in range(len(spectrum)):
        if freqs[index] > 49.5 and freqs[index] <50.5:
            spectrum[index]  = 0
    
    filtred_signal = irfft(spectrum)

    fig, (ax1, ax2) = plt.subplots(2)

    cutter_top  = 20000
    cutter_top = cutter_top if cutter_top <= frame.shape[0] else  frame.shape[0]

    cutter_buttom = 15000
    cutter_bottom = cutter_buttom if cutter_buttom < cutter_top else 0 

    plt.subplot(2, 1, 1)
    plt.plot(frame['value'].to_numpy()[cutter_bottom:cutter_top])

    plt.subplot(2, 1, 2)
    plt.plot(filtred_signal[cutter_bottom:cutter_top])

    plt.savefig(image_folder.format('Lab2_6.png'))
    plt.close()



if __name__ == '__main__':
    main()


