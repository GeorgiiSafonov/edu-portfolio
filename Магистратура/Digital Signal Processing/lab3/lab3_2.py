import numpy as np 
import scipy.fft as fft 
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io.wavfile import write as save_wav
from scipy.io.wavfile import read as read_wav


def main():
    root_folder = 'data/{}'
    image_folder = 'imgs/{}'
    
    #*ECG processing
    frame = pd.read_csv(root_folder.format('ecg.dat'), sep=' ')
    
    duration = frame.iloc[-1]['time'] - frame.iloc[0]['time']

    sample_rate = int(frame.shape[0]/duration)

    #*Spectrum for raw signal
    spectrum = fft.rfft(frame['value'].to_numpy())
    freqs = fft.rfftfreq(frame.shape[0], 1/sample_rate)

    #*Filtration 
    labels  = (1,  1,  0,  0,  1,   1)
    bands =   (0, 40, 40, 60, 60, 500)
    coeffs = signal.firls(121, bands, labels, fs = sample_rate)

    filtred_data = signal.lfilter(coeffs, [1], frame['value'].to_numpy())
    
    new_spectrum =  fft.rfft(filtred_data)
    
    plt.subplot(2,1,1)
    plt.plot(freqs, spectrum)
    plt.subplot(2,1,2)
    plt.plot(freqs, new_spectrum)
    plt.savefig(image_folder.format('lab_3_2_spectrum.png'))
    plt.close()
    
    plt.subplot(2,1,1)
    plt.plot(frame['value'].to_numpy()[15000:20000])
    plt.subplot(2,1,2)
    plt.plot(filtred_data[15000:20000])
    plt.savefig(image_folder.format('lab_3_2_data.png'))
    plt.close()

    #* Tune processing
    rate, data = read_wav(root_folder.format('tune.wav'))    
    
    bands = [0,12000,12000,16000,16000,rate//2]
    labels = [1,1,0,0,1,1]

    coeffs = signal.firls(1101, bands, labels, fs = rate)

    filtred_data = signal.lfilter(coeffs, [1], data).astype('int16')

    save_wav(root_folder.format(f'filtred_tune.wav'),rate, filtred_data)




if __name__ == '__main__':
    main()