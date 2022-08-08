from scipy.fft import fft, ifft, rfft, irfft, rfftfreq
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as save_wav
from scipy.io.wavfile import read as read_wav

def main():
    root_folder = 'data/{}'

    rate, data = read_wav(root_folder.format('tune.wav'))    

    #* Built spectrum
    spectrum = rfft(data)

    #* Get freqs
    freqs = rfftfreq(len(data), 1/rate)

    
    plt.plot(freqs, spectrum)
    plt.savefig('imgs/lab2_5.png')

    #* Treshold value
    threshold = 3000
    for index in range(len(freqs)):
        if freqs[index] >= threshold:
            spectrum[index] = 0
    
    filtred_signal =irfft(spectrum).astype('int16')


    save_wav(root_folder.format(f'filtred_tune_{threshold}.wav'),rate, filtred_signal)




if __name__ == '__main__':
    main()