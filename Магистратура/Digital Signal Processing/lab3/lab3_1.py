import numpy as np 
import scipy.fft as fft 
from scipy import signal
import matplotlib.pyplot as plt




def main():
    img_root = 'imgs/{}'
    #Sample rate
    sr = 8000

    #*Signal generation
    #? accepted only 50–150 Hz; 350–750 Hz; 900–1500 Hz
    a = [10, 5, 10, 5,20, 17] #amplitudes
    f = [100, 1200, 600,800, 1000, 3000] #freqs

    #*Split timeline by sampling rate
    time_line = np.linspace(0,1,sr)
    
    #*Generate signal values
    values =  [np.sum([a_ * np.cos(2 * np.pi *  f_ * t) for a_, f_ in zip(a,f)])  for t in time_line] 
    
    plt.figure(figsize=(300,10))
    plt.plot(time_line, values)
    plt.savefig(img_root.format('3_1_signal_plot.png'))
    plt.close()

    #* Built spectrum
    spectrum = abs(fft.fftshift(fft.fft(values)))
    freqs = fft.fftshift(fft.fftfreq(len(values), 1/sr))

    plt.figure(figsize=None)
    plt.plot(freqs, spectrum)
    plt.savefig(img_root.format('3_1_spectrum_plot.png'))
    plt.close()


    #* Number of coefficients
    N = 351

    #* axis of symmetry
    M = (N - 1) // 2
    
    #*List of angular frequencies
    w_s = np.linspace(0,  np.pi, M)

    #* Create matrix
    F = np.array([[2 * np.cos(w_s[j] * (M - i)) if i < M else 1 for i in range(M + 1)] for j in range(M)])

    #* Labels for 0->2pi interval (0-drop, 1-append)
    interv_labels = [1 if (w >= 2 * np.pi * 50 / sr and w <= 2 * np.pi * 150 / sr)
      or (w >= 2 * np.pi * 350 / sr and w <= 2 * np.pi * 750 / sr)
      or (w >= 2 * np.pi * 900 / sr and w <= 2 * np.pi * 1500 / sr) else 0 for w in w_s]
    

    #* LS-method

    h_ = list(np.linalg.pinv(F)  @ interv_labels)

    h_ += h_[:-1][::-1]
    

    #* Convolution (signal + filter)
    filtred_values = signal.lfilter(h_, [1], values)

    #* New spectrum
    filtred_spectrum = abs(fft.fftshift(fft.fft(filtred_values)))

    plt.subplot(2,1,1)
    plt.plot(freqs, spectrum)
    plt.subplot(2,1,2)
    plt.plot(freqs, filtred_spectrum)
    plt.savefig(img_root.format('lab_3_1_filtred.png'))
    plt.close()

    freqs_, response = signal.freqz(h_, fs = 8000) 

    plt.plot(freqs_, abs(response ))
    plt.savefig(img_root.format('Lab_3_1_freqz (manual).png'))
    plt.close()
    #* SCIPY METHOD
    #? accepted only 50–150 Hz; 350–750 Hz; 900–1500 Hz
    freqs_intervals = (0,50,50,150,150,350,350,750,750,900,900,1500,1500,4000)
    labels = (0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0)

    coeffs = signal.firls(111, freqs_intervals, labels, fs = sr)
    
    filtred_values = signal.lfilter(coeffs, [1], values)

    #*Spectrum for filtred data
    filtred_spectrum_scipy = abs(fft.fftshift(fft.fft(filtred_values)))

    plt.subplot(2,1,1)
    plt.plot(freqs, filtred_spectrum)
    plt.subplot(2,1,2)
    plt.plot(freqs, filtred_spectrum_scipy)
    plt.savefig(img_root.format('lab_3_1_filtred_scipy.png'))
    plt.close()

    freqs_, response = signal.freqz(coeffs, fs = 8000) 
    plt.plot(freqs_, abs(response) )
    plt.savefig(img_root.format('Lab_3_1_freqz (scipy).png'))
    plt.close()

    



if __name__ == '__main__':
    main()