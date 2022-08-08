import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.printoptions(precision=10)


def main():
    #Load dataframe
    frame = pd.read_csv('data/Vibration_data.csv').sort_values('second').reset_index()
    frame['second'] = frame['second'] - frame['second'].min()

    #* Sample size
    N = frame.shape[0]

    #* Sampling rate calculating
    SR = N/frame['second'].max()
    print ('Sampling rate (Hz): {0:.10f}'.format(SR))
    
    #* Spectrum calculating
    spectrum = np.fft.rfft(frame['value'].to_numpy())


    #* Build plots
    fig, (ax1, ax2) = plt.subplots(2)
    cut_frame = frame.query('second <= 1')
    ax1.plot(cut_frame['second'], cut_frame['value'])
    ax2.plot(np.fft.fftfreq(N, 1/SR)[:1000], (np.abs((spectrum))/N)[:1000], alpha = 0.5)
    plt.savefig('imgs/Lab2_4.png')
    plt.close()


if __name__  == '__main__':
    main()
