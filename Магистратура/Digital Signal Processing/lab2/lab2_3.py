import numpy as np
import matplotlib.pyplot as plt


# x_1 fucntion
def x_1(t,f):
    return(np.exp(2*np.pi*1j*f*t))

# x_2 fucntion
def x_2(t, amplitudes, rates):
    return(sum(
        [amplitudes[i]*\
            np.cos(2*np.pi*rates[i]*t)\
             for i in range(len(amplitudes))]
        ))

# x_3 fucntion
def x_3(t, amplitudes, rates, t_0):
    if t <= t_0:
        A = amplitudes[0]
        f = rates[0]
    else:
        A = amplitudes[1]
        f = rates[1]
    return(A*np.cos(2*np.pi*f*t))


def main():
    #? Sampling rate in Hz
    SR = 5000

    #? Signal length in s
    signal_lenght = 6000

    #? Value of time step
    step = 1000/5000
    N = int(round(signal_lenght/step))


    #* First signal params

    #? x_1_rate - signal frequency
    x_1_rate = 2


    #* Second signal params

    #? x_2_rates - signal frequencies
    x_2_rates = [2,0.2,1]

    #? x_2_amplitudes - signal amplitudes
    x_2_amplitudes = [1,1.4,0.8]


    #* Third signal params

    #? x_3_rates - signal frequencies
    x_3_rates = [2,4]

    #? x_3_amplitudes - signal amplitudes
    x_3_amplitudes = [2,3]

    #? x_3_t_thresh - time threshold (t_0)
    x_3_t_thresh = 2000

    #* Samples generation
    t = 0

    #? First signal list
    x_1_sample =  []

    #? Second signal list
    x_2_sample =  []

    #? Third signal list
    x_3_sample =  []

    for index in range(N):
        
        x_1_sample.append(x_1(t, x_1_rate))
        x_2_sample.append(x_2(t, x_2_amplitudes, x_2_rates))
        x_3_sample.append(x_3(t, x_3_amplitudes, x_3_rates, x_3_t_thresh))

        t += step

    #* Samples to nparray
    x_1_sample = np.array(x_1_sample)
    x_2_sample = np.array(x_2_sample)
    x_3_sample = np.array(x_3_sample)

    #* Spectrum calculating
    x_1_spectrum = np.fft.ifft(x_1_sample)
    x_2_spectrum = np.fft.rfft(x_2_sample)
    x_3_spectrum = np.fft.rfft(x_3_sample)

    #* Build plots
    plt.plot(np.fft.fftfreq(N, 1/SR)/1000, np.abs(x_1_spectrum)/N)
    plt.savefig('imgs/Lab2_3_1.png')
    plt.close()

    plt.plot(np.fft.rfftfreq(N, 1/SR)/1000, np.abs(x_2_spectrum)/N)
    plt.savefig('imgs/Lab2_3_2.png')
    plt.close()

    plt.plot(np.fft.rfftfreq(N, 1/SR)/1000, np.abs(x_3_spectrum)/N)
    plt.savefig('imgs/Lab2_3_3.png')
    plt.close()


if __name__ == '__main__':
    main()