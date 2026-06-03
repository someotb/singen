import numpy as np
from matplotlib import pyplot as plt

FS = 1000
UP_KOEF = 3


def sin_gen(freq: int, duration: int):
    fs_step = 1 / FS
    fs_list = np.arange(0, duration - fs_step, fs_step)
    sin = np.sin(2 * np.pi * freq * fs_list + 1)
    return sin, fs_list


def interpolation(signal, up_koef: int):
    signal_up = np.zeros(len(signal) * up_koef)
    for i in range(len(signal_up)):
        if i % up_koef == 0:
            signal_up[i] = signal[i // up_koef]
        else:
            signal_up[i] = 0
    return signal_up


sin, t = sin_gen(40, 1)
sin_up = interpolation(sin, UP_KOEF)

fft_sin = np.abs(np.fft.fft(sin))
fft_freq = np.fft.fftfreq(len(sin), 1 / FS)

fft_sin_up = np.abs(np.fft.fft(sin_up))
fft_freq_up = np.fft.fftfreq(len(sin_up), 1 / (FS * UP_KOEF))

plt.figure()
plt.plot(t, sin)

plt.figure()
plt.plot(sin_up)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(fft_freq, fft_sin)
plt.subplot(2, 1, 2)
plt.plot(fft_freq_up, fft_sin_up)

plt.show()
