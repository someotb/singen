import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import firwin, lfilter

FS = 1000
UP_KOEF = 3
DURATION = 1


def sin_gen(freq: int, duration: int):
    fs_step = 1 / FS
    fs_list = np.arange(0, duration - fs_step, fs_step)
    sin = np.sin(2 * np.pi * freq * fs_list + 1)
    return sin, fs_list


def interpolation(signal, up_koef: int):
    signal_up = np.zeros(len(signal) * up_koef)
    signal_up[::up_koef] = signal
    numtaps = 4 * up_koef + 1
    h = firwin(numtaps, cutoff=1.0 / up_koef, window="hamming")
    filtered = lfilter(h * up_koef, 1.0, signal_up)
    return filtered


sin, t = sin_gen(40, DURATION)
sin_up = interpolation(sin, UP_KOEF)
t_up = np.arange(0, DURATION, 1 / len(sin_up))

fft_sin = np.abs(np.fft.fft(sin)) / len(sin)
fft_freq = np.fft.fftfreq(len(sin), 1 / FS)
fft_freq_sh = np.fft.fftshift(fft_freq)
fft_log = np.fft.fftshift(20 * np.log10(fft_sin))

fft_sin_up = np.abs(np.fft.fft(sin_up)) / len(sin_up)
fft_freq_up = np.fft.fftfreq(len(sin_up), 1 / (FS * UP_KOEF))
fft_freq_sh_up = np.fft.fftshift(fft_freq_up)
fft_log_up = np.fft.fftshift(20 * np.log10(fft_sin_up))


plt.figure()
plt.plot(t, sin)

plt.figure()
plt.plot(sin_up)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(fft_freq_sh, fft_log)
plt.subplot(2, 1, 2)
plt.plot(fft_freq_sh_up, fft_log_up)

plt.show()
