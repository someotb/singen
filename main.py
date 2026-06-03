import numpy as np
from matplotlib import pyplot as plt


def sin_gen(freq: int, duration: int):
    fs = 1001
    fs_step = 1 / fs
    fs_list = np.arange(0, duration, fs_step)
    sin = np.sin(2 * np.pi * freq * fs_list)
    return sin, fs_list


def interpolation(signal, up_koef: int):
    signal_up = np.zeros(len(signal) * up_koef)
    for i in range(len(signal_up)):
        if i % up_koef == 0:
            signal_up[i] = signal[i // up_koef]
        else:
            signal_up[i] = 0
    return signal_up


sin, t = sin_gen(500, 1)
sin_up = interpolation(sin, 3)

plt.figure()
plt.plot(t, sin)

plt.figure()
plt.plot(sin_up)

plt.show()
