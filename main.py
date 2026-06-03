import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import firwin, lfilter

FS = 1000
F = 20
UP_KOEF = 3
DEC_KOEF = 3
DURATION = 1


def sin_gen(freq: int, duration: int):
    fs_step = 1 / FS
    fs_list = np.arange(0, duration - fs_step, fs_step)
    sin = np.sin(2 * np.pi * freq * fs_list + 1)
    return sin, fs_list


def interpolation(signal, up_koef: int):
    signal_up = np.zeros(len(signal) * up_koef)
    signal_up[::up_koef] = signal
    numtaps = 8 * up_koef + 1
    h = firwin(numtaps, cutoff=1.0 / up_koef, window="hamming")
    filtered = lfilter(h * up_koef, 1.0, signal_up)
    return filtered


def decimacia(signal, dec_koef: int):
    return signal[::dec_koef]


def check_error():
    frequencies = [40, 100, 300]
    delay = 4
    errors = {}

    for freq in frequencies:
        sin_base, _ = sin_gen(freq, 1)
        sin_up = interpolation(sin_base, UP_KOEF)
        sin_dec = decimacia(sin_up, DEC_KOEF)

        sin_dec_aligned = sin_dec[delay:]
        sin_base_aligned = sin_base[: len(sin_dec_aligned)]

        error = np.abs(sin_dec_aligned - sin_base_aligned)
        mae = np.mean(error)

        errors[freq] = error
        print(f"MAE после up и dec(sin{freq}): {mae:.6f}")

    plt.figure(figsize=(10, 8))
    for i, freq in enumerate(frequencies, 1):
        plt.subplot(3, 1, i)
        plt.plot(errors[freq])
        plt.title(f"Вектор ошибок(sin{freq})")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


sin, t = sin_gen(F, DURATION)
sin_up = interpolation(sin, UP_KOEF)
t_up = np.arange(0, DURATION, 1 / len(sin_up))
sin_dec = decimacia(sin_up, DEC_KOEF)

fft_sin = np.abs(np.fft.fft(sin)) / len(sin)
fft_freq = np.fft.fftfreq(len(sin), 1 / FS)
fft_freq_sh = np.fft.fftshift(fft_freq)
fft_log = np.fft.fftshift(20 * np.log10(fft_sin))

fft_sin_up = np.abs(np.fft.fft(sin_up)) / len(sin_up)
fft_freq_up = np.fft.fftfreq(len(sin_up), 1 / (len(sin_up)))
fft_freq_sh_up = np.fft.fftshift(fft_freq_up)
fft_log_up = np.fft.fftshift(20 * np.log10(fft_sin_up))

fft_sin_dec = np.abs(np.fft.fft(sin_dec)) / len(sin_dec)
fft_freq_dec = np.fft.fftfreq(len(fft_sin_dec), 1 / (len(sin_dec)))
fft_freq_sh_dec = np.fft.fftshift(fft_freq_dec)
fft_log_dec = np.fft.fftshift(20 * np.log10(fft_sin_dec))

check_error()

plt.figure()
plt.plot(t, sin)
plt.title("Исходный сигнал")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.grid(True)

plt.figure()
plt.plot(t_up, sin_up)
plt.title("Сигнал после интерполяции")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.grid(True)

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(fft_freq_sh, fft_log)
plt.title("Спектр исходного сигнала")
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда, дБ")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(fft_freq_sh_up, fft_log_up)
plt.title("Спектр после интерполяции")
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда, дБ")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(fft_freq_sh_dec, fft_log_dec)
plt.title("Спектр после децимации")
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда, дБ")
plt.grid(True)

plt.tight_layout()
plt.show()
