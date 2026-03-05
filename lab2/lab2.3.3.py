#3.3 Part C
# (1) Plot the audio signal “Cafe_with_noise.wav” using matplotlib.


from scipy.io.wavfile import read as read_wav

sample_rate, data = read_wav("/Users/amandavielmann/Downloads/Cafe_with_noise.wav")

print(f"The original sampling frequency is: {sample_rate} Hz")





import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as read_wav

sample_rate, data = read_wav("/Users/amandavielmann/Downloads/Cafe_with_noise.wav")


if len(data.shape) > 1:
    data = data[:,0]

t = np.arange(len(data)) / sample_rate

plt.figure(figsize=(10,4))
plt.plot(t, data)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Original Signal")
plt.show()

# (2) Analyze the signal in the frequency domain to separate regions of human voice to noise.#
import numpy as np
import matplotlib.pyplot as plt

x = data.astype(np.float32)

N = len(x)
X = np.fft.rfft(x)                       
freqs = np.fft.rfftfreq(N, d=1/sample_rate)

mag = np.abs(X) / N

plt.figure(figsize=(10,4))
plt.plot(freqs, mag)
plt.xlim(0, 8000)                          
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum (FFT)")
plt.grid(True)
plt.show()

#(3) Implement a low pass filter to remove the noise and obtain only the voice signal.#
import numpy as np
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt

def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


sample_rate, data = scipy.io.wavfile.read("/Users/amandavielmann/Downloads/Cafe_with_noise.wav")


if data.ndim > 1:
    data = data[:,0]


data = data.astype(np.float32) / 32767.0


filtered = lowpass(data, cutoff=3500, sample_rate=sample_rate, poles=6)


times = np.arange(len(data)) / sample_rate

plt.figure(figsize=(10,4))
plt.plot(times[:2000], data[:2000], label="Noisy")
plt.plot(times[:2000], filtered[:2000], label="Filtered")
plt.legend()
plt.show()
