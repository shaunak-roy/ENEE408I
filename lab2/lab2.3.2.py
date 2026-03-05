##Lab 2##

from scipy.io.wavfile import read as read_wav
import numpy as np
import matplotlib.pyplot as plt

#3.2 Part B
# (1) Calculate the rms values for each of the audio signals M1.wav, M2.wav and M3.wav using
# Python. (Here M1.wav is the audio signal received by the microphone M1)

m1_sampling_rate, m1 = read_wav("M1.wav")
m2_sampling_rate, m2 = read_wav("M2.wav")
m3_sampling_rate, m3 = read_wav("M3.wav")

m1 = m1.astype(np.float64)
m2 = m2.astype(np.float64)
m3 = m3.astype(np.float64)

m = np.vstack([m1, m2, m3])
squared = np.square(m)
means = np.mean(squared, axis=1)
rms = np.sqrt(means)
print("RMS:", rms)

# (3) Using crosscorrelation, calculate the time delay between the audio signals received
# between M1 and M2 using Python. Try to compute this without the use of inbuilt functions.

N = len(m1)
M = len(m2)

time_per_sample = 1/m1_sampling_rate

m1_padded = np.pad(m1, M-1)
cross = np.zeros(N + M - 1)

for i in range(N + M - 1):
    segment = m1_padded[i:i+M]
    cross[i] = np.sum(segment * m2)


print("Time delay: ", np.argmax(cross) * time_per_sample, "sec")

# (4) Angle theta using law of cosines + law of sines (prof derivation)

c = 343.0      # speed of sound (m/s)
r = 0.10       # robot radius = 10 cm = 0.10 m

# using our time delay we calculated
dt = np.argmax(cross) * time_per_sample   # seconds

# distance difference from time delay: d2 - d1
delta_d = c * dt

# estimate d1 and d2 from RMS (simple model: amplitude ~ 1/d so d ~ 1/RMS)
rms1 = float(rms[0])   # M1 rms
rms2 = float(rms[1])   # M2 rms

# Solve for d1, d2 using:
# d2 - d1 = delta_d
# d1/d2 = rms2/rms1   (since d ~ 1/RMS)
ratio = rms2 / rms1
if abs(1.0 - ratio) < 1e-12:
    raise ValueError("RMS ratio too close to 1; cannot estimate d1 and d2 reliably.")

d2 = delta_d / (1.0 - ratio)
d1 = ratio * d2

# Make distances positive if needed (just in case sign conventions produce negative)
if d1 < 0 and d2 < 0:
    d1 *= -1
    d2 *= -1

# phi1 = arccos( (-d2^2 + d1^2 + 4r^2) / (4 d1 r) )
cos_phi1 = (-d2**2 + d1**2 + 4*(r**2)) / (4*d1*r)
cos_phi1 = np.clip(cos_phi1, -1.0, 1.0)
phi1 = np.arccos(cos_phi1)

# d3 = sqrt(d1^2 + r^2 - 2 d1 r cos(phi1))
d3 = np.sqrt(d1**2 + r**2 - 2*d1*r*np.cos(phi1))

# theta = -pi/2 + arcsin( (d1 sin(phi1)) / d3 )
arg = (d1 * np.sin(phi1)) / d3
arg = np.clip(arg, -1.0, 1.0)
theta = -np.pi/2 + np.arcsin(arg)

print("dt =", dt, "sec")
print("delta_d = d2 - d1 =", delta_d, "m")
print("Estimated d1 =", d1, "m")
print("Estimated d2 =", d2, "m")
print("phi1 (rad) =", phi1)
print("d3 (m) =", d3)
print("theta (rad) =", theta)
print("theta (deg) =", np.degrees(theta))


