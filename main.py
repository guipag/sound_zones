from AlgoSZ import PressureMatchingGrad, metrics
from Room import Room
import numpy as np
import matplotlib.pyplot as plt


Fs = 4000
Ts = 2
nFir = 350

room = Room(nFir, Fs)

## Moller :
room.addMicArray((0.5, 2.5), 0.1, 'B')
room.addMicArray((4.5, 2.5), 0.1, 'D')
room.addHP([(0, 0), (2.5, 0), (5, 0), (5, 2.5), (5, 5), (2.5, 5), (0, 5), (0, 2.5)])
#room.setDim((6, 6), 0.5)
## GP :
#room.addMicArray((1, 1), 0.1, 'B')
#room.addMicArray((2, 2), 0.1, 'D')
#room.addHP([(0, 2.4), (0, 1.8), (0, 1.2), (0, 0.6), (0.6, 0), (1.2, 0), (1.8, 0), (2.4, 0), (3, 0.6), (3, 1.2), (3, 1.8),
#     (3, 2.4), (2.4, 3), (1.8, 3), (1.2, 3), (0.6, 3)])

## Lucas :
#room.addHP([(-0.58,0.01),(-0.505, 0.01),(-0.225,0.01),(-0.15,0.01),(0.15,0.01),(0.225,0.01),(0.505,0.01),(0.58,0.01)])
#room.addMic([(-0.235,0),(- 0.235,-0.05),(- 0.235,-0.1),(- 0.235,-0.15),(- 0.255,0),(- 0.255,-0.05),
#             (- 0.255,- 0.1),(- 0.255,- 0.15),(- 0.275,0),(- 0.275,- 0.05),(-0.275,- 0.1),(- 0.275,- 0.15),
#             (- 0.455,0),(- 0.455,- 0.05),(- 0.455,- 0.1),(- 0.455,- 0.15),(- 0.475,0),(- 0.475,- 0.05),
#             (- 0.475, - 0.1),(- 0.475,- 0.15),(-0.495,0),(- 0.495,- 0.05),(- 0.495,- 0.1),(- 0.495,- 0.15)], 'B')

#room.addMic([(-0.235+0.73,0),(- 0.235+0.73,-0.05),(- 0.235+0.73,-0.1),(- 0.235+0.73,-0.15),(- 0.255+0.73,0),(- 0.255+0.73,-0.05),
#             (- 0.255+0.73,- 0.1),(- 0.255+0.73,- 0.15),(- 0.275+0.73,0),(- 0.275+0.73,- 0.05),(-0.275+0.73,- 0.1),(- 0.275+0.73,- 0.15),
#             (- 0.455+0.73,0),(- 0.455+0.73,- 0.05),(- 0.455+0.73,- 0.1),(- 0.455+0.73,- 0.15),(- 0.475+0.73,0),(- 0.475+0.73,- 0.05),
#             (- 0.475+0.73, - 0.1),(- 0.475+0.73,- 0.15),(-0.495+0.73,0),(- 0.495+0.73,- 0.05),(- 0.495+0.73,- 0.1),(- 0.495+0.73,- 0.15)], 'D')


room.generateRi()

PMt = PressureMatchingGrad(room)
PMt.setDesire(94, 5)

w, e = PMt.solve(0.5, 1e-6, min_err=1e-8)

plt.figure()
plt.plot(10*np.log10(e[1:]))



# cont = []
# f_t = np.linspace(50, 2000, 100)

x = np.zeros((room.nbHP, Ts * Fs))
t = np.arange(Ts * Fs) / Fs
for no_in in range(room.nbHP):
    x[no_in, :] = np.sin(2 * np.pi * 100 * t)

# for f in f_t:
#     x = np.zeros((room.nbHP, Ts * Fs))
#     t = np.arange(Ts * Fs) / Fs
#     for no_in in range(room.nbHP):
#         x[no_in, :] = np.sin(2 * np.pi * f * t)
#
#     sig = room.conv(x)
#     cont.append(20 * np.log10(np.mean(CT(sig[9:18, :], sig[0:9, :]))))

CT, press, err = metrics(room, w, x, 0.1)


plt.figure()
plt.plot(np.arange(CT.shape[1])/Fs, CT.T)
plt.grid()
plt.show()
