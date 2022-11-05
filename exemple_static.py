import matplotlib.pyplot as plt
import numpy as np
import PressureMatching as PM
import Room
from AlgoSZ import CT

if __name__ == '__main__':
    Fs = 4000
    Ts = 2
    nFir = 350

    room = Room.Room(nFir, Fs)

    room.addMicArray((0.5, 2.5), 0.1, 'B')
    room.addMicArray((4.5, 2.5), 0.1, 'D')
    room.addHP([(0, 0), (2.5, 0), (5, 0), (5, 2.5), (5, 5), (2.5, 5), (0, 5), (0, 2.5)])

    room.plot()

    room.generateRi()

    PMt = PM.PressureMatching(room)

    PMt.setDesire(target=94, noHP=6)
    w = PMt.solve(beta=0.5, regu=1e-6)

    room.addFilter(w)

    x = np.zeros((room.nbHP, Ts * Fs))
    t = np.arange(Ts * Fs) / Fs
    for no_in in range(room.nbHP):
        x[no_in, :] = np.sin(2 * np.pi * 100 * t)

    sig = room.conv(x)

    print(20*np.log10(CT(x, sig)))

    plt.figure()
    plt.plot(sig.T)
    plt.show()