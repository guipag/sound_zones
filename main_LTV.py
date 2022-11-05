import matplotlib.pyplot as plt
import numpy as np
import Room
from AlgoSZ import metrics, PressureMatchingGradLTV
import tqdm

if __name__ == '__main__':
    Fs = 4000
    Ts = 2
    nFir = 350

    room = Room.RoomLTV(nFir, Fs)

    room.addMicArray((0.5, 2.5), 0.1, 'B')
    room.addMicArray((4.5, 2.5), 0.1, 'D')
    room.addHP([(0, 0), (2.5, 0), (5, 0), (5, 2.5), (5, 5), (2.5, 5), (0, 5), (0, 2.5)])

    t_tot = np.arange(int(Ts*Fs))/Fs
    room.setT(0)

    PMt = PressureMatchingGradLTV(room)
    PMt.setDesire(94, 6)
    w = np.zeros((int(Ts*Fs), room.nbHP, nFir))
    e = np.zeros(int(Ts*Fs))
    for n, t in tqdm.tqdm(enumerate(t_tot[::])):
       PMt.room.setT(t)

       w[n, :, :], e[n] = PMt.step(0.5, 1e-6)

    with open('var_w_05.npy', 'wb') as f:
        np.save(f, w)
    # with open('var_w_05.npy', 'rb') as f:
    #     w = np.load(f)
    plt.figure()
    plt.plot(e)
    plt.show()

    x = np.zeros((room.nbHP, Ts * Fs))
    t = np.arange(Ts * Fs) / Fs
    sig = np.random.randn(len(t))
    f = 100
    for no_in in range(room.nbHP):
        x[no_in, :] = np.sin(2 * np.pi * f * t)

    CT, press, err = metrics(room, w, x, 0.1)

    plt.figure()
    plt.plot(np.arange(CT.shape[1])/Fs, 20*np.log10(CT.T))
    plt.xlabel('Time (s)')
    plt.ylabel('Contrast (dB)')
    plt.title('Contrast {} Hz'.format(f))
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(np.arange(press.shape[1]) / Fs, 20 * np.log10(press.T))
    plt.xlabel('Time (s)')
    plt.ylabel('Pression (dB)')
    plt.title('Pression {} Hz'.format(f))
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(np.arange(err.shape[1]) / Fs, err.T*100)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (%)')
    plt.title('Error {} Hz'.format(f))
    plt.grid()
    plt.tight_layout()

    plt.show()

    # room.plot()

    pass