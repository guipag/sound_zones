import matplotlib.pyplot as plt
import numpy as np
import Room
from AlgoSZ import metrics, PressureMatchingGradLTV, PressureMatching
import tqdm

if __name__ == '__main__':
    Fs = 2000#4000
    Ts = 1#2
    nFir = 200#350

    room = Room.RoomLTV(nFir, Fs)

    room.addMicArray((0.5, 2.5), 0.1, 'B')
    room.addMicArray((4.5, 2.5), 0.1, 'D')
    room.addHP([(0, 0), (2.5, 0), (5, 0), (5, 2.5), (5, 3.75),(5, 5),(3.75,5), (2.5, 5),(1.25, 5), (0, 5), (0, 3.75), (0, 2.5)])

    t_tot = np.arange(int(Ts*Fs))/Fs
    room.setT(0)

    PMt = PressureMatchingGradLTV(room)
    PMs = PressureMatching(room)
    PMt.setDesire(94, 6)
    PMs.setDesire(94, 6)
    w_s = np.zeros((int(Ts*Fs), room.nbHP, nFir))
    e = np.zeros(int(Ts*Fs))
    PMt.room.setT(0)
    PMs.room.setT(0)
    for n, t in tqdm.tqdm(enumerate(t_tot[::])):
        if (n%400) == 0:  #10cm
            PMs.room.setT(t)
            PMs.setDesire(94, 7)
            w_s_temp = PMs.solve(0.5, 1e-10)
        w_s[n, :, :] = w_s_temp
        #w_s[n, :, :], e[n] = PMt.step(0.5, 1e-10)
    #
    with open('var_w_1_10cmb_s.npy', 'wb') as f:
        np.save(f, w_s)
    # with open('var_w_1_10cmb.npy', 'rb') as f:
    #     w = np.load(f)
    # plt.figure()
    # plt.plot(e)

    x = np.zeros((room.nbHP, Ts * Fs))
    t = np.arange(Ts * Fs) / Fs
    sig = np.random.randn(len(t))
    f = 800
    for no_in in range(room.nbHP):
        x[no_in, :] = np.sin(2 * np.pi * f * t)

    CT, press, err = metrics(room, w_s, x, 0.1)

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
    plt.plot(np.arange(press.shape[1]) / Fs, 20 * np.log10(press[15,:]))
    plt.xlabel('Time (s)')
    plt.ylabel('Pression (dB)')
    plt.title('Pression {} Hz'.format(f))
    plt.grid()
    plt.tight_layout()

    plt.figure()
    plt.plot(np.arange(err.shape[1]) / Fs, err.T)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (%)')
    plt.title('Error {} Hz'.format(f))
    plt.grid()
    plt.tight_layout()

    # plt.figure()
    # plt.plot(np.arange(nFir) / Fs, w[1000,7,:])
    # plt.xlabel('Time (s)')
    # plt.ylabel('Error (%)')
    # plt.title('Error {} Hz'.format(f))
    # plt.grid()
    # plt.tight_layout()

    #plt.show()

    room.plot()

    pass