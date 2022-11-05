from Room import *
from AlgoSZ import *



if __name__ == "__main__":
    Fs = 4000
    room = Room(350, Fs)

    #room.addMic([(1.1, 1), (0.9, 1), (1.1, 1.1), (1, 1.1), (0.9, 1.1), (0.9, 0.9), (1, 0.9), (1.1, 0.9), (1, 1)], 'B')
    room.addMicArray((1,1), 0.1, 'B')
    #room.addMic([(2.1, 2), (1.9, 2), (2.1, 2.1), (2, 2.1), (1.9, 2.1), (1.9, 1.9), (2, 1.9), (2.1, 1.9), (2, 2)], 'D')
    room.addMicArray((2,2), 0.1, 'D')

    # room.addMic([(-0.235,0),(- 0.235,-0.05),(- 0.235,-0.1),(- 0.235,-0.15),(- 0.255,0),(- 0.255,-0.05),
    #              (- 0.255,- 0.1),(- 0.255,- 0.15),(- 0.275,0),(- 0.275,- 0.05),(-0.275,- 0.1),(- 0.275,- 0.15),
    #              (- 0.455,0),(- 0.455,- 0.05),(- 0.455,- 0.1),(- 0.455,- 0.15),(- 0.475,0),(- 0.475,- 0.05),
    #              (- 0.475, - 0.1),(- 0.475,- 0.15),(-0.495,0),(- 0.495,- 0.05),(- 0.495,- 0.1),(- 0.495,- 0.15)], 'B')
    #
    # room.addMic([(-0.235+0.73,0),(- 0.235+0.73,-0.05),(- 0.235+0.73,-0.1),(- 0.235+0.73,-0.15),(- 0.255+0.73,0),(- 0.255+0.73,-0.05),
    #              (- 0.255+0.73,- 0.1),(- 0.255+0.73,- 0.15),(- 0.275+0.73,0),(- 0.275+0.73,- 0.05),(-0.275+0.73,- 0.1),(- 0.275+0.73,- 0.15),
    #              (- 0.455+0.73,0),(- 0.455+0.73,- 0.05),(- 0.455+0.73,- 0.1),(- 0.455+0.73,- 0.15),(- 0.475+0.73,0),(- 0.475+0.73,- 0.05),
    #              (- 0.475+0.73, - 0.1),(- 0.475+0.73,- 0.15),(-0.495+0.73,0),(- 0.495+0.73,- 0.05),(- 0.495+0.73,- 0.1),(- 0.495+0.73,- 0.15)], 'D')

    room.addHP([(0, 2.4), (0, 1.8), (0, 1.2), (0, 0.6), (0.6, 0), (1.2, 0), (1.8, 0), (2.4, 0), (3, 0.6), (3, 1.2), (3, 1.8), (3, 2.4), (2.4, 3), (1.8, 3), (1.2, 3), (0.6,3)])
    # room.addHP([(-0.58,0.01),(-0.505, 0.01),(-0.225,0.01),(-0.15,0.01),(0.15,0.01),(0.225,0.01),(0.505,0.01),(0.58,0.01)])
    #room.setDim([4, 3], 0.3)
    room.generateRi()
    #room.plot()

    PMt = PressureMatchingFourier(room)
    f=200
    #PMf = PressureMatchingFourier(room)
    # test = ACC(room)
    PMt.setDesire(f, 94, 3)
    #PMf.setDesire(100, 94, 1)

    w = PMt.solve(f, 1)

    P = room.convFourier(w, f)
    Lppm = 20 * np.log10(abs(P) / 2e-5)

    Lppm = 20 * np.log10(abs(Ppm) / 2e-5)
    contrast = 10*np.log10(np.mean(np.abs(Ppm[0:9])**2)/np.mean(np.abs(Ppm[10:19])**2))

    room.addFilter(w, 500)
    #wf = PMf.solve(100, 10)

    #mat_contents = sio.loadmat('matlab.mat')

    #w = np.reshape(w, (room.nbHP, room.nFir))
    cont = []
    err = []
    f_t = np.linspace(50, 1500, 300)

    h_t = room.getH('B')
    h_t = h_t[2, :, :]
    Ts = 1

    plt.figure()
    plt.plot(h_t[1,:])
    plt.show()

    for f in f_t:
        x = np.zeros((room.nbHP, Ts*Fs))
        t = np.arange(Ts*Fs)/Fs
        for no_in in range(room.nbHP):
            x[no_in, :] = np.sin(2*np.pi*f*t)
            # if no_in == 0:
            #     p_t = conv_MIMO(h_t, 2e-5*10**(94/20)*x)

        sig = room.conv(x)
        cont.append(20*np.log10(np.mean(CT(sig[10:19, :], sig[0:10, :]))))
        # err.append(abs(np.mean(np.sqrt(np.mean(sig[0:9, :]**2)) - np.sqrt(np.mean(p_t**2)))) / (np.mean(np.sqrt(np.mean(p_t**2)))) * 100)

    print(cont)

    plt.figure()
    plt.semilogx(f_t, cont)
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(w[0, :])
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(w[1, :])
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(w[3, :])
    plt.grid()
    plt.show()
    pass
    # plt.figure()
    # plt.plot(f_t, err)
    # plt.show()
