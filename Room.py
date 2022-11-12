import numpy as np
import matplotlib.pyplot as plt
import math
import operator
import pyroomacoustics as pra
from scipy.fftpack import ifft
from conv_py import *
import concurrent.futures
from functools import partial

def dist(pos1, pos2):
    return math.sqrt(sum(v * v for v in tuple(map(operator.sub, pos1, pos2))))

def RIAnech(args, nFir, Ts):
    hp = args[0]
    mic = args[1]
    H = np.zeros(nFir, dtype=complex)
    Freq = np.arange(nFir) / (nFir * Ts)
    H[:int(nFir / 2)] = 1 / (4 * np.pi * dist(hp, mic)) * np.exp(-1J * 2 * np.pi * Freq[:int(nFir / 2)] * dist(hp, mic) / 343)
    H[int(nFir / 2) + 1:] = np.conjugate(H[int(nFir / 2) - 1:0:-1])
    return np.real(ifft(H))
def longest(a):
    return max(len(a), *map(longest, a)) if isinstance(a, list) and a else 0


class Mic:
    def __init__(self, pos = (0, 0, 0), type = 'B'):
        self.pos = pos
        self.type = type


class MicLTV(Mic):
    V = 1

    def __init__(self, pos, type = 'B'):
        self.traj = pos
        self.t = 0
        self.type = type

    def setT(self, t):
        self.t = t

    @property
    def pos(self):
        return self.traj[0], self.traj[1]+MicLTV.V*self.t


class HP:
    def __init__(self, pos = (0, 0, 0)):
        self.pos = pos


class Room:
    C0 = 343
    RHO = 1.2

    def __init__(self, nFir, Fs):
        self.nbHP = 0
        self.nbMic = 0
        self.nFir = nFir
        self.Ts = 1/Fs

        self.mic = []
        self.hp = []

        self.h = []

    @staticmethod
    def conv_MIMO(h, x):
        dim = h.shape
        sig = np.zeros((dim[0], dim[1] + x.shape[1] - 1))
        for no_in in range(dim[0]):
            sig[no_in, :] = np.convolve(h[no_in, :], x[no_in, :])
        return sig

    def conv(self, inp, withFilter = True):
        if withFilter and hasattr(self, 'filt'):
            inp = self.conv_MIMO(self.filt, inp)
        out = np.zeros((self.nbMic, inp.shape[1]+self.h.shape[2]-1))
        for i in range(self.nbMic):
            for j in range(self.nbHP):
                out[i, :] += np.convolve(self.h[j, i, :], inp[j, :])
        return out

    def setH(self, hp, mic, h):
        self.h[hp, mic, 0:len(h)] = h
        return self

    def getH(self, type = None):
        if type == None:
            return self.h
        elif type == 'B':
            return self.h[:, np.where(np.array([x.type for x in self.mic]) == 'B')[0].tolist(), :]
        elif type == 'D':
            return self.h[:, np.where(np.array([x.type for x in self.mic]) == 'D')[0].tolist(), :]
        else:
            return -1

    def setDim(self, dim, rt60):
        e_absorption, max_order = pra.inverse_sabine(rt60, dim)
        self.room = pra.ShoeBox(dim, fs=1/self.Ts, max_order=max_order, materials=pra.Material(e_absorption))

    def getG(self, f, type = None):
        G = np.zeros((self.nbMic, self.nbHP), dtype="complex")
        mic = np.array(self.mic)[np.where(np.array([x.type for x in self.mic]) == type)[0].tolist()]

        for i in range(len(mic)):
            for j in range(self.nbHP):
                G[j, i] = self.GAnech(self.hp[j], mic[i], f)
        return G

    def longest(self, l):
        if not isinstance(l, list):
            return len(l)
        return max(
            [len(l)]
            + [len(subl) for subl in l if isinstance(subl, list)]
            + [self.longest(subl) for subl in l]
        )

    def generateRi(self):
        # si room existe alors lancer pyroom
        if hasattr(self, 'room'):
            for hp in self.hp:
                self.room.add_source([item for item in hp.pos])
            for mic in self.mic:
                self.room.add_microphone([item for item in mic.pos])
            self.room.image_source_model()
            self.room.compute_rir()
            self.nFir = self.longest(self.room.rir)
            self.h = np.zeros((self.nbHP, self.nbMic, self.nFir))
            for i in range(self.nbMic):
                for j in range(self.nbHP):
                    self.setH(j, i, self.room.rir[i][j])
        else:
            self.h = np.zeros((self.nbHP, self.nbMic, self.nFir))
            for i in range(self.nbMic):
                for j in range(self.nbHP):
                    self.setH(j, i, self.RIAnech(self.hp[j], self.mic[i]))
        return self

    def generateRiFast(self):
        # self.h = np.zeros((self.nbHP, self.nbMic, self.nFir))
        # for i in range(self.nbMic):
        #     for j in range(self.nbHP):
        #         self.h[j, i, :] = self.RIAnech(self.hp[j], self.mic[i])
                # self.setH(j, i, self.RIAnech(self.hp[j], self.mic[i]))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            h = np.array(
                list(executor.map(partial(RIAnech,nFir=self.nFir,Ts=self.Ts) , zip([[hp.pos, mic.pos] for hp in self.hp for mic in self.mic]))))
        return self

    def setMic(self, mic):
        self.mic = mic
        self.nbMic = len(mic)
        if hasattr(self, 'room'):
            self.room.add_microphone_array(pra.MicrophoneArray(R, self.room.fs))
        return self

    def addMic(self, pos, type):
        if isinstance(pos, tuple):
            self.mic.append(Mic(pos, type))
            self.nbMic += 1
        else:
            for pos_l in pos:
                self.mic.append(Mic(pos_l, type))
                self.nbMic += 1
        return self

    def addMicArray(self, c, e, type):
        return self.addMic([(c[0]+e, c[1]), (c[0]-e, c[1]), (c[0]+e, c[1]+e), (c[0], c[1]+e), (c[0]-e, c[1]+e), (c[0]-e, c[1]-e), (c[0], c[1]-e), (c[0]+e, c[1]-e), (c[0], c[1])], type)

    def addHP(self, pos):
        if isinstance(pos, tuple):
            self.hp.append(HP(pos))
            self.nbHP += 1
        else:
            for pos_l in pos:
                self.hp.append(HP(pos_l))
                self.nbHP += 1
        return self

    def addFilter(self, filt):
        self.filt = filt
        return self

    def convFourier(self, filt, f):
        Ppm = np.zeros(len(self.mic), dtype="complex")

        for idx_m, mic in enumerate(self.mic):
            for idx_h, hp in enumerate(self.hp):
                Ppm[idx_m] += self.GAnech(hp, mic, f) * filt[idx_h]

        return Ppm

    @staticmethod
    def dist(pos1, pos2):
        return math.sqrt(sum(v*v for v in tuple(map(operator.sub, pos1, pos2))))

    def RIAnech(self, hp, mic):
        H = np.zeros(self.nFir, dtype=complex)
        Freq = np.arange(self.nFir)  / (self.nFir*self.Ts)
        H[:int(self.nFir / 2)] = 1/(4*np.pi*self.dist(hp.pos, mic.pos)) * np.exp(-1J * 2 * np.pi * Freq[:int(self.nFir / 2)] * self.dist(hp.pos, mic.pos)/self.C0)
        H[int(self.nFir / 2) + 1:] = np.conjugate(H[int(self.nFir / 2) - 1:0:-1])
        return np.real(ifft(H))

    def GAnech(self, hp, mic, f):
        k = 2 * np.pi * f / Room.C0
        return 1j * k * Room.RHO * Room.C0 * np.exp(-1j * k * self.dist(hp.pos, mic.pos)) / (4 * np.pi * self.dist(hp.pos, mic.pos))

    def plot(self):
        plt.figure()
        coord = []
        for mic in self.mic:
            coord.append(mic.pos)
        x, y = zip(*coord)
        plt.scatter(x, y)
        coord = []
        for hp in self.hp:
            coord.append(hp.pos)
        x, y = zip(*coord)
        plt.scatter(x, y)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.grid()
        plt.show()


class RoomLTV(Room):
    def __init__(self, nFir, Fs):
        super(RoomLTV, self).__init__(nFir, Fs)
        self.t = 0

    def addMic(self, pos, type):
        self.mic.append(MicLTV(pos, type))
        self.nbMic += 1
        return self

    def addMicArray(self, c, e, type):
        for pos_mic in [(c[0]+e, c[1]), (c[0]-e, c[1]), (c[0]+e, c[1]+e), (c[0], c[1]+e), (c[0]-e, c[1]+e), (c[0]-e, c[1]-e), (c[0], c[1]-e), (c[0]+e, c[1]-e), (c[0], c[1])]:
            self.addMic(pos_mic, type)
        return self

    def setT(self, t):
        self.t = t
        for mic in self.mic:
            mic.setT(t)
        self.generateRi()
        return self

    def getAllH(self, t_tot):
        h = np.zeros((len(t_tot), self.nbMic, self.nbHP, self.nFir))
        for no_mic in range(self.nbMic):
            for n, t in enumerate(t_tot):
                self.setT(t)
                h[n, no_mic, :, :] = np.squeeze(self.h[:, no_mic, :])
        return h

    def conv(self, inp, withFilter = True):
        t_tot = np.arange(inp.shape[1])*self.Ts
        if withFilter and hasattr(self, 'filt'):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                inp = np.array(list(executor.map(conv_LTV_MIMO_par, zip([self.filt[:, no_hp, :] for no_hp in range(self.nbHP)], inp))))
        h = np.zeros((self.nbMic, len(t_tot), self.nbHP, self.nFir))
        with open('var_h_1_10cm.npy', 'rb') as f:
            h = np.load(f)
        # for no_mic in range(self.nbMic):
        #     for n, t in enumerate(t_tot):
        #         self.setT(t)
        #         h[no_mic, n, :, :] = np.squeeze(self.h[:, no_mic, :])
        # with open('var_h_1_10cmb.npy', 'wb') as f:
        #     np.save(f, h)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            out = np.array(list(executor.map(partial(conv_par, y=inp), list(zip(range(self.nbMic), [h[no_mic, :, :, :] for no_mic in range(self.nbMic)])))))

        return out


if __name__ == "__main__":
    Fs = 10000
    Ts = 2

    t = np.arange(int(Ts*Fs))/Fs
    x = np.sin(2*np.pi*100*t)

    h = np.ones((2, len(x)))
    h[:, :len(x)//2] = 0.5

    y = conv_LTV(h, x)

    plt.figure()
    plt.plot(t, y)
    plt.show()