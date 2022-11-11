import numpy as np
import scipy.linalg as lg
import scipy.signal as sg


class AlgoSZ:
    def __init__(self, room):
        self.room = room
        self.nFir = room.nFir
        self.nbHP = room.nbHP
        # à modifier pour prendre en compte des différences possibles
        self.nbMic = room.nbMic//2

    def getToeplitz(self, H, mic, hp):
        return lg.toeplitz(np.concatenate((np.squeeze(H[hp, mic, :]), np.zeros(self.nFir - 1))), np.concatenate(([H[hp, mic, 0]], np.zeros(self.nFir - 1))))

    def setH(self, HB, HD):
        self.HB = np.block(
            [[self.getToeplitz(HB, no_Mic, no_HP) for no_HP in range(self.nbHP)] for no_Mic in range(self.nbMic)])
        self.HD = np.block(
            [[self.getToeplitz(HD, no_Mic, no_HP) for no_HP in range(self.nbHP)] for no_Mic in range(self.nbMic)])
        self.HB_d = HB

        return self

    @staticmethod
    def conv_MISO(h, x):
        dim = h.shape
        sig = np.zeros(dim[1] + x.shape[1] - 1)
        for no_in in range(dim[0]):
            sig += np.convolve(h[no_in, :], x[no_in, :])
        return sig

    @staticmethod
    def conv_MIMO(h, x):
        dim = h.shape
        sig = np.zeros((dim[0], dim[1] + x.shape[1] - 1))
        for no_in in range(dim[0]):
            sig[no_in, :] = sg.oaconvolve(h[no_in, :], x[no_in, :])
        return sig


class PressureMatching(AlgoSZ):
    def __init__(self, room):
        super().__init__(room)
        self.setH(room.getH('B'), room.getH('D'))

    def setDesire(self, target, noHP):
        if isinstance(noHP, tuple):
            pass
        else:
            self.d = np.block([np.concatenate((2e-5*10**(target/20)*np.squeeze(self.HB_d[noHP,no_Mic,:]), np.zeros(self.nFir-1))) for no_Mic in range(self.nbMic)])
        self.hpTarget = noHP
        self.target = target

        return self

    def solve(self, beta, regu):
        mat_inv = (beta * self.HB.T @ self.HB + (1 - beta) * self.HD.T @ self.HD + regu * np.eye(self.HB.shape[1]))
        # h = lg.inv(mat_inv) @ (self.HB.T @ self.d)
        h = lg.solve(mat_inv, self.HB.T @ self.d)
        h = np.reshape(h, (self.nbHP, self.nFir))
        return h


class PressureMatchingGrad(AlgoSZ):
    def __init__(self, room):
        super().__init__(room)
        self.setH(room.getH('B'), room.getH('D'))

    def setDesire(self, target, noHP):
        self.d = np.block([np.concatenate((2e-5*10**(target/20)*np.squeeze(self.HB_d[noHP, no_Mic ,:]), np.zeros(self.nFir - 1))) for no_Mic in range(self.nbMic)])

        return self

    # def solve(self, beta, regu, min_err = 1e-10):
    #     h = np.zeros(self.nFir*self.nbHP)
    #     mu = 2
    #     err = [1]
    #     while err[-1] > min_err:
    #         h_0 = h
    #         h = h - 1/2 * mu * ((beta * self.HB.T @ self.HB + (1 - beta) * self.HD.T @ self.HD + regu * np.eye(self.HB.shape[1])) @ h - 2 * self.HB.T @ self.d)
    #         # descente stochastique / batch (ADAM)
    #         err.append(np.mean((h-h_0)**2))
    #
    #     h = np.reshape(h, (self.nbHP, self.nFir))
    #     return h, err
    def solve(self, beta, regu, min_err = 1e-10):
        h = np.zeros(self.nFir*self.nbHP)
        mu = 1
        err = [1]
        while err[-1] > min_err:
            h_0 = h

            H_t = (beta * self.HB.T @ self.HB + (1 - beta) * self.HD.T @ self.HD + regu * np.eye(self.HB.shape[1]))
            r = 2*(self.HB.T @ self.d-H_t@h)
            mu = (r.T@r)/(2*r.T@H_t.T@r)
            # regu = (r.T@H_t@H_t.T@r@r.T@H_t@H_t.T@H_t@r-r.T@H_t@h@r.T@H_t@H_t.T@H_t@H_t.T@r)/(r.T@H_t@h@r.T@H_t@H_t.T@H_t@h-r.T@H_t@H_t.T@r@h.T@H_t.T@H_t@h)
            h = h + mu * r
            # descente stochastique / batch (ADAM)
            err.append(np.mean((h-h_0)**2))

        h = np.reshape(h, (self.nbHP, self.nFir))
        return h, err


class PressureMatchingGradLTV(PressureMatchingGrad):
    def __init__(self, room):
        super().__init__(room)
        self.w = np.zeros(self.nFir * self.nbHP)

    def step(self, beta, regu):
        mu = 1
        w_0 = self.w
        self.setH(self.room.getH('B'), self.room.getH('D'))
        H_t = (beta * self.HB.T @ self.HB + (1 - beta) * self.HD.T @ self.HD + regu * np.eye(self.HB.shape[1]))
        r = 2 * (self.HB.T @ self.d - H_t @ self.w)
        mu = (r.T @ r) / (2 * r.T @ H_t.T @ r)
        # self.w = self.w - 1 / 2 * mu * ((beta * self.HB.T @ self.HB + (1 - beta) * self.HD.T @ self.HD + regu * np.eye(self.HB.shape[1])) @ self.w - 2 * self.HB.T @ self.d)
        self.w = self.w + mu * r
        e = np.mean((self.w - w_0) ** 2)
        return np.reshape(self.w, (self.nbHP, self.nFir)), e


class PressureMatchingAdaptive(AlgoSZ):
    def __init__(self, room):
        super().__init__(room)
        self.setH(room.getH('B'), room.getH('D'))

    def setDesire(self, target, noHP):
        self.d = np.block([np.concatenate((2e-5*10**(target/20)*np.squeeze(self.HB_d[noHP,no_Mic,:])+2e-5*10**(target/20)*np.squeeze(self.HB_d[noHP+1,no_Mic,:]), np.zeros(self.nFir-1))) for no_Mic in range(self.nbMic)])
        self.HPtarget = noHP
        return self

    def calcDesiredSignal(self, inp, target):
        h = self.room.h[self.HPtarget, np.where(np.array([x.type for x in self.room.mic]) == 'B')[0].tolist(), :]
        self.sig_d = room.conv_MIMO(h, 2e-5*10**(target/20)/(np.sqrt(np.mean(inp**2)))*inp)
        return self

class PressureMatchingFourier(AlgoSZ):
    def __init__(self, room):
        super().__init__(room)

    def setDesire(self, f, target, noHP):
        self.Pt = 2e-5*10**(target/20)*self.room.getG(f, 'D')[:, noHP]

    def solve(self, f, regu):
        Qpm = lg.inv((np.conjugate(np.transpose(self.room.getG(f, 'B'))) @ self.room.getG(f, 'B')) + np.conjugate(np.transpose(self.room.getG(f, 'D'))) \
                         @ self.room.getG(f, 'D') + regu * np.eye(self.nbHP)) @ (np.conjugate(np.transpose(self.room.getG(f, 'B'))) @ self.Pt)
        return Qpm


class ACCFourier(AlgoSZ):
    def __init__(self, room):
        super().__init__(room)

    def solve(self, f, regul):
        Gb = self.room.getG(f, 'B')
        Gd = self.room.getG(f, 'D')
        Gdp = Gd.conj().T
        Gbp = Gb.conj().T
        eVal, Qac = np.linalg.eig((Gbp @ Gb + Gdp @ Gd + regul * np.eye(self.nbHP)))
        val, ne = np.amax(eVal), np.argmax(eVal)
        return Qac[:, ne]


def CT(x, y):
    nbChannel = x.shape[0]
    CT = np.zeros(nbChannel)
    for no_channel in range(nbChannel):
        CT[no_channel] = np.sqrt(np.mean(x[no_channel, :]**2)) / np.sqrt(np.mean(y[no_channel, :]**2))
    return CT


def CT_ltv(x, y, nInt):
    len_sig = x.shape[1]
    CT = np.zeros(len_sig)
    for n in range(nInt, len_sig):
        CT[n] = np.mean(np.sqrt(np.mean(y[n-nInt:n]**2)))/np.mean(np.sqrt(np.mean(x[n-nInt:n]**2)))
    return 20*np.log10(CT)


def metrics(room, filter, inp, tInt):
    room.addFilter(filter)
    z = room.conv(inp)
    nInt = int(tInt/room.Ts)
    len_sig = inp.shape[1]
    CT = np.zeros((room.nbMic//2, len_sig))
    for n in range(nInt, len_sig):
        CT[:, n] = np.sqrt(np.mean(z[0:room.nbMic//2, n - nInt:n] ** 2, 1)) / np.mean(np.sqrt(np.mean(z[room.nbMic//2:room.nbMic, n - nInt:n] ** 2, 1)))

    press = np.zeros((room.nbMic, len_sig))
    for n in range(nInt, len_sig):
        press[:, n] = np.sqrt(np.mean(z[:, n - nInt:n] ** 2, 1)) / 2e-5

    inp_err = np.zeros_like(inp)
    inp_err[6, :] = inp[6, :]
    err = np.zeros((room.nbMic//2, len_sig))
    p = room.conv(inp_err, False)
    for n in range(nInt, len_sig):
        err[:, n] = (np.sqrt(np.mean(p[0:room.nbMic//2, n - nInt:n] ** 2, 1)) - 2e-5*10**(94/20)) / np.sqrt(np.mean(p[0:room.nbMic//2, n - nInt:n] ** 2, 1))

    return CT, press, err