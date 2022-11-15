import Room
import AlgoSZ
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lg

class MH_SZ(AlgoSZ.AlgoSZ):
    def __init__(self, room, Ic, Ip):
        super().__init__(room)
        self.nFir = room.nFir
        self.wb = np.zeros(room.nFir * room.nbHP * Ic)
        self.Ic = Ic
        self.Ip = Ip
        self.u = np.zeros(room.nFir)
        self.A = np.roll(np.eye(self.nFir - 1), -1)
        self.A[-1, -1] = 0
        self.b = np.zeros(room.nFir - 1)
        self.b[1] = 1
        self.k = 0

    def step(self, u):
        self.u = np.roll(self.u, -1)
        self.u[0] = u
        self.k += 1

    def setInput(self, inp):
        self.inp = inp
        self.h = room.getAllH(np.arange(len(inp))*self.room.Ts) # h = (len(t_tot), self.nbMic, self.nbHP, self.nFir)

    def setTarget(self):
        self.p_t = 0
        pass

    def solve(self, a_d, a_t, a_w, a_env, a_bs):
        Q_d = a_d * self.Ub().T @ self.psib_d().T @ self.psib_d() @ self.Ub()
        Q_b = a_t * self.Ub().T @ self.psib_b().T @ self.psib_b() @ self.Ub()
        Q_w = a_w * np.eye(len(Q_d))
        Q_env = a_env * np.kron(np.ones(self.room.nbHP * self.Ic), self.w_env)
        Q_bs = a_bs * self.Ub().T @ self.psib_s().T @ self.psib_s() @ self.Ub()
        q_d = a_d * self.Ub().T @ self.psib_d().T @ self.gammab_d(self.k) @ self.z
        q_b = a_t * self.Ub().T @ self.psib_b().T @ (self.gammab_b(self.k) @ self.z - self.p_t)
        q_bs = a_bs * self.Ub().T @ self.psib_s().T @ self.gammab_s() @ self.z
        return lg.solve(Q_d + Q_b + Q_w + Q_env + Q_bs, q_d + q_b + q_bs)

    def setWenv(self, w_env):
        self.w_env = w_env

    def psi(self, mic, hp, k):
        psi = np.zeros((self.Ip, self.Ip))
        for n1 in range(self.Ip):
            for n2 in range(self.Ip):
                if n1 >= n2:
                    psi[n1, n2] = self.h[k + n1, mic, hp, (n1-n2)]
        return psi

    def psib(self, mic, k):
        return np.block([self.psi(mic, hp, k) for hp in range(self.room.nbHP)])

    def psib_b(self):
        mic_b = np.where(np.array([x.type for x in self.room.mic]) == 'B')[0].tolist()
        return np.block([[self.psib(no_mic)] for no_mic in mic_b])

    def psib_d(self):
        mic_b = np.where(np.array([x.type for x in self.room.mic]) == 'D')[0].tolist()
        return np.block([[self.psib(no_mic)] for no_mic in mic_b])

    def gamma(self, mic, hp, k):
        gamma = np.zeros((self.Ip, self.nFir))
        for n in range(self.Ip):
            gamma[n, :] = self.h[k + n, mic, hp, :].T @ self.A**n
        return gamma

    def gammab(self, mic, k):
        return np.block([self.gamma(mic, hp, k) for hp in range(self.room.nbHP)])

    def gammab_b(self, k):
        mic_b = np.where(np.array([x.type for x in self.room.mic]) == 'B')[0].tolist()
        return np.block([self.gammab(no_mic, k) for no_mic in mic_b])

    def gammab_d(self, k):
        mic_b = np.where(np.array([x.type for x in self.room.mic]) == 'D')[0].tolist()
        return np.block([self.gammab(no_mic, k) for no_mic in mic_b])

    def ul(self, k):
        if k < self.nFir:
            return np.concatenate([np.flip(self.inp[:k]), np.zeros(self.nFir - k)])
        else:
            return np.flip(self.inp[k - self.nFir:k])

    def ur(self, k):
        if k > len(self.inp) - self.nFir:
            return np.concatenate([self.inp[k-1:], np.zeros(self.nFir - (len(self.inp) - k + 1))])
        else:
            return self.inp[k:k + self.nFir]

    def Ur(self):
        U = np.zeros((self.Ip, self.Ic * self.nFir))
        for n in range(self.Ic):
            U[n, n*self.nFir:(n+1)*self.nFir] = self.ul(self.k + n).T
        for n in range(self.Ic, self.Ip):
            U[n, (self.Ic-1) * self.nFir:self.Ic * self.nFir] = self.ul(self.k + n).T
        return U

    def Ub(self):
        U = np.zeros((self.room.nbHP * self.Ip, self.room.nbHP * self.Ic * self.nFir))
        for n in range(self.room.nbHP):
            U[n * self.Ip:(n+1) * self.Ip, n * self.Ic * self.nFir: (n+1) * self.Ic * self.nFir] = self.Ur()
        return U

if __name__ == "__main__":
    nFir = 10
    Fs = 4000

    room = Room.RoomLTV(nFir=nFir, Fs=Fs)
    MH = MH_SZ(room=room, Ic=5, Ip=10)

    room.addMicArray((0.5, 2.5), 0.1, 'B')
    room.addMicArray((4.5, 2.5), 0.1, 'D')
    room.addHP([(0, 0), (2.5, 0), (5, 0), (5, 2.5), (5, 5), (2.5, 5), (0, 5), (0, 2.5)])

    x = np.arange(1, 20)
    MH.setInput(x)

    MH.k = 10
    U=MH.Ub()
    print(U)
    plt.matshow(U)
    plt.show()