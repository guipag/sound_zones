import numpy as np
from conv import conv_LTV_k

# def conv_LTV_k(h, x, k):
#     row = 0
#
#     for ii in range(len(x)):
#         if 0 <= k - ii < h.shape[0]:
#             row += h[k-ii,  ii] * x[ii]
#     return row


def conv_LTV(h, x):
    sig = np.zeros(len(x))
    h = np.ascontiguousarray(h)
    x = np.ascontiguousarray(x)
    for ii in range(len(x)):
        sig[ii] = conv_LTV_k(h, x, ii)
    return sig


def conv_LTV_MIMO(h, x):
    dim = h.shape
    y = np.zeros_like(x)
    for no_hp in range(dim[1]):
        y[no_hp, :] = conv_LTV(np.squeeze(h[:, no_hp, :]).T, x[no_hp, :])
    return y

def conv_LTV_MIMO_par(args):
    h, x = args[0], args[1]
    return conv_LTV(h.T, x)

def conv_LTV_MISO(h, x):
    dim = h.shape
    y = np.zeros(dim[0])
    for no_hp in range(dim[1]):
        y += conv_LTV(np.squeeze(h[:, no_hp, :]).T, x[no_hp, :])
    return y

def conv_par(args, y):
    h = args[1]
    return conv_LTV_MISO(h, y)