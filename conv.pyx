import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef double conv_LTV_k(double[:,:] h, double[:] x, int k):
    cdef double row = 0
    cdef Py_ssize_t ii
    cdef int len_sig = len(x)
    cdef int nFir = h.shape[0]

    for ii in range(len_sig):
        if 0 <= k - ii < nFir:
            row = row + h[k-ii, ii] * x[ii]
    return row

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef double[:] conv_LTV(double[:,:] h, double[:] x):
    cdef int len_sig = len(x)
    cdef double[:] sig = np.zeros(len_sig)
    cdef Py_ssize_t ii

    for ii in range(len_sig):
        sig[ii] = conv_LTV_k(h, x, ii)
    return sig

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef double[:,:] conv_LTV_MIMO(double[:,:,:] h, double[:,:] x):
    cdef Py_ssize_t[:] dim = h.shape
    cdef double[:,:] y = np.zeros_like(x)
    cdef Py_ssize_t no_hp

    for no_hp in range(dim[1]):
        y[no_hp, :] = conv_LTV(np.squeeze(h[:, no_hp, :]).T, x[no_hp, :])
    return y

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# cpdef double[:] conv_LTV_MISO(double[:,:,:] h, double[:,:] x):
#     cdef Py_ssize_t[:] dim = h.shape
#     cdef double[:] y = np.zeros(dim[0])
#     cdef Py_ssize_t no_hp
#
#     for no_hp in range(dim[1]):
#         y[:] = y[:] + conv_LTV(np.squeeze(h[:, no_hp, :]).T, x[no_hp, :])
#     return y


#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def conv_LTV_MIMO(double[:,:,:] h, double[:,:] x):
#     dim = h.shape
#     y = np.zeros_like(x)
#     for no_hp in range(dim[1]):
#         y[no_hp, :] = conv_LTV(np.squeeze(h[:, no_hp, :]), x[no_hp, :])
#     return y
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def conv_MIMO(double[:,:] h, double[:,:] x):
#     cdef int nbHP = h.shape[0]
#     cdef int nFir = h.shape[1]
#     cdef int len_sig = x.shape[1]
#     cdef double[:,:] sig = np.zeros((nbHP, nFir + len_sig - 1))
#     cdef int no_in, k, ii
#     for no_in in prange(nbHP, nogil=True):
#         for k in range(nFir):
#             for ii in range(len_sig):
#                 if 0 <= k - ii < nFir:
#                     sig[no_in, ii] = sig[no_in, ii] + h[no_in, k - ii] * x[no_in, ii]
#     return sig
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def conv_MISO(double[:,:,:] h, double[:,:] inp):
#     cdef int nbHP = h.shape[0]
#     cdef int nbMic = h.shape[1]
#     cdef int nFir = h.shape[2]
#     cdef int lenSig = inp.shape[1]
#     cdef double[:,:] out = np.zeros((nbMic, lenSig+nFir-1))
#     for i in range(nbMic):
#         for j in range(nbHP):
#             out[i, :] = out[i, :] + np.convolve(h[j, i, :], inp[j, :])
#     return out