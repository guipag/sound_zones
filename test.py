from multiprocessing import Pool
from PressureMatching import *
from conv_py import *
import concurrent.futures
import time
from functools import partial

def conv_par(args, y):
    # with open('var_y_05.npy', 'rb') as f:
    #     y = np.load(f)
    # with open('var_h_05.npy', 'rb') as f:
    #     h = np.load(f)
    no_mic = args[0]
    h = args[1]
    return conv_LTV_MISO(h, y)


if __name__ == '__main__':
    t1 = time.perf_counter()
    with open('var_y_05.npy', 'rb') as f:
        y = np.load(f)
    with open('var_h_05.npy', 'rb') as f:
        h = np.load(f)
    # conv_p = partial(conv_par, y=y)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        z = np.array(list(executor.map(partial(conv_par, y=y), list(zip(range(8), [h[no_mic, :, :, :] for no_mic in range(8)])))))
    t2 = time.perf_counter()

    print(t2-t1)

    pass
