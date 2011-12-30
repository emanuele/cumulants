import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.double
ctypedef np.double_t DTYPE_t


def compute_cumulant4(data):
    """Compute the fourth order cross-cumulant with lags 0. This is
    the straightforward implementation, highly inefficient.

    data = channels x time.
    """
    channels = data.shape[0]
    tensor = np.zeros((channels, channels, channels, channels))
    for ch0 in range(channels):
        for ch1 in range(channels):
            E01 = np.mean(data[ch0, :] * data[ch1, :])
            for ch2 in range(channels):
                E02 = np.mean(data[ch0, :] * data[ch2, :])
                E12 = np.mean(data[ch1, :] * data[ch2, :])
                for ch3 in range(channels):
                    E03 = np.mean(data[ch0, :] * data[ch3, :])
                    E13 = np.mean(data[ch1, :] * data[ch3, :])
                    E23 = np.mean(data[ch2, :] * data[ch3, :])
                    tensor[ch0, ch1, ch2, ch3] = np.mean(data[ch0, :] * data[ch1, :] * data[ch2, :] * data[ch3, :]) - E01 * E23 - E02 * E13 - E03 * E12
    return tensor

def compute_cumulant4_fast(data):
    """Compute the fourth order cross-cumulant with lags 0. This is
    the straightforward implementation, highly inefficient.

    Speed-up strategy: compute only once entries with differs only by
    a permutation or indices.

    data = channels x time.
    """
    channels = data.shape[0]
    tensor = np.zeros((channels, channels, channels, channels))+np.inf
    for ch0 in range(channels):
        for ch1 in range(channels):
            E01 = np.mean(data[ch0, :] * data[ch1, :])
            for ch2 in range(channels):
                E02 = np.mean(data[ch0, :] * data[ch2, :])
                E12 = np.mean(data[ch1, :] * data[ch2, :])
                for ch3 in range(channels):
                    chs_sorted = np.sort([ch0, ch1, ch2, ch3])
                    if tensor[chs_sorted[0], chs_sorted[1], chs_sorted[2], chs_sorted[3]]!=np.inf :
                        tensor[ch0, ch1, ch2, ch3] = tensor[chs_sorted[0], chs_sorted[1], chs_sorted[2], chs_sorted[3]]
                        continue
                    E03 = np.mean(data[ch0, :] * data[ch3, :])
                    E13 = np.mean(data[ch1, :] * data[ch3, :])
                    E23 = np.mean(data[ch2, :] * data[ch3, :])
                    tensor[ch0, ch1, ch2, ch3] = np.mean(data[ch0, :] * data[ch1, :] * data[ch2, :] * data[ch3, :]) - E01 * E23 - E02 * E13 - E03 * E12
    return tensor

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_cumulant4_faster_plain(np.ndarray[DTYPE_t, ndim=2] data):
    """Compute the fourth order cross-cumulant with lags 0. This is
    the straightforward implementation, highly inefficient.

    data = channels x time.
    """
    cdef Py_ssize_t channels = data.shape[0]
    cdef Py_ssize_t timepoints = data.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=4] tensor = np.zeros((channels, channels, channels, channels))
    cdef np.ndarray[DTYPE_t, ndim=2] E = np.zeros((channels, channels)) + np.inf
    cdef Py_ssize_t ch0, ch1, ch2, ch3, i
    cdef DTYPE_t cumulant
    for ch0 in range(channels):
        for ch1 in range(ch0, channels):
            if E[ch0, ch1]==np.inf:
                # E[ch0, ch1] = np.mean(data[ch0, :] * data[ch1, :])
                E[ch0, ch1] = 0.0
                for i in range(timepoints):
                    E[ch0, ch1] += data[ch0, i] * data[ch1, i]
                E[ch0, ch1] /= timepoints
                E[ch1, ch0] = E[ch0, ch1]
            for ch2 in range(ch1, channels):
                if E[ch0, ch2]==np.inf:
                    # E[ch0, ch2] = np.mean(data[ch0, :] * data[ch2, :])
                    E[ch0, ch2] = 0.0
                    for i in range(timepoints):
                        E[ch0, ch2] += data[ch0, i] * data[ch2, i]
                    E[ch0, ch2] /= timepoints
                    E[ch2, ch0] = E[ch0, ch2]
                if E[ch1, ch2]==np.inf:
                    # E[ch1, ch2] = np.mean(data[ch1, :] * data[ch2, :])
                    E[ch1, ch2] = 0.0
                    for i in range(timepoints):
                        E[ch1, ch2] += data[ch1, i] * data[ch2, i]
                    E[ch1, ch2] /= timepoints
                    E[ch2, ch1] = E[ch1, ch2]
                for ch3 in range(ch2, channels):
                    if E[ch0, ch3]==np.inf:
                        # E[ch0, ch3] = np.mean(data[ch0, :] * data[ch3, :])
                        E[ch0, ch3] = 0.0
                        for i in range(timepoints):
                            E[ch0, ch3] += data[ch0, i] * data[ch3, i]
                        E[ch0, ch3] /= timepoints
                        E[ch3, ch0] = E[ch0, ch3]
                    if E[ch1, ch3]==np.inf:
                        # E[ch1, ch3] = np.mean(data[ch1, :] * data[ch3, :])
                        E[ch1, ch3] = 0.0
                        for i in range(timepoints):
                            E[ch1, ch3] += data[ch1, i] * data[ch3, i]
                        E[ch1, ch3] /= timepoints
                        E[ch3, ch1] = E[ch1, ch3]
                    if E[ch2, ch3]==np.inf:
                        # E[ch2, ch3] = np.mean(data[ch2, :] * data[ch3, :])
                        E[ch2, ch3] = 0.0
                        for i in range(timepoints):
                            E[ch2, ch3] += data[ch2, i] * data[ch3, i]
                        E[ch2, ch3] /= timepoints
                        E[ch3, ch2] = E[ch2, ch3]

                    cumulant = 0.0
                    for i in range(timepoints):
                        cumulant += data[ch0, i] * data[ch1, i] * data[ch2, i] * data[ch3, i]
                    cumulant /= timepoints
                    cumulant += - E[ch0, ch1] * E[ch2, ch3] - E[ch0, ch2] * E[ch1, ch3] - E[ch0, ch3] * E[ch1, ch2]
                    # cumulant = np.mean(data[ch0, :] * data[ch1, :] * data[ch2, :] * data[ch3, :]) - E[ch0, ch1] * E[ch2, ch3] - E[ch0, ch2] * E[ch1, ch3] - E[ch0, ch3] * E[ch1, ch2]

                    tensor[ch0, ch1, ch2, ch3] = cumulant
                    tensor[ch0, ch1, ch3, ch2] = cumulant
                    tensor[ch0, ch2, ch1, ch3] = cumulant
                    tensor[ch0, ch2, ch3, ch1] = cumulant
                    tensor[ch0, ch3, ch1, ch2] = cumulant
                    tensor[ch0, ch3, ch2, ch1] = cumulant
                    
                    tensor[ch1, ch0, ch2, ch3] = cumulant
                    tensor[ch1, ch0, ch3, ch2] = cumulant
                    tensor[ch1, ch2, ch0, ch3] = cumulant
                    tensor[ch1, ch2, ch3, ch0] = cumulant
                    tensor[ch1, ch3, ch0, ch2] = cumulant
                    tensor[ch1, ch3, ch2, ch0] = cumulant
                    
                    tensor[ch2, ch1, ch0, ch3] = cumulant
                    tensor[ch2, ch1, ch3, ch0] = cumulant
                    tensor[ch2, ch0, ch1, ch3] = cumulant
                    tensor[ch2, ch0, ch3, ch1] = cumulant
                    tensor[ch2, ch3, ch1, ch0] = cumulant
                    tensor[ch2, ch3, ch0, ch1] = cumulant
                    
                    tensor[ch3, ch1, ch2, ch0] = cumulant
                    tensor[ch3, ch1, ch0, ch2] = cumulant
                    tensor[ch3, ch2, ch1, ch0] = cumulant
                    tensor[ch3, ch2, ch0, ch1] = cumulant
                    tensor[ch3, ch0, ch1, ch2] = cumulant
                    tensor[ch3, ch0, ch2, ch1] = cumulant
    return tensor
