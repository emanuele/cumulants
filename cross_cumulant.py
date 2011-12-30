"""Compute the cumulant of a vector process.
"""

import numpy as np
import itertools

def combinations_with_replacement(iterable, r):
    """From
    http://docs.python.org/library/itertools.html#itertools.combinations_with_replacement
    Useful for Python previous than v2.7 where
    itertools.combinations_with_replacement() is not available.
    """
    # combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
    pool = tuple(iterable)
    n = len(pool)
    if not n and r:
        return
    indices = [0] * r
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != n - 1:
                break
        else:
            return
        indices[i:] = [indices[i] + 1] * (r - i)
        yield tuple(pool[i] for i in indices)


def combinations_with_replacement2(iterable, r):
    """From
    http://docs.python.org/library/itertools.html#itertools.combinations_with_replacement
    Slightly slower than combinations_with_replacement().

    Useful for Python previous than v2.7 where
    itertools.combinations_with_replacement is not available.
    """
    pool = tuple(iterable)
    n = len(pool)
    for indices in itertools.product(range(n), repeat=r):
        if sorted(indices) == list(indices):
            yield tuple(pool[i] for i in indices)


def compute_cumulant2(data):
    """Compute the second order cross-cumulant with lag 0. This is
    the straightforward implementation, highly inefficient.

    data = channels x time.
    """
    channels = data.shape[0]
    tensor = np.zeros((channels, channels))
    for ch0 in range(channels):
        for ch1 in range(channels):
            tensor[ch0, ch1] = np.mean(data[ch0, :] * data[ch1, :])
    return tensor

def compute_cumulant3(data):
    """Compute the third order cross-cumulant with lags 0. This is
    the straightforward implementation, highly inefficient.

    data = channels x time.
    """
    channels = data.shape[0]
    tensor = np.zeros((channels, channels, channels))
    for ch0 in range(channels):
        for ch1 in range(channels):
            for ch2 in range(channels):
                tensor[ch0, ch1, ch2] = np.mean(data[ch0, :] * data[ch1, :] * data[ch2, :])
    return tensor
        
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


def compute_cumulant4_faster(data):
    """Compute the fourth order cross-cumulant with lags 0. This is
    the straightforward implementation, highly inefficient.

    Speed-up strategy: use iterators and itertools to explore
    combinations with replacement and compute expectations without
    repetitions. Then copy the results to all entries within a
    permutation of the indices.

    data = channels x time.
    """
    channels = data.shape[0]
    tensor = np.zeros((channels, channels, channels, channels))
    E = np.zeros((channels, channels)) + np.inf
    for ch0, ch1, ch2, ch3 in combinations_with_replacement(range(channels), 4):
        for a, b in itertools.combinations([ch0, ch1, ch2, ch3], 2):
            # compute only necessary E[x_a*x_b] and keep results for future use:
            if E[a, b]==np.inf:
                E[a, b] = np.mean(data[a, :] * data[b, :])
                E[b, a] = E[a, b]
        cumulant = np.mean(data[ch0, :] * data[ch1, :] * data[ch2, :] * data[ch3, :])  - E[ch0, ch1] * E[ch2, ch3] - E[ch0, ch2] * E[ch1, ch3] - E[ch0, ch3] * E[ch1, ch2]
        for c0,c1,c2,c3 in itertools.permutations([ch0, ch1, ch2, ch3]):
            tensor[c0, c1, c2, c3] = cumulant
    return tensor


def compute_cumulant4_faster_plain(data):
    """Compute the fourth order cross-cumulant with lags 0. This
    implementation is analogous to compute_cumulant4_faster but it
    does not use smart iterators but only plain for loops and explicit
    unrolling of the permutations. This implementation is meant to be
    easily ported to Cython.

    data = channels x time.
    """
    channels = data.shape[0]
    tensor = np.zeros((channels, channels, channels, channels))
    E = np.zeros((channels, channels)) + np.inf
    for ch0 in range(channels):
        for ch1 in range(ch0, channels):
            if E[ch0, ch1]==np.inf:
                E[ch0, ch1] = np.mean(data[ch0, :] * data[ch1, :])
                E[ch1, ch0] = E[ch0, ch1]
            for ch2 in range(ch1, channels):
                if E[ch0, ch2]==np.inf:
                    E[ch0, ch2] = np.mean(data[ch0, :] * data[ch2, :])
                    E[ch2, ch0] = E[ch0, ch2]
                if E[ch1, ch2]==np.inf:
                    E[ch1, ch2] = np.mean(data[ch1, :] * data[ch2, :])
                    E[ch2, ch1] = E[ch1, ch2]
                for ch3 in range(ch2, channels):
                    if E[ch0, ch3]==np.inf:
                        E[ch0, ch3] = np.mean(data[ch0, :] * data[ch3, :])
                        E[ch3, ch0] = E[ch0, ch3]
                    if E[ch1, ch3]==np.inf:
                        E[ch1, ch3] = np.mean(data[ch1, :] * data[ch3, :])
                        E[ch3, ch1] = E[ch1, ch3]
                    if E[ch2, ch3]==np.inf:
                        E[ch2, ch3] = np.mean(data[ch2, :] * data[ch3, :])
                        E[ch3, ch2] = E[ch2, ch3]
                    cumulant = np.mean(data[ch0, :] * data[ch1, :] * data[ch2, :] * data[ch3, :]) - E[ch0, ch1] * E[ch2, ch3] - E[ch0, ch2] * E[ch1, ch3] - E[ch0, ch3] * E[ch1, ch2]

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


def generate_data(num_channels=10, sample_rate_Hz=20, t_min=-0.5, t_max=2.5, noise_level=1.0):
    """Simple generator of multivariate timeserie. For testing purpose.
    """
    # sample_rate_Hz = 20 # sample rate of recordings
    # t_min = -0.5 # The beginning of each trial: 500ms before stimulus offset
    # t_max = 2.5 # The end of each trial: 2500ms after stimulus offset
    t = np.linspace(t_min, t_max, (t_max-t_min)*sample_rate_Hz+1) # timepoints of one trial
    # num_channels = 10
    # noise_level = 1.0
    components = [[1.0,0.5,0.0], [0.2,2.0,0.1], [0.2,5.0,0.0], [0.5,11.0,0.8]]
    x = np.zeros(len(t))
    for amplitude, frequency_hz, phase_rad in components:
        x += amplitude*np.sin(2.0*np.pi*frequency_hz*t+phase_rad)        
    # plt.figure()
    # plt.plot(t, x)
    data = np.ones((num_channels, t.size)) * x + noise_level * np.random.randn(num_channels, t.size)
    # plt.figure()
    # plt.plot(t, data.T)
    return data


if __name__=='__main__':

    import time

    try:
        import cross_cumulant_cy as cy
        cython = True
    except ImportError:
        cython = False

    num_channels = 10
    data = generate_data(num_channels=num_channels)
    
    t0 = time.time()
    tensor2 = compute_cumulant2(data)
    t = time.time()
    print "tensor2:", tensor2.shape, t-t0, "sec."
    t0 = time.time()
    tensor3 = compute_cumulant3(data)
    t = time.time()
    t = time.time()
    print "tensor3:", tensor3.shape, t-t0, "sec."    
    t0 = time.time()
    tensor4 = compute_cumulant4(data)
    t = time.time()
    print "tensor4:", tensor4.shape, t-t0, "sec."
    t0 = time.time()
    tensor4_fast = compute_cumulant4_fast(data)
    t = time.time()
    print "tensor4 fast:", tensor4_fast.shape, t-t0, "sec."
    t0 = time.time()
    tensor4_faster = compute_cumulant4_faster(data)
    t = time.time()
    print "tensor4 faster:", tensor4_faster.shape, t-t0, "sec."
    t0 = time.time()
    tensor4_faster_plain = compute_cumulant4_faster_plain(data)
    t = time.time()
    print "tensor4 faster plain:", tensor4_faster_plain.shape, t-t0, "sec."
    if cython:
        t0 = time.time()
        tensor4_faster_plain_cy = cy.compute_cumulant4_faster_plain(data)
        t = time.time()
        print "tensor4 faster plain cython:", tensor4_faster_plain_cy.shape, t-t0, "sec."

    print "tensor4 == tensor4_fast :",
    try:
        np.testing.assert_almost_equal(tensor4, tensor4_fast, decimal=10)
        print True
    except AssertionError:
        print False

    print "tensor4 == tensor4_faster :",
    try:
        np.testing.assert_almost_equal(tensor4, tensor4_faster, decimal=10)
        print True
    except AssertionError:
        print False

    print "tensor4 == tensor4_faster_plain :",
    try:
        np.testing.assert_almost_equal(tensor4, tensor4_faster_plain, decimal=10)
        print True
    except AssertionError:
        print False

    if cython:
        print "tensor4 == tensor4_faster_plain_cy :",
        try:
            np.testing.assert_almost_equal(tensor4, tensor4_faster_plain_cy, decimal=10)
            print True
        except AssertionError:
            print False
    
