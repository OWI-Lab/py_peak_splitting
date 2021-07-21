"""
Simulate Peak Splitting signals
---------------------------------
this script contains several functions which allow you to simulate different additive noise signals.
The main focus is on simulating additive noise signals which qualitatively resemble additive noise
due to peak splitting artefacts.

_ps1: randomized jumps when exceeding hard coded thresholds on absolute signal amplitudes
_ps2: randomized jumps when exceeding randomized thresholds on absolute signal amplitudes
_ps3: continous swithcing between peaks
_gn: additive gaussian white noise
_rw: additive (bounded) random walk
"""


""""""
import numpy as np

ERROR_TOLERANCE = 1e-6

"""
    Examples: how to generate signales which have peak splitting a-like artefacts.

    import numpy as np 
    import matplotlib.pyplot as plt

    # simple harmonic with n=3 components
    --------------------------------------------

    arr0 = dummy_signal(f=[0.2, 0.456, 0.32], Fs=Fs, duration=duration, amplitude=None, phi=None)
    scale = 0.5*np.max(np.abs(np.diff(arr)))

    # peak splitting a-like artefacts
    --------------------------------------------
   
    # forced jumps at exceeding absolute signal amplitudes:
    arr1 = arr0 + _ps1(arr=arr0, jump_v=[1, 1], jump_s=[0, 0], jot=[2, 4], scale=1, scale_by_amplitude=False)
    
    # forced jumps at exceeding thresholds, thresholds now include probabilistic criterium.
    arr2 = arr0 + _ps2(arr=arr0, jump_v=[1, 1], jump_s=[0,0], p=[0.75, 0.75], scale=1, jot=[2,4], scale_by_amplitude=False)
    
    # "continous flickering"
    arr3 = arr0 + _ps3(arr=arr0,  jump_v=[1], jump_s=[0], scale=1, jot=[2], scale_by_amplitude=True)
    
    # additive random walk and gaussian noise
    arr4 = arr0 + _gn(arr=arr0, var_val=0.1, as_factor=False)
    arr5 = arr0  + _rw(N=len(arr),LL=-2, UL=2, qstep=.05)
    arr4 = arr0 + _rw(N=len(arr), LL=-5, UL=5, qstep=0.1))
    arr5 = arr0 + _gn(arr=arr0, var_val=1, as_factor=False))

    # visualize the AWGN ARW: 
    --------------------------------------------         
    diff4 = np.hstack([0, np.diff(arr4)])
    diff5 = np.hstack([0, np.diff(arr5)])

    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.plot(t, arr4, c='b')
    ax.plot(t, arr5, c='r')
  
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.plot(t, diff4)
    ax.plot(t, diff5)
    
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.hist(diff4, bins=np.int(10*np.ptpt(arr4)))
    ax.hist(diff5, bins=np.int(10*np.ptpt(arr5)))
    plt.show()  


    # visualize the PS signals: 
    --------------------------------------------         
    diff0 = np.hstack([0, np.diff(arr0)])
    diff1 = np.hstack([0, np.diff(arr1)])
    diff2 = np.hstack([0, np.diff(arr2)])
    diff3 = np.hstack([0, np.diff(arr3)])

    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.plot(t, arr0, c='k')
    ax.plot(t, arr1)
    ax.plot(t, arr2)
    ax.plot(t, arr3)
  
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.plot(t, diff0, c='k')
    ax.plot(t, diff1)
    ax.plot(t, diff2)
    ax.plot(t, diff3)
    
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.hist(diff0, bins=np.int(10*np.ptpt(arr4)), color='k', alpha=0.25)
    ax.hist(diff1, bins=np.int(10*np.ptpt(arr5)), alpha=0.25)
    ax.hist(diff2, bins=np.int(10*np.ptpt(arr4)), alpha=0.25)
    ax.hist(diff3, bins=np.int(10*np.ptpt(arr5)), alpha=0.25)
    plt.show() 

    # visualize the AWGN ARW: 
    --------------------------------------------         
    diff4 = np.hstack([0, np.diff(arr4)])
    diff5 = np.hstack([0, np.diff(arr5)])

    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.plot(t, arr0, c='k')
    ax.plot(t, arr4, c='b')
    ax.plot(t, arr5, c='r')
  
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.plot(t, diff4)
    ax.plot(t, diff5)
    
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.hist(diff4, bins=np.int(10*np.ptpt(arr4)), alpha=0.25)
    ax.hist(diff5, bins=np.int(10*np.ptpt(arr5)), alpha=0.25)
    plt.show()    
    
    --------------------------------------------    
    
    
"""


def default_signal(Fs=50, duration=20):
    arr0 = dummy_signal(f=[0.2, 0.456, 0.32], Fs=Fs, duration=duration, amplitude=None, phi=None)
    scale = 1.25 * np.max(np.abs(np.diff(arr0)))
    t = np.linspace(0, duration - 1 / Fs, Fs * duration)
    # TODO unresolved reference to _ps7
    arr = arr0 + _ps7(arr=arr0, jump_v=[1, 1], jump_s=[0, 0], p=[0.97, 0.97], scale=1, jot=[2, 4],
                      scale_by_amplitude=False)
    return arr, t, arr0


def dummy_signal(f=[0.2, 0.456, 0.32], Fs=50, duration=600, amplitude=None, phi=None):
    """
    :param f: list containing N frequencies 
    :param Fs: sampling frequency
    :param duration: duration of signal in seconds
    :param amplitude: list containing N amplitudes 
    :param phi: list containing N phase angles
    :return:
    """
    if not isinstance(f, (list, np.ndarray)):
        f = [f]
    if phi is None:
        phi = list(2 * np.pi * np.random.rand(len(f)))
    if amplitude is None:
        amplitude = list(0.75 + 0.5 * 2 * np.pi * np.random.rand(len(f)))

    t = np.linspace(0, duration - 1 / Fs, int(Fs * duration))
    sinoid = np.zeros(len(t))
    for a, freq, p in zip(amplitude, f, phi):
        sinoid += a * np.sin(t * 2 * np.pi * freq + p)
    return sinoid


def _ps1(arr, jump_v=[1, 1], jump_s=[0, 0], jot=[2, 4], scale=1, scale_by_amplitude=False):
    """
    Forced jumps with stochastic component, at exceeding predefined thresholds.
    :param arr: noise free input signal contained in 1-d np.ndarray
    :param jump_v: list of M determinstic jump values
    :param jump_s: list of M standard deviations to randomize jump amplitude
    :param scale: scale factor for jump amplitudes (deprecated)
    :param jot: list of M jump occurence thresholds
    :param scale_by_amplitude: Boolen to make jump levels depend on signal amplitude
    :return x:  additive noise contribution due to peak splitting  
    """
    # scale jumps
    jump_v = [scale * j for j in jump_v]

    # to make jumps amplitude dependent - does affect signs if not absolute.
    N = len(arr)
    arr = arr - np.mean(arr)
    asf = np.sign(arr)
    asf = np.where(arr >= 0, arr / np.max(arr), arr / np.abs(np.min(arr)))
    asf = arr / np.ptp(arr)

    # preallocate
    x = list()
    for jv, js, jt in zip(jump_v, jump_s, jot):
        sign = 1
        ra = np.random.rand(len(arr))
        for ri, sf, av in zip(ra, asf, arr):

            if np.abs(av) > jt and sign == 1:
                if sign == 1:
                    if scale_by_amplitude is True:
                        amp = sf * jv + js * np.random.randn()
                    elif scale_by_amplitude is False:
                        amp = np.sign(sf) * jv + js * np.random.randn()
                x.append(sign * amp)
                sign = -1 * sign
            elif np.abs(av) <= jt and sign == -1:
                x.append(sign * amp)
                sign = -1 * sign
            else:
                x.append(0)

    x = np.asarray(x)
    x = np.sum(np.reshape(x, (len(jump_v), N)), axis=0)
    x = np.cumsum(x)
    return x


def _ps2(arr, jump_v=[1, 2], jump_s=[0, 0], p=[0.15, 0.15], scale=1, jot=[1, 2], scale_by_amplitude=False):
    """
    Forced jumps with stochastic component, at exceeding predefined thresholds - threholds are a rnadom variable now,
    which causes some flickering around thresholds.
    :param arr: noise free input signal contained in 1-d np.ndarray
    :param jump_v: list of M determinstic jump values
    :param jump_s: list of M standard deviations to randomize jump amplitude
    :param jot: list of M jump occurence thresholds; threshold on absolute signal deciding whether a jump has to occur. 
    :param p: list of M standard deviations which are used to randomize the jot (jump occurence thresholds)
    :param scale: scale factor for jump amplitudes (deprecated)
    :param scale_by_amplitude: Boolen to make jump levels depend on signal amplitude
    :return x:  additive noise contribution due to peak splitting  
    """
    # scale jumps
    jump_v = [scale * j for j in jump_v]

    # to make jumps amplitude dependent
    N = len(arr)
    arr = arr - np.mean(arr)

    # different amplitude scale factors - also to derive sign 
    asf = np.sign(arr)
    asf = np.where(arr >= 0, arr / np.max(arr), arr / np.abs(np.min(arr)))
    asf = arr / np.ptp(arr)

    # preallocate
    x = list()
    for jv, js, jp, jt in zip(jump_v, jump_s, p, jot):
        sign = 1
        ra = np.random.rand(len(arr))
        for ri, sf, av in zip(ra, asf, arr):

            if np.abs(av) > jt + jp * np.random.randn() and sign == 1 and ri > jp:
                if sign == 1:
                    if scale_by_amplitude is True:
                        amp = sf * jv + js * np.random.randn()
                    elif scale_by_amplitude is False:
                        amp = np.sign(sf) * jv + js * np.random.randn()
                x.append(sign * amp)
                sign = -1 * sign
            elif np.abs(av) <= jt + jp * np.random.randn() and sign == -1 and ri > jp:
                x.append(sign * amp)
                sign = -1 * sign
            else:
                x.append(0)

    x = np.asarray(x)
    x = np.sum(np.reshape(x, (len(jump_v), N)), axis=0)
    x = np.cumsum(x)
    return x


def _ps3(arr, jump_v=[1], jump_s=[0], scale=1, jot=[1], scale_by_amplitude=True):
    """
    Some kind of continous flickering, back and forth peak-swithcing - upon exceeding amplitude thresholds.
    :param arr: noise free input signal contained in 1-d np.ndarray
    :param jump_v: list of M determinstic jump values
    :param jump_s: list of M standard deviations to randomize jump amplitude
    :param scale: scale factor for jump amplitudes (deprecated)
    :param jot: list of M jump occurence thresholds; threshold on absolute signal deciding whether a jump has to occur. 
    :param scale_by_amplitude: Boolen to make jump levels depend on signal amplitude
    :return x:  additive noise contribution due to peak splitting
    """
    # scale jumps
    jump_v = [scale * j for j in jump_v]

    # to make jumps amplitude dependent - does affect signs if not absolute.
    N = len(arr)
    arr = arr - np.mean(arr)
    asf = np.sign(arr)

    # preallocate
    x = list()
    for jv, js, jt in zip(jump_v, jump_s, jot):
        sign = 1
        ra = np.random.rand(len(arr))
        for ri, sf, av in zip(ra, asf, arr):
            if np.abs(av) > jt:
                if sign == 1:
                    if scale_by_amplitude is True:
                        amp = np.abs(1 - (jt - av)/jt) * sf*jv + js*np.random.randn()
                    else:
                        amp = sf*jv + js*np.random.randn()
                x.append(sign * amp)
                sign = -1 * sign
            elif np.abs(av) <= jt and sign== -1:
                x.append(sign * amp)
                sign = -1 * sign
            else:
                x.append(0)

    x = np.asarray(x)
    x = np.sum(np.reshape(x, (len(jump_v), N)), axis=0)
    x = np.cumsum(x)
    return x


def _gn(arr, var_val=1, as_factor=False):
    """" add additive white gaussian noise (AWGN) to a time series contained in arr. the AWGN as defined by the noise covariance matrix.
    :param var_val: if float: specify variance value - constant for every sensor. if list: entries for diagonal covariance matrix
    if array: full covariance matrix
    :param as_factor: specify "noise variance" as ratio of signal variance
    """
    if arr.ndim == 1:
        arr = np.atleast_2d(arr)
        flatten_output = True

    # how should covmat be computed:
    if as_factor is True:
        cov_mat = var_val * np.var(arr, axis=1)
    else:
        cov_mat = var_val * np.eye(np.min(arr.shape))

    # random array
    n = np.random.randn(arr.shape[0], arr.shape[1])
    if n.shape[0] > n.shape[1]:
        n = n.T
    # sample std. to exactly one
    m = np.atleast_2d(np.std(n, axis=1)).T
    n = n / m

    # multiply n wit square root of covariance matrix to obtain target covariances:
    std_mat = np.sqrt(cov_mat)
    n = std_mat.dot(n)

    if flatten_output:
        n = n.flatten()
        arr = arr.flatten()

    return n


def _rw(N, LL=-5, UL=5, qstep=0.1):
    """ add additive random walk (ARW) of lenght N, constrained by a lower (LL) and upper bound (UL). 
    :param N: number of samples of the ARW to be generated
    :param LL: lower bound for random walk 
    :param UL: upper bound for random walk 
    :param qstep: standard deviation of random step
    :return:
    """
    qs_step = 0.1
    u = list()
    for i in range(N):

        # build time random walk
        load = qstep * np.random.randn(1)[0]
        if i == 0:
            # u.append(np.random.randint(low=LL - 0.1, high=UL - 0.1))
            u.append(0)
        else:
            add = u[i - 1] + load
            # reverse last step if boundary has been exceeded
            if add > UL:
                add = add - np.abs(2 * load)
            elif add < LL:
                add = u[i - 1] + np.abs(2 * load)
            u.append(add)
    return np.asarray(u)





