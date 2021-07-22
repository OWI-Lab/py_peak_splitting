"""This module contains an implementation of a decision filter based random telegraph noise removal tool; In which a threshold based 
decision filter is used for the purpose of segment wise outlier detection in the sample-wise difference signal corresponding 
to the corrupted raw signal. The threshold used for outlier detection is derived from the histogram of the difference signal.
Subsequently each detected outlier is replaced using a regression method applied on the sliding window method. Once the detection 
and replacement are completed, the cumulative sum of the de-noised difference signal is computed to arrive at a de-noised
realisation of the noisy time series.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

class ThFilter(object):
    """The ThFilter class implements a threshold filter for detecting outliers in the sample wise difference signal in 
    combination with a regression method for replacing detected outliers by estimates; The sample wise difference signal 
    is computed from the noisy input data; the cumulative sum of the de-noised difference signal is computed to arrive 
    at a de-noised realisation of the corrupted input array.

    :param arr: A number of Raw signals containing jumps stored in an array of shape (n_signals x n_samples). 
    :type arr: Numpy.ndarray.
    """

    def __init__(self, arr):
        """Constructor method
        """
        # instantiate object attributes and ensure that array is two dimensionsal
        self.arr = np.atleast_2d(arr)
        self.labels = []
        self.clusters = []
        self.counts = []

    def sig_ptp(self, order=0):
        """Peak to peak amplitude of provided raw signals, or a differenced version of those signal; The order of the sample
        wise differencing is set using the order parameter.

        :param order: order of sample wise differencing operation
        :type order: int
        :return: list of peak to peak values from self.arr
        :rtype: list
        """
        rng_lst = list()
        for arr in np.atleast_2d(self.arr):
            if order > 0:
                arr = np.diff(arr, n=order)
            rng_lst.append(np.ptp(arr))
        return rng_lst
    
    def estimate_th(self, arr=None, th=500, bin_density=10, margin=1, centric=True, aggressive=False, plot=False, kde=False, yscale='log', lhb=0.2):
        """Method to estimate the threshold setting for the outlier detection. The threshold estimates are derived from 
        the histogram of the difference signal(s) as: the average of the absoute bin centers (positive and negative) for
        which the histogram count first drops below the value specified with 'th';when aggressive is set to True the 
        value of 'th' will effectively be reduced to 0.

        :param arr: (Optional) noisy input array containing peak splitting / RTN artefacts; defaults to self.arr. 
        :type arr: numpy.ndarray, with shape (n_signals, n_smaples)
        :param th: Threshold value for determining historgram bin counts which are used to estimate the domain
        :type th: float
        :param bin_density: Number of bins per unit - histograms will be created with identical bin density
        :type bin_density: int
        :param margin: Scale factor which can be applies on the final th estimates
        :type margin: float 
        :param centric: True: Scan bin centers on matching threshold conditions for negative and positive axis separately
        :type centric: Bool
        :param aggressive: Replace threshold values by zero; Only recommended if the jumps are significanlty larger than the nominal difference samples.
        :type aggressive: Bool
        :param plot: Option to plot the histogram with corresponding threshold values for each signal
        :type plot: Bool
        :param kde: Option to overlay histogram with kernel density estimate - default bandwidth is set to 0.1.
        :type kde: Bool
        :param yscale: Parameter controlling the default scaling of the y-axis. 
        :type yscale: str
        :param lhb: Optional parameter when aggressive=True: provides lower search bound on histrogram bin centers. 
        :type lhb: float
        :return: list of derived threshold levels 
        :rtype: list
        """
        if arr is None:
            arr = self.arr
        th_lst = list()
        for _arr in np.atleast_2d(arr):

            # Low level call to estimate th from single array:
            _arr = np.hstack([0, np.diff(_arr)])
            nbins = np.int(np.floor(np.ptp(_arr) * bin_density))
            hist, bin_edges = np.histogram(a=_arr, bins=nbins)
            bin_centers = np.asarray([(bin_edges[k] + bin_edges[k + 1]) / 2 for k in range(len(bin_edges) - 1)])

            # simple rules to determine upper and lower th: use average as output
            if centric is True:
                upper = bin_centers[(bin_centers >= 0) & (hist < th) & (hist != 0)][0]
                lower = bin_centers[(bin_centers < 0) & (hist < th) & (hist != 0)][-1]
                th_est = margin * 0.5 * np.ptp([lower, upper])
            elif centric is False:
                th_est = margin * 0.5 * np.ptp(bin_centers[hist > th])

            if aggressive is True:
                # aim for first zeros outside [-0.2,0.2] and select largest bin center as th:
                upper = bin_centers[(bin_centers >= lhb) & (hist == 0)]
                lower = bin_centers[(bin_centers < -lhb) & (hist == 0)]
                upper = upper[0] if len(upper) > 0 else 1E12
                lower = lower[-1] if len(lower) > 0 else 1E12
                th_est = margin * np.max(np.abs([lower, upper]))

            # option to visualise the result per signal
            if plot is True:
                fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                _arr_1 = np.copy(_arr)
                if kde is False:
                    ax.hist(_arr_1, np.int(np.floor(np.ptp(_arr_1) * bin_density)))
                elif kde is True:
                    ax.hist(_arr_1, np.int(np.floor(np.ptp(_arr_1) * bin_density)), density=True)
                    kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
                    kde.fit(_arr_1.reshape(-1, 1))
                    _arr_1_x = np.linspace(np.min(_arr_1), np.max(_arr_1), np.int(np.floor(np.ptp(_arr_1) * bin_density)))
                    logprob = kde.score_samples(_arr_1_x.reshape(-1, 1))
                    ax.fill_between(_arr_1_x, np.exp(logprob), color='g', alpha=0.65)
                ax.axvline(-th_est, lw=3, c='r')
                ax.axvline(+th_est, lw=3, c='r')
                if yscale == 'log':
                    plt.yscale('log')

        th_lst.append(th_est)
        return th_lst



    def estimate_oocs(self, arr=None, th=500, bin_density=10, margin=1, centric=True, aggressive=False):
        """Method to estimate the out of cluster score, based on the default threshold estimates. Use in combination 
         with _sig_ptp and a histogram plot to assess severity of peak splitting.

        :param arr: (Optional) noisy input array containing peak splitting / RTN artefacts; defaults to self.arr. 
        :type arr: npumpy.ndarra, with shape (n_signals, n_smaples)
        :param th: threshold value for determining historgram bin counts which are used to estimate the domain
        :type th: float
        :param bin_density: number of bins per unit - histograms will be created with identical bin density
        :type bin_density: int
        :param margin: scale factor which can be applies on the final th estimates
        :type margin: float 
        :param centric: True: start from center of histogram and take th values as the bin centers at which the histogram count first drops below the th on bin counts.
        :type centric: Bool
        :param aggressive: in addition to the above, try a more aggressive approach and replace th values if the aggressive th estimates are not larger than 1.25 of the default estimates.
        :type aggressive: Bool
        :return: list of scores
        :rtype: list
        """
        if arr is None:
            arr = self.arr

        ths = list()
        for _arr in arr:
            ths.append(self.estimate_th(_arr, th=th, bin_density=bin_density, margin=margin, centric=centric,
                                        aggressive=aggressive))

        n = np.max(arr.shape)
        scores = list()
        for sig, thv in zip(arr, ths):
            arr = np.copy(sig.data)
            ads = np.abs(np.diff(arr))
            scores.append(np.count_nonzero(ads > thv) / n)
        return scores

    def _recon_th(self, arr, time, th=0.5, nbuffer=25, method='median', settings=None):
        """ Method to perform the reconstruction for an individual signal/array.

        :param arr: noisy input array containing peak splitting / RTN artefacts.
        :type arr: npumpy.ndarra, with shape (1, n_smaples)
        :param time: time array or pseudo time array containgin sample indices
        :type time: numpy.ndarray
        :param th: threshold level for outlier detection
        :type th: float
        :param nbuffer: Number of samples in data buffer which is used to create outlier replacements
        :type nbuffer: int
        :param method: string to select outlier replacement method 'polyfit', 'mean', 'median', 'random'
        :type method: str
        :param settings: settings dictionary for np.polyfit command.
        :type settings: dict
        :return: array holding the reconstructed signal, list containing the labels, list containing clusters
        :rtype: numpy.ndarray
        """
        
        # define default settings for polyfit - ensure full is False. all others are good.
        if settings is None and method == 'polyfit':
            settings = dict(deg=3, full=False, w=None, cov=False)
        elif settings is not None and method == 'polyfit':
            settings.update(full=False)
           
        # difference data

        arr_1 = np.hstack([0, np.diff(np.copy(arr))])
        #arr_1 = np.diff(np.copy(arr), prepend=0)
        N = len(arr_1)

        # outlier indices: for difference and time domain signal
        idx_all = np.arange(0, np.max(arr_1.shape))
        idx_in = idx_all[np.abs(arr_1) < th]
        idx = idx_all[np.abs(arr_1) > th]

        recon_dd = np.copy(arr_1)
        for c, i in enumerate(idx):

            # setup adaptive buffer and validate: BUG 
            idx_near = idx_in[(idx_in > i - nbuffer) & (idx_in < i + nbuffer)]   # get all indices within buffer bounds 
            if len(idx_near) < nbuffer:                                          # if there are lss in prximity than buffer length. 
                idx_near = np.sort(idx_near)
            else:                                                                # if it turns out that the number of samples in proximity is lower than nbuffer
                idx_near = np.sort(idx_near[0:nbuffer])                          # this is ofcourse complete bull

            # setup adaptive buffer and validate:
            idx_near = idx_in[(idx_in >= i - 2*nbuffer) & (idx_in <= i + 2*nbuffer)]   
            idx_near_rel = np.argsort(np.abs(idx_near - i))                          
            idx_near_rel = idx_near_rel[0:nbuffer]
            idx_near = np.sort(idx_near[idx_near_rel])

            ## use deterministic indexing - find closest entry and build buffer around that point. 
            #N = len(idx_in)
            #_nb = np.int(np.ceil(nbuffer/2))
            #pos_idx = np.argsort(np.abs(idx_in-i))[0]
            #if i > _nb and i < N-_nb:
            #    idx_near = idx_in[pos_idx-_nb:pos_idx+_nb]
            #elif i <= _nb:
            #    idx_near = idx_in[0:pos_idx+_nb]    
            #elif i >= N-_nb:
            #    idx_near = idx_in[pos_idx-_nb:]

            # validate buffer
            idx_near = np.asarray([val_me for val_me in idx_near if arr_1[val_me] < th])

            # reconstruction methods: this can be done nicer.
            if method == 'polyfit':
                t = time[idx_near]
                x = arr_1[idx_near]
                tr = np.asarray(time[i])
                pfit = np.polyfit(t, x, **settings)
                pred = np.polyval(pfit, tr)
                recon_dd[i] = np.float(pred)

            # replacement by mean, median or mode:
            if method == 'mean':
                recon_dd[i] = np.mean(arr_1[idx_near])
            if method == 'median':
                recon_dd[i] = np.median(arr_1[idx_near])
            if method == 'random':
                recon_dd[i] = np.random.choice(arr_1[idx_near], size=1)

        # setup clusters and labels
        clusters = [idx_in, idx]
        labels = np.zeros(np.max(arr_1.shape),)
        labels[idx_in] = 0
        labels[idx] = -1
        labels = labels

        return np.cumsum(recon_dd), labels, clusters

    def recon_th(self, arr=None, time=None, th=None, method='median', nbuffer=25, settings=None, ):
        """ Method do perform a reconstruction on signals contained in a numpy.ndarray of shape (n_signals, n_samples).      
        uses 'self._recon_th' to perform a reconstruction on the individual arrays/signals.
        
        :param arr: (Optional) noisy input array containing peak splitting / RTN artefacts. Defaults to 'self.arr'.
        :type arr: npumpy.ndarra, with shape (1, n_smaples)
        :param time: (Optional) time array or pseudo time array containgin sample indices
        :type time: numpy.ndarray
        :param th: (Optional) threshold level for outlier detection. Defaults to thresholds derived with 'self.estimate_th'
        :type th: float
        :param nbuffer: Number of samples in data buffer which is used to create outlier replacements
        :type nbuffer: int
        :param method: string to select outlier replacement method 'polyfit', 'mean', 'median', 'random'
        :type method: str
        :param settings: settings dictionary for np.polyfit command.
        :type settings: dict
        :return: array holding the reconstructed signal, a list containing the lists of labels, a list containing lists of clusters
        :rtype: numpy.ndarray
        """
        
        if arr is None:
            arr = self.arr
   
        if time is None: 
            time = np.arange(0, np.max(arr.shape))
        
        # th from estimate_th, or from int/float
        if th is None or len(th) != len(arr):
            th = self.estimate_th(arr, th=500, bin_density=10)
        if isinstance(th, (np.int, np.float)):
            th = list(th * np.ones(np.min(arr.shape), ))

        re_list = list()
        cl_list = list()
        la_list = list()
        for _arr, th_val in zip(arr, th):
            recon, labels, clusters = self._recon_th(arr=_arr, time=time, method=method, settings=settings, th=th_val, nbuffer=nbuffer)
            re_list.append(recon)
            cl_list.append(clusters)
            la_list.append(labels)
       
        rec = np.asarray(re_list)
        return np.atleast_2d(rec), cl_list, la_list
        

    # for consistency with other reconstruction methods:
    def _clusters(self):
        """Construct list of samples corresponding to all inliers and outliers respectively.

        :return: list holding lists of indiced corresponding to inlier and outliers respectively.
        :rtype: list
        """
        clusters = list()
        for labels in self.labels:
            clusters.append(
                [[k for k, li in enumerate(labels.astype(list)) if li == lu] for lu in list(set(labels))])
        return clusters

    def _counts(self):
        """Count inliers and outliers

        :return: list of lists holding the counts of inliers and outliers per signal.
        :rtype: list
        """
        counts = [[len(cl) for cl in clusters] for clusters in self.clusters]
        return counts

    def _scores(self):
        """Compute ratio between outliers and the total amount of samples.

        :return: list containging score for each signal
        :rtype: list
        """
        scores = [counts[1] / (counts[0] + counts[1]) if len(counts) > 1 else -1 for counts in self.counts]
        return scores
