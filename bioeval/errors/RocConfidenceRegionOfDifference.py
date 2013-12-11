#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AUTHOR Romain Giot <giot.romain@gmail.com>
LICENCE GPL
"""

from joblib import Parallel, delayed
import numpy as N
import matplotlib.pyplot as plt
from roc import PolarRocCurve, ROCScores
from utils import get_confidence_limits

#Force to use latex backend
params = {'text.usetex': True}
plt.rcParams.update(params)


def bootstrap_helper(original1, original2, original1polar, original2polar, angles):
    """Wrapper to be able to use multiprocessing.
    @param original1 = original ROC scores of system 1
    @param original2 = original ROC scores of system 2
    @param estimated_roc1 = estimation of the roc curve 1
    @param estimated_roc2 = estimation of the roc curve 2
    """

    # Boostrap the two curves
    bootstraped_rocscores1 = \
        original1.bootstrap().get_roc().get_polar_representation()
    bootstraped_rocscores2 = \
        original2.bootstrap().get_roc().get_polar_representation()
    bootstraped_rocscores1.shrink_angles(angles)
    bootstraped_rocscores2.shrink_angles(angles)


    # Difference for ROC1
    diff1b1 = PolarRocCurve.substract_curves(\
        bootstraped_rocscores1,
        original1polar)

    # Difference for ROC2
    diff2b2 = PolarRocCurve.substract_curves(\
        bootstraped_rocscores2,
        original2polar)

    # Difference between ROC1 and ROC2
    diff1b2b = PolarRocCurve.substract_curves(\
        bootstraped_rocscores1,
        bootstraped_rocscores2)


    return diff1b1, diff2b2, diff1b2b



class ROCConfidenceRegionOfDifference(object):
    """Compute the confidence region of difference of two independant ROC
    curves.
    """

    def __init__(self, roc1, roc2):
        """Initialize the procedure.

        @param roc1: first roc scores
        @param roc2: second roc scores
        """
        self.roc1 = roc1
        self.roc2 = roc2

    def get_confidence_region(self, nb_iter=1000, alpha=0.05, angles=None):
        """Launch the computation of the confidence region of the two ROC
        curves using a parametric way.

        @param nb_iter : Number of bootstrap to do
        @param alpha: confidence
        @param angles: angles to use
        """

      
        ##############
        # First step #
        ##############

        # Get estimated ROC curves for the two ROCs
        estimated_roc1 = self.roc1.get_roc().get_polar_representation()
        estimated_roc1.shrink_angles(angles)
        estimated_roc2 = self.roc2.get_roc().get_polar_representation()
        estimated_roc2.shrink_angles(angles)

        if angles is None:
            angles = estimated_roc1.get_raw_theta()

        # Compute difference for each angle
        original_diff = PolarRocCurve.substract_curves(\
            estimated_roc1,
            estimated_roc2)

        #############################
        # Repeat for each bootstrap #
        #############################
        diffs = Parallel(n_jobs=-1, verbose=True)(
                    delayed(bootstrap_helper)(self.roc1, self.roc2,
                        estimated_roc1, estimated_roc2, angles)
                            for iteration in range(nb_iter))


        # Get the differences
        diffs = N.asarray(diffs)
        diffs1b1 = diffs[:,0]
        diffs2b2 = diffs[:,1]
        diffs1b2b = diffs[:,2]
        del diffs

        ###################
        # Compute results #
        ###################
        #Get t
        min_angle = N.min(angles)
        max_angle = N.max(angles)
        N1 = self.roc1.get_number_of_presentations()
        N2 = self.roc2.get_number_of_presentations()
        t = N.abs((angles - min_angle)*(max_angle - angles))/ \
                ((max_angle - min_angle)*(max_angle - min_angle) * 100 * (N1 + N2))

        #Get s
        tmp = N.mean(diffs1b2b, axis=0)
        tmp = N.square(diffs1b2b - tmp)
        tmp = N.sum(tmp, axis=0)
        tmp = tmp/float(nb_iter-1)
        tmp = tmp + t
        s = N.sqrt(tmp)
        del tmp

        #get replicated normalized difference
        e = (diffs1b1 - diffs2b2)/s

        #Compute maximum absolute standard residual
        #Pass extrem values (problem with them ...)
        min_e = N.amin(e[:,1:-1], axis=1)
        max_e = N.amax(e[:,1:-1], axis=1)
        min_e.shape = (-1,1)
        max_e.shape = (-1,1)

        indicies1 = N.where( N.abs(min_e) < N.abs(max_e) )
        indicies2 = N.where( N.abs(min_e) >= N.abs(max_e) )

        w = N.empty( (e.shape[0],1) )
        w[indicies1] = max_e[indicies1]
        w[indicies2] = min_e[indicies2]

        #Get percentiles
        delta_L, delta_U = get_confidence_limits(w, alpha)

        #Get confidence interval
        R = original_diff
        rU = R - (delta_L * s)
        rL = R - (delta_U * s)


        ##################
        # Display result #
        ##################
        plt.hlines([0], N.pi/2, N.pi)
        for diff in diffs1b2b:
            plt.plot(angles, diff, linestyle="steps:", color='gray')
        plt.plot(angles, rU, linewidth=3, color='red')
        plt.plot(angles, rL, linewidth=3, color='red')
        plt.plot(angles, original_diff, linewidth=2, color='blue')
        plt.ylim((-0.45,0.45))
        plt.yticks(N.linspace(-0.4,0.4,5))
        plt.xlim((N.pi/2,N.pi))
        plt.xticks(N.linspace(N.pi/2, N.pi, 5),
                [r'$\pi/2$', r'', r'$3\pi/4$', '', r'$\pi$'])
        plt.xlabel(r'$\theta$')#todo set latex notation
        plt.ylabel(r'$\hat{r}_{\theta}^{(1)}-\hat{r}_{\theta}^{(2)}$')
        plt.gca().invert_xaxis()



if __name__ == "__main__":
    """Test"""
    import sys
    sys.path.append('./../')
    import data.banca as banca

    print 'Load sources'
    source = banca.Banca()
    #db1 = source.get_data('g1', 'IDIAP_voice_gmm_auto_scale_33_300_pca')
    #db2 = source.get_data('g2', 'IDIAP_voice_gmm_auto_scale_33_300_pca')
    db1 = source.get_data('g1', 'IDIAP_voice_gmm_auto_scale_25_10_pca')
    db2 = source.get_data('g2', 'IDIAP_voice_gmm_auto_scale_25_200_pca')


    print 'Build score roc'
    rocscores1 = ROCScores(db1)
    rocscores2 = ROCScores(db2)


    plt.figure()

    print 'Compute the difference.'
    conf = ROCConfidenceRegionOfDifference(rocscores1, rocscores2)
    conf.get_confidence_region(nb_iter=1000, alpha=0.05, angles=
            N.linspace(N.pi/2,N.pi, 100))
    plt.show()
