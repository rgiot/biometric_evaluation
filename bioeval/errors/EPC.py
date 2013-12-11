#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AUTHOR Romain Giot <romain.giot@ensicaen.fr>
LICENCE GPL

The Expected Performance Curve
"""

from traits.api import HasStrictTraits, Instance, \
        Float, Array
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt


from joblib import Parallel, delayed

#Force to use latex backend
params = {'text.usetex': True}
plt.rcParams.update(params)


import roc
from utils import get_confidence_limits

# TODO Write the area under the curve code
class EPC(HasStrictTraits):
    """Expected Performance Curve.
    Build the expected performance curve, based on two sets of scores.
    """

    # First system
    devel = Instance(roc.ROCScores)

    # Second system
    test = Instance(roc.ROCScores)

    # Comuted curve
    curve = Array(None)


    def compute_epc(self, alpha_min=0.00001, alpha_max=1, alpha_step=1000,
            nb_thresholds=1000):
        """Do all the computing stuff for the epc.

        @param alpa_min: minimum value to variate alpha
        @param alpha_max: maximum value (not reach) to variate alpha
        @param alpha_step: Number of alphas
        """

        res = []

        # Get the list of thresholds
        develroc = self.devel.get_roc(nb_threshold=nb_thresholds)
        thresholds = develroc.get_raw_thresholds()
        # loop on several alphas
        for alpha in np.linspace(alpha_min, alpha_max, alpha_step, endpoint=False):
            # Get errors on all thresholds
            weighted_errors = develroc.get_weighted_errors(alpha)
            # Get minimum value
            idx = np.argmin(weighted_errors)
            # Get best theta for this alpha
            theta_star = thresholds[idx]

            # Compute error on test set (HTER)
            testerror = self.test.get_weighted_error_at_threshold(theta_star, 0.5)
        
            res.append( (alpha, testerror) )


        self.curve = np.array(res)

    def display_epc(self, display_infos=True):
        """Display the curve.EPC
        """

        plt.plot(self.curve[:,0], self.curve[:,1], linewidth=2)

        plt.xlabel('$\\alpha$')
        plt.ylabel('HTER')



def bootstrap_helper(devel, test,
        alpha_min, alpha_max, alpha_step, nb_thresholds):
    """Compute the epc on a boostraped version of the scores."""
    # Bootstrap curves
    bdevel = devel.bootstrap()
    btest = test.bootstrap()

    # Get epc
    bepc = EPC(devel=bdevel,test= btest)
    bepc.compute_epc(alpha_min=alpha_min,
                     alpha_max=alpha_max,
                     alpha_step=alpha_step,
                     nb_thresholds=nb_thresholds)

    #assert np.all(alphas == bepc.curve[:,0])
    #bcurves.append(bepc.curve[:,1])
    return bepc.curve[:,1]

class BootstrapedEPC(HasStrictTraits):
    """Compute a bootstraped version of the epc.
    Use the same bootstraping technic than in the Schuckers book.
    """


    # First system
    devel = Instance(roc.ROCScores)

    # Second system
    test = Instance(roc.ROCScores)


    def compute_epc(self, nb_bootstrap=1000, confidence=0.05, alpha_min=0.00001, alpha_max=1, alpha_step=1000,
            nb_thresholds=1000):


        # Get initial result
        base_epc = EPC(devel=self.devel, test=self.test)
        base_epc.compute_epc(alpha_min, alpha_max, alpha_step, nb_thresholds)
        base_curve = base_epc.curve[:,1]
        alphas = base_epc.curve[:,0]
        bcurves = []
        plt.plot(alphas, base_curve, linewidth=2, color='blue')


        bcurves = Parallel(n_jobs=-1, verbose=1)\
                (delayed(bootstrap_helper)(self.devel, self.test,
                                           alpha_min, alpha_max, alpha_step, nb_thresholds) \
                    for i in range(nb_bootstrap))
#        for b in range(nb_bootstrap):
#            bcurves.append(bootstrap_helper(self.devel, self.test))
        bcurves = np.array(bcurves)

        for bcurve in bcurves:
            plt.plot(alphas, bcurve, linestyle='steps:', color='gray')


        diff = bcurves - base_curve
        # TODO get real s
        s = np.std(diff, axis=0)
        e = diff/s

        #Compute maximum absolute standard residual
        min_e = np.amin(e, axis=1)
        max_e = np.amax(e, axis=1)
        min_e.shape = (-1,1)
        max_e.shape = (-1,1)

        indicies1 = np.where( np.abs(min_e) < np.abs(max_e) )
        indicies2 = np.where( np.abs(min_e) >= np.abs(max_e) )

        w = np.empty( (e.shape[0],1) )
        w[indicies1] = max_e[indicies1]
        w[indicies2] = min_e[indicies2]

        #Get ROC confidence
        delta_L, delta_U = get_confidence_limits(w, confidence)
        rU = base_curve - (delta_L * s)
        rL = base_curve - (delta_U * s)

        plt.plot(alphas, rU, linewidth=2, color='red')
        plt.plot(alphas, rL, linewidth=2, color='red')

        plt.xlabel('$\\alpha$')
        plt.ylabel('HTER')



if __name__ == "__main__":
    import sys
    sys.path.append('./../')
    import data.banca as banca
    import roc


    source = banca.Banca()
    #db1 = source.get_data('g1', 'IDIAP_voice_gmm_auto_scale_25_10_pca')
    #db2 = source.get_data('g1', 'IDIAP_voice_gmm_auto_scale_25_200_pca')
    db1 = source.get_data('g1', 'SURREY_face_nc_man_scale_100')
    db2 = source.get_data('g2', 'SURREY_face_nc_man_scale_100')

    rocscores1 = roc.ROCScores(db1)
    rocscores2 = roc.ROCScores(db2)

    epc = EPC(devel=rocscores1, test=rocscores2)
    epc.compute_epc()

    plt.figure()
    epc.display_epc()



    db1 = source.get_data('g1', 'SURREY_face_svm_man_scale_2.00')
    db2 = source.get_data('g2', 'SURREY_face_svm_man_scale_2.00')

    rocscores1 = roc.ROCScores(db1)
    rocscores2 = roc.ROCScores(db2)

    epc = EPC(devel=rocscores1, test=rocscores2)
    epc.compute_epc()

    epc.display_epc()

    plt.legend(['SURREY\_FACE\_NC\_100', 'SURREY\_FACE\_SVM\_2.0'])


    plt.figure()
    epc = BootstrapedEPC(devel=rocscores1, test=rocscores2)
    epc.compute_epc(1000)

    plt.show()
