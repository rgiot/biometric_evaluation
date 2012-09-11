#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from scipy import stats
from joblib import Parallel, delayed
import numpy as N
import biometricinformation as bi
from utils import get_confidence_limits

"""
This module manage the False Non Match Rate computation.

@todo Fusionner le code avec FTA

AUTHOR Romain Giot <giot.romain@gmail.com>
"""

GALLERY, PROBE, PRESENTATION, ERROR = (0, 1, 2, 3)

def bootstrap_helper(fnmr, estimated_error):
    """Function allowing to call bootstrap procedure in a parallel way.
    @param fnmr: FNMR on which we call the bootstrap.
    """
    bootstrap = fnmr.bootstrap()
    estimated_error_bootstrap = bootstrap.get_error_estimation()

    return estimated_error_bootstrap - estimated_error


class FNMR(bi.Score):
    """Get information on False Non Match Rate.
    """
    def __init__(self, data):
        """Build the FNMR computation code

        @param data: The data to use
        """

        super(FNMR, self).__init__(data)


    def get_error_estimation(self):
        """Returns the estimation of the Failure To Acquire."""
        return N.mean(self._data[:,ERROR])

    def bootstrap(self):
        """Return a boostrapped version of the FNMR.
        To get the boostrap, we sample n_A individuals with replacement from
        the n_A individuals from which there are acquisition attempts.
        """

        #Get the list of ids
        ids = self.get_raw_gallery_userid()
        #Sample (with replacement) indicies
        indicies = N.random.random_integers(0, len(ids)-1, len(ids))
        #Get the real ids
        selected = [ids[indice] for indice in indicies ]

        #Build the new list of users and samples
        new = None
        for new_id, old_id in enumerate(selected):
            #loop all other the selected users
            samples = N.copy(self.get_presentations_of_user(old_id))
            #Change id
            samples[:,GALLERY] = new_id

            #Append to list
            if new is None:
                new = samples
            else:
                new = N.vstack( (new, samples))

        return FNMR(new)

    def get_confidence_interval_bootstrap(self, alpha, repet = 4000):
        """Return the confidence interval computed in a bootsrap way.

        @param alpha: Confidence on the interval
        @param repet: Number of repetition

        @todo save the diff errors ? I norder to display otherwise
        """

        diff_error = []

        #Result from the complete set
        estimated_error = self.get_error_estimation()

        #Launch boostrap procedure
        #diff_error = [compute_boostrap_error() for i in range(repet)]
        diff_error = Parallel(n_jobs=-1)(delayed(bootstrap_helper)(self,
            estimated_error) \
                for i in range(repet))


        #Get the percentile value of the difference
        e_lower, e_upper = get_confidence_limits(diff_error, alpha)

        return estimated_error, max(0, estimated_error-e_upper), min(1, estimated_error-e_lower)

    def get_correlation_estimation(self):
        """We estimate the correlation (\psi) as Following.

        Corr(A_{ik}, A_{i\prime{}k\prime}) = 

          1       if i = i\prime, k = k\prime
          \psi    if i = i\prime, k != k\prime
          0 otherwise

        @todo verify if result is correct
        """

        estimated_error = self.get_error_estimation()
        numerator, denominator = 0.0, 0.0

        #loop on all users id
        for i in self.get_raw_gallery_userid(unique=True):
            M_i = self.get_presentations_number_of_user(i)
            denominator = denominator + M_i*(M_i-1)

            presentations = self.get_presentations_of_user(i)
            #loop on their presentation
            for j, pres1 in enumerate(presentations):
                #second loop on their presentation
                for jprime, pres2 in enumerate(presentations):
                    if j == jprime: continue

                    Diij = pres1[ERROR]
                    Diijprime = pres2[ERROR]
                    numerator = numerator + \
                            (Diij - estimated_error) * \
                            (Diijprime - estimated_error)

        return numerator/denominator

    def get_effective_sample_size(self):
        """Returns the effective sample size.
        @todo test if it works
        """

        N_users = self._data.shape[0]
        correlation = self.get_correlation_estimation()

        tmp = N.array([self.get_presentations_number_of_user(user) \
                        for user in self.get_raw_gallery_userid(unique=True)])
        M0 = N.sum(tmp**2) / N.sum(tmp)

        return N_users / (1 + correlation*(M0-1))


    def get_confidence_interval_largesample(self, alpha):
        """Compute the confidence interval in a large sample way.
        The method does not verify if the large sample method can be used,
        it's up to the user to do this verification.

        @param alpha: Confidence on the interval
        """

        sample_size = self.get_effective_sample_size()
        estimated_error = self.get_error_estimation()
        correlation = self.get_correlation_estimation()

        tmp = N.array([self.get_presentations_number_of_user(user) \
                        for user in self.get_raw_gallery_userid(unique=True)])
        M0 = N.sum(tmp**2) / N.sum(tmp)

        Mmean = self.get_mean_number_of_presentations()

        # Compute parts of the equation
        A = estimated_error * (1 - estimated_error )
        B = 1 + correlation*(M0 - 1)
        C = self.get_number_of_users() * Mmean

        percentile = (1.0-alpha/2.0)*100.0
        D = stats.scoreatpercentile( \
                stats.norm.pdf(N.linspace(-4,0.0001,4)),
                alpha)

        width = D*math.sqrt(A*B/C)

        return max(0, estimated_error - width), min(1,estimated_error + width)

