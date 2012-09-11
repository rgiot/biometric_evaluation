#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as N
from scipy import stats
import math

from joblib import Memory
from joblib import Parallel, delayed

import biometricinformation as bi
from utils import get_confidence_limits

"""
The aim of this module is to analyse all that
relates of Failure To Acquire.

AUTHOR Romain Giot <giot.romain@gmail.com>

"""

#Set the order of the columns
USER, PRESENTATION, ERROR = (0, 1, 2)

def bootstrap_helper(fta, estimated_error):
    """Function allowing to call bootstrap procedure in a parallel way.
    @param fta: Fta on which we call the bootstrap.
    @todo opitmize by leavung fta out
    """
    bootstrap = fta.bootstrap()
    estimated_error_bootstrap = bootstrap.get_error_estimation()

    return estimated_error_bootstrap - estimated_error


class FTA(bi.AbstractBiometricData):
    """A FTA emebds the list of Failure To Acquire from
    a defined biometric system.

    The FTA are stored in an array of the following shape:
    user   presentation #   error (or not)
    1      1                0
    1      2                1
    1      3                0
    2      1                0
    2      2                0
    ...
    """

    def __init__(self, data):
        """Build an FTA from a filename or numpy array.

        @param data: if it is a string, this is the filename
                     of the file to load. Otherwise it is the numpy
                     array.
        """

	super(FTA, self).__init__(data)


        #Cache
        mem = Memory(cachedir='/tmp/', mmap_mode='r')
      #  self.get_value = mem.cache(self.get_value)

    def __str__(self):
        """Gives a string representation of the FTA"""
        res =  "Failure to Acquire\n"
        res += "====================\n\n"

        res += "\tTotal number of presentations: %d\n" % self._data.shape[0]
        nb_users = self.get_number_of_users()
        res += "\tNumber of users: %d\n" % nb_users

        nb_presentations = \
                [self.get_presentations_number_of_user(user) \
                    for user in self.get_users_id() ]

        if len(N.unique(nb_presentations)) == 1:
            res += "\tNumber of presentation per users: %d\n" % \
            nb_presentations[0]
        else:
            res += "\n\tPresentations per users:\n"
            for index, userid in enumerate(self.get_users_id()):
                res += "\t\tUser %d => %d\n" % (userid, nb_presentations[index])

            res += "\nMean number of presentation: %d" % self.get_mean_number_of_presentations()

        return res

    def get_presentations_of_user(self, userid):
        """Returns the presentations of the user.
        The result is an array like that:
        user presentation #    error

        @param userid: Id of the required user
        """
        indicies = N.where(self._data[:,USER] == userid)
        return self._data[indicies]

    def get_users_id(self):
        """Returns the id of the users. Usefull when some users
        are deleted from the dataset.
        """
        return N.unique(self._data[:,USER])
#Specific
    def get_error_estimation_slow(self):
        """Returns the estimation of the Failure To Acquire."""

        sum = 0.0
        nb = 0
        for userid in self.get_users_id():
            for presentation in \
                self.get_presentations_of_user(userid):
                    error  = presentation[ERROR]
                    sum = sum + error
                    nb = nb + 1

        return sum/nb

    def get_error_estimation(self):
        """Returns the estimation of the Failure To Acquire."""
        return N.mean(self._data[:,ERROR])

    def get_confidence_interval(self, alpha = 0.1, method=None):
        """Returns the confidence interval of the FTAR.

        @param alpha: Confidence on the interval
        @param method: Type of method too use ('largesample' or 'boostrap')
        @return : Confidence interval
        """

        #Choose the appropriate method if not set
        if method == None:
            if self.can_use_large_sample_approach():
                method = 'largesample'
            else:
                method = 'boostrap'

        #Call the right method
        if method == 'largesample':
            return self.get_confidence_interval_largesample(alpha)
        else:
            return self.get_confidence_interval_bootstrap(alpha)

    def get_confidence_interval_bootstrap(self, alpha, repet = 4000):
        """Return the confidence interval computed in a bootsrap way.

        @param alpha: Confidence on the interval
        @param repet: Number of repetition

        @todo save the diff errors ? I norder to display otherwise
        """
        def compute_boostrap_error():
            """Launch one boostrap computing.
            Function created only for multiprocessing."""
            bootstrap = self.bootstrap()
            estimated_error_bootstrap = bootstrap.get_error_estimation()

            return estimated_error_bootstrap - estimated_error

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

        return max(0, estimated_error-e_upper), min(1, estimated_error-e_lower)

    def bootstrap(self):
        """Return a boostrapped version of the FTA.
        To get the boostrap, we sample n_A individuals with replacement from
        the n_A individuals from which there are acquisition attempts.
        """

        #Get the list of ids
        ids = self.get_users_id()
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
            samples[:,USER] = new_id

            #Append to list
            if new is None:
                new = samples
            else:
                new = N.vstack( (new, samples))

        return FTA(new)

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
                        for user in self.get_users_id()])
        K0 = N.sum(tmp**2) / N.sum(tmp)

        Kmean = self.get_mean_number_of_presentations()

        # Compute parts of the equation
        A = estimated_error * (1 - estimated_error )
        B = 1 + correlation*(K0 - 1)
        C = self.get_number_of_users() * Kmean

        percentile = (1-alpha/2)*100
        D = stats.scoreatpercentile( \
                stats.norm.pdf(N.linspace(-4,0.0001,4)),
                alpha)

        width = D*math.sqrt(A*B/C)

        return max(0, estimated_error - width), min(1,estimated_error + width)

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
        for i in self.get_users_id():
            K_i = self.get_presentations_number_of_user(i)
            denominator = denominator + K_i*(K_i-1)

            presentations = self.get_presentations_of_user(i)
            #loop on their presentation
            for k, pres1 in enumerate(presentations):
                #second loop on their presentation
                for kprime, pres2 in enumerate(presentations):
                    if k == kprime: continue

                    Aik = pres1[ERROR]
                    Aikprime = pres2[ERROR]
                    numerator = numerator + \
                            (Aik - estimated_error) * \
                            (Aikprime - estimated_error)

        return numerator/denominator


    def get_effective_sample_size(self):
        """Returns the effective sample size.
        @todo test if it works
        """

        N_gamma = self._data.shape[0]
        correlation = self.get_correlation_estimation()

        tmp = N.array([self.get_presentations_number_of_user(user) \
                        for user in self.get_users_id()])
        K0 = N.sum(tmp**2) / N.sum(tmp)

        return N_gamma / (1 + correlation*(K0-1))

    def can_use_large_sample_approach(self):
        """Test if we can use the large sample approach."""
        eff_size = self.get_effective_sample_size()
        est_err = self.get_error_estimation()

        A = eff_size * est_err
        B = eff_size * (1-est_err)

        print 'A=%f\nB=%f\n' % (A, B)
        return A>=10 and B>=10


