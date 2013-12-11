# -*- coding: utf-8 -*-

"""
This module embeds all the things concerning ROC scores and ROC curves.

AUTHOR Romain Giot <giot.romain@gmail.com>
LICENCE GPL

TODO move all the plotting things elsewhere
"""
from traits.api import HasTraits, String, Enum

import numpy as N
from scipy.integrate import trapz
from scipy.interpolate import interp1d
import scipy.interpolate
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import biometricinformation as bi
from utils import get_confidence_limits


#Force to use latex backend
params = {'text.usetex': True}
plt.rcParams.update(params)

def bootstrap_helper(original, estimated_roc):
    """Wrapper to be able to use multiprocessing.
    @param original = original ROC scores
    @param estimated_roc = estimation of the roc curve
    """
    #Bootstrap distribution
    bootstraped_rocscores = original.bootstrap()

    #Get roc curve
    bootstraped_roc = \
        bootstraped_rocscores.get_roc().get_polar_representation()
    bootstraped_roc.shrink_angles()

    #Get difference between curves
    diff = PolarRocCurve.substract_curves(\
            bootstraped_roc,
            estimated_roc)

    return bootstraped_roc, diff

def progress_range(a, b=None):
    """Act as the range method, but display progression."""

    if b is None:
        numbers = range(a)
        maxnb = a
    else:
        numbers = range(a, b)
        maxnb = b-a

    total = 0
    last_percent_displayed = 0
    for nb in numbers:
        total = total + 1
        actual_percent = total/float(maxnb)

        if actual_percent - last_percent_displayed >= 0.05:
            last_percent_displayed = actual_percent
            print '%f/100 jobs launched' % (100*last_percent_displayed)

        yield nb

class ROCScores(bi.Score):
    """ROC scores. This class embeds the information of to compute the ROC
    curve.
      * gallery_user
      * prob_user
      * presentation_number
      * score
    @todo: Rename this class and move it ?
    """

    _type = Enum('distance','score')

    def __init__(self, data, scoretype='score'):
        """Build a ROC.

        @param data: The data to use
        @param type: to type of score (distance or score)
        """

        super(ROCScores, self).__init__(data)
        self._type = scoretype

    def get_unique_scores_list(self):
        """Returns the list of scores (only one instance per score)"""
        return N.unique(self.get_raw_scores())





    def bootstrap(self):
        """Returns a boostrap version of the ROC matche score data.
        The methodology is the following:
            * sample with replacement individuals from the probe
            * sample with replacement the gallery
        """

        # sample with replacement from individual in the probe
        probe_id = self.get_raw_prob_userid(unique=True)
        indicies = N.random.random_integers(0, len(probe_id)-1, len(probe_id))
        b = probe_id[indicies]

        gallery_id = self.get_raw_gallery_userid(unique=True)

        new_data = None
        for b_i in b:
            # resample with replacement individuals in the prob
            indicies = N.random.random_integers(0, len(gallery_id)-1, len(gallery_id))
            h_i = gallery_id[indicies]

            # and take the scores for each pairs of individuals
            for h_ik in h_i:
                indicies = N.where(N.logical_and(\
                        self._data[:,self.GALLERY] == h_ik,
                        self._data[:,self.PROBE] == b_i))[0]
                samples = self._data[indicies]

                if samples.shape[0] > 0:
                    if new_data is None:
                        new_data = samples
                    else:
                        new_data = N.vstack( (new_data, samples) )

        return ROCScores(new_data, self._type)


    def get_weighted_error_at_threshold(self, threshold, alpha=0.5):
        """Return the performance of the system at the given threshold.
        when alpha=0.5, returns the HTER
        Performance is computed like that:
        $\alpha*FAR(threshold) + (1-\alpha)*FRR(threshold)
        """
        assert alpha > 0 and alpha < 1

        #Split scores
        intra = self.get_genuine_presentations()
        inter = self.get_impostor_presentations()

        # Get decision errors
        if self._type == 'distance':
            intra_reject = intra.superior_to_threhold(threshold)
            inter_accept = inter.inferior_to_threhold(threshold)

        else:
            intra_reject = intra.inferior_to_threhold(threshold)
            inter_accept = inter.superior_to_threhold(threshold)

        # Get errors
        FNMR = intra_reject.get_estimated_error()
        FMR = inter_accept.get_estimated_error()


        return alpha*FMR + (1-alpha)*FNMR

    def get_roc(self, nb_threshold=1000):
        """Compute the ROC curve."""

        #Split scores
        intra = self.get_genuine_presentations()
        inter = self.get_impostor_presentations()

        #Get thresholds to test
        thresholds = self.get_unique_scores_list()
        if thresholds.shape[0] > nb_threshold:
            start = min(intra.min(), inter.min())
            end = max(intra.max(), inter.max())
            thresholds = N.linspace(start, end, nb_threshold)
            print 'ROC creation/ Reduce threshold list'
        else:
            print 'ROC creation/ Use all thresholds (not enough present ...)'


        #res
        res = None

        if self._type == 'distance':
            #Whole loop (unusefull test done only one time)
            for threshold in thresholds:
                intra_reject = intra.superior_to_threhold(threshold)
                inter_accept = inter.inferior_to_threhold(threshold)

                FNMR = intra_reject.get_estimated_error()
                FMR = inter_accept.get_estimated_error()

                if res == None:
                    res = (threshold, FNMR, FMR)
                else:
                    res = N.vstack( (res, (threshold, FNMR, FMR)) )
        else:
            for threshold in thresholds:
                intra_reject = intra.inferior_to_threhold(threshold)
                inter_accept = inter.superior_to_threhold(threshold)

                FNMR = intra_reject.get_estimated_error()
                FMR = inter_accept.get_estimated_error()

                if res == None:
                    res = (threshold, FNMR, FMR)
                else:
                    res = N.vstack( (res, (threshold, FNMR, FMR)) )


        return CartesianROCCurve(res, self._type)

    def display_bootstraped_difference(self):
        """Display in the actual figure, the histogram of the 
        eer difference between the orignal curve and the bootstrapes ones.
        """

        plt.hist(self._diff_eers, 50, fill=False)
        plt.xlabel('$e_\chi$')
        plt.ylabel('Frequency')



    def get_confidence_interval(self, alpha=0.05, repet=1000, angles=None):
            """Return the confidence interval of the ROC curve.
            The curves are displayed in the actual figure.
            The method also compute the EER interval.

            @todo: split the method in several parts

            @param repet: Number of repetition
            """
            #######################
            # Launch boostrapping #
            #######################

            #Calculate the estimated ROC and its EER
            estimated_roc = self.get_roc().get_polar_representation()
            estimated_eer = estimated_roc.get_eer()
            self.estimated_eer = estimated_eer
            estimated_roc.shrink_angles(angles)

            #Launch computation in parralel
            res = Parallel(n_jobs=-1, verbose=1)(delayed(bootstrap_helper)(self,
                                                                estimated_roc) \
                for i in range(repet))

            # Merge results
            tmp_bootstrap, tmp = zip(*res)
            del res #delete old results

            #Display the curves, compute the EER and free memory
            for bootstraped_roc in tmp_bootstrap:
                bootstraped_roc.tmpplot() # Display curve
            #Compute EER
            bootstraped_eers = [bootstraped_roc.get_eer() for bootstraped_roc in tmp_bootstrap]
            self.bootstraped_eers = bootstraped_eers
            del tmp_bootstrap # delete inutile roc curves


            ################################
            ##### Compute EER interval #####
            ################################
            diff_eers = N.asarray(bootstraped_eers) - estimated_eer
            del bootstraped_eers # free memory

            lower, upper = get_confidence_limits(diff_eers, alpha)
            lower_eer = estimated_eer - upper
            upper_eer = estimated_eer - lower
            #Store diff_eer for offline interpretation
            self._diff_eers = diff_eers

            ################################
            ##### Extract information ######
            ################################

            tmp = N.asarray(tmp)
            #Compute adjusted standard deviation
            theta = estimated_roc.get_raw_theta() # List of theta angles (the same for all)
            theta_min = N.min(theta)
            theta_max = N.max(theta)

            Nb = self.get_number_of_presentations()
            numerator = N.abs( (theta - theta_min) * (theta_max - theta))
            denominator =  ((theta_max - theta_min)**2)*(10**2)*Nb
            t = numerator/float(denominator + N.finfo(N.float).eps)

            s = N.sqrt(1/float(len(tmp)-1)*N.sum(tmp**2, axis=0) + t)
            assert s.shape[0] == len(theta)

            #Compute residual error
            e = tmp / s

            assert e.shape[0] == len(tmp) and e.shape[1] == len(theta)
            # Remove boundaries
            e = e[:, 1:-1]

            #Compute maximum absolute standard residual
            min_e = N.amin(e, axis=1)
            max_e = N.amax(e, axis=1)

            assert len(min_e) == len(max_e) == len(tmp)
            min_e.shape = (-1,1)
            max_e.shape = (-1,1)

            indicies1 = N.where( N.abs(min_e) < N.abs(max_e) )
            indicies2 = N.where( N.abs(min_e) >= N.abs(max_e) )

            w = N.empty( (len(tmp),1) )
            w[indicies1] = max_e[indicies1]
            w[indicies2] = min_e[indicies2]

            #Get ROC confidence
            delta_L, delta_U = get_confidence_limits(w, alpha)


            assert len(w) == len (tmp)
            R = estimated_roc.get_raw_r()
            rU = R - (delta_L * s)
            rL = R - (delta_U * s)

            #plot boundaries
            PolarRocCurve.plot_boundary(estimated_roc.get_raw_theta(), rL)
            PolarRocCurve.plot_boundary(estimated_roc.get_raw_theta(), rU)

            estimated_roc.plot() #displayed here to be in front


            print "EER information"
            print "==============="
            print 'Estimated EER: %0.6f'  % estimated_eer
            print 'Lower boundary: %0.6f' % lower_eer
            print 'Upper boundary: %0.6f' % upper_eer
            print
            return estimated_eer, (lower_eer, upper_eer)

class ROCCurve(bi.AbstractBiometricData):
    """Abstract ROC curve.
    Implementation are representation under polar and cartesian coordiantes
    """
    _type = Enum('score', 'distance')
    pass

class CartesianROCCurve(ROCCurve):
    """Store the results of a ROC curve.
    The data are composed of 3 fields (in this order):

      * decision threshold
      * False Non Match Rate
      * False Match Rate
    """


    def __init__(self, data, scoretype='score'):
        """Build a ROC curve.

        @param data: The data to use
        @param type: to type of score (distance or score)
        """

        super(ROCCurve, self).__init__(data)
        self._type = scoretype

    def get_threshold_for_FMR(self, fmr=0.01):
        """Compute the threshold gving the fmr.

        @param fmr: False Match Rate for which we want to obtain the threshold.
        """
        interT = interp1d(self.get_raw_FMR(), self.get_raw_thresholds(),
                bounds_error=True,)
        return interT(fmr)

    def get_raw_FNMR(self):
        """Returns False Non Match Rate"""
        return self._data[:,1]

    def get_raw_FMR(self):
        """Returns False Match Rate"""
        return self._data[:,2]

    def get_raw_TMR(self):
        """Returns True Match Rate"""
        return 1 - self.get_raw_FNMR()

    def get_raw_thresholds(self):
        return self._data[:,0]

    def get_auc(self):
        """Returns the Area Under the Curve."""
        x = self._data[:,2]
        y = 1 - self._data[:,1]

        #Sort if necessary
        indicies = N.argsort(x)
        x = x[indicies]
        y = y[indicies]

        #Compute integral
        return trapz(y,x)

    def get_weighted_errors(self, alpha):
        """Returns a weighted error for each threshold:
          $\alpha*FAR(threshold) + (1-\alpha)*FRR(threshold)

        Parameters
        ==========
         - alpha : float (0<alpha<1)
            Weight to compute the error
        """
        return alpha*self.get_raw_FMR() + (1-alpha)*self.get_raw_FNMR()

    def get_weighted_error_at_threshold(self, alpha, threshold):
        """Compute the error at the required threshold.

        Parameters
        ==========
         - alpha : float (0<alpha<1)
            Weight to compute the error
         - threshold: float
            Decision threshold at which we want the error
        """
        raise "Not implemetned"

    def get_eer(self):
        """Return the EER.
        @todo give a better value (cf. biosecure)
        """
        diff = self._data[:,1] - self._data[:,2]

        if diff[0] < 0:
            indicies = N.where(diff<0)[0]
            indice = len(indicies)
            return N.mean( self._data[indice, [1,2]] )
        else:
            raise Exception("To code")

    def plot_FMR_FNMR_versus_threshold(self):
        """Plot the False MAthc Rate and Fals Non MAtch Rate versus the
        threshold.
        """
        plt.xlabel('Threshold')
        plt.ylabel('Error Rate')

        plt.plot(self.get_raw_thresholds(),
                 self.get_raw_FNMR(),
                 color='blue',
                 linewidth=2,
                 label='FNMR')
        plt.plot(self.get_raw_thresholds(),
                 self.get_raw_FMR(),
                 color='red',
                 linewidth=2,
                 label='FMR')

        plt.legend()
        plt.title('FNMR and FMR vs Threshold')

    def plot_det(self):
        """Plot the DET curve"""

        def inverse(array):
            """Compute the inverse gaussian distribution"""
            mu = N.mean(array)
            nu = N.std(array)

            return N.sqrt(\
                    nu/(2*N.pi*(array**3) + N.finfo(N.float).eps)) * \
                N.exp(\
                ((-nu*((array-mu))**2))/ \
                (2.0*(mu**2)*array + N.finfo(N.float).eps))

        plt.xlabel('$\Phi^{-1}(\hat{FMR})$')
        plt.ylabel('$\Phi^{-1}(\hat{FNMR})$')




#        plt.plot(inverse(self.get_raw_FMR()),
#                 inverse(self.get_raw_FNMR()))
        plt.loglog(self.get_raw_FMR(), self.get_raw_FNMR(), linewidth=2,
        color='blue')

    def plot(self, linewidth=2, color='black'):
        """Plot the ROC curve on a graphics"""
        plt.xlabel('FMR')
        plt.ylabel('TMR')
        plt.plot(self._data[:,2], 1 - self._data[:,1], linewidth=linewidth, color=color)
        plt.ylim((-0.025,1.025))
        plt.xlim((-0.025,1.025))
        plt.xticks(N.arange(0,1.1,.2))
        plt.yticks(N.arange(0,1.1,.2))

    def tmpplot(self):
        """Plot a bootstraped version"""
        plt.plot(self._data[:,2], 1 - self._data[:,1], color='gray',
                linestyle='steps:')


    def get_polar_representation(self):
        """Returns a polar representation of the ROC curve."""

        # Get the coordinates
        X = self.get_raw_FMR()
        Y = self.get_raw_TMR()
        Thr = self.get_raw_thresholds()

        # Change of axis
        X = X - 1

        # Move to polar
        R = N.sqrt(X**2 + Y**2)
        phi = N.arccos(X/R)

        res = N.empty( (len(X),3))
        res[:,0] = Thr
        res[:,1] = phi
        res[:,2] = R

        return PolarRocCurve(res)

class PolarRocCurve(ROCCurve):
    """ROC curve in a polar coordinate.
    Information stored are:
     * threshold
     * theta
     * r
    """

    THRESHOLD = 0
    THETA = 1
    R = 2

    def get_raw_theta(self):
        """Returns the angles."""
        return self._data[:, self.THETA]

    def get_raw_thresholds(self):
        """Returns the thresholds."""
        return self._data[:, self.THRESHOLD]

    def get_raw_r(self):
        """Returns the radius."""
        return self._data[:, self.R]

    def shrink_angles(self, angles=None):
        """Limit the computed angles to the choosen set.
        Before this step the list of computed angles only depends
        on the point of the curve. They may be not in the list of attended
        angles. For these ones, we use a linear interpolation to get them.
        @param angles: Set of angles to use
        """

        if angles is None:
            #Choose the angle list
            low = N.pi/2
            high = N.pi
            T = 99
            angles = N.linspace(low, high, T)

        #Build interpolation  functions
        if True:
            interR = interp1d(self.get_raw_theta(), self.get_raw_r(),
                    bounds_error=False,
                    fill_value=1)
        else:
            interR = \
            interpolate.InterpolatedUnivariateSpline(self.get_raw_theta(), self.get_raw_r())
        interT = interp1d(self.get_raw_theta(), self.get_raw_thresholds(),
                bounds_error=False)

        #Interpolate
        new_data = N.empty( (len(angles), 3) )
        new_data[:,self.THRESHOLD] = interT(angles)
        new_data[:,self.THETA] = angles
        new_data[:,self.R] = interR(angles)


        #Replace
        self._data = new_data



    @classmethod
    def substract_curves(cls, roc1, roc2):
        """Compare two curves together (p 163).
        For each angle, substract the two radii

        @param roc1: First curve (original one)
        @param roc2: Second curve (bootstraped one)

        @return An array of differences (size of the number of angles)
        """
        #Angles must be identic
        assert N.all(roc1.get_raw_theta() == roc2.get_raw_theta())

        return roc1.get_raw_r() - roc2.get_raw_r()

    def plot(self, linewidth=2, color='blue'):
        """Plot the ROC curve on a graphics"""

        FMR = N.cos(self._data[:, self.THETA])*self._data[:, self.R] + 1
        TMR = N.sin(self._data[:, self.THETA])*self._data[:, self.R]

        plt.xlabel('FMR')
        plt.ylabel('TMR')
        plt.plot(FMR, TMR, linewidth=linewidth, color=color)

        plt.ylim((-0.025,1.025))
        plt.xlim((-0.025,1.025))
        plt.xticks(N.arange(0,1.1,.2))
        plt.yticks(N.arange(0,1.1,.2))

    def tmpplot(self):
        FMR = N.cos(self._data[:, self.THETA])*self._data[:, self.R] + 1
        TMR = N.sin(self._data[:, self.THETA])*self._data[:, self.R]


        plt.plot(FMR, TMR, color='gray',
                linestyle='steps:')

    @classmethod
    def plot_boundary(cls, theta, radius):
        """Plot the boundaries given in parameter."""

        FMR = N.cos(theta)*radius + 1
        TMR = N.sin(theta)*radius

        idx = N.where(FMR < 0)
        FMR[idx] = 0
        idx = N.where(FMR > 1)
        FMR[idx] = 1

        idx = N.where(TMR < 0)
        TMR[idx] = 0
        idx = N.where(TMR > 1)
        TMR[idx] = 1


        plt.plot(FMR, TMR, color='red',
                linewidth=2)


    def get_eer(self):
        """Returns the EER (previously computed).

        TODO: verify if it ONLY works for distance !! Otherwasie path for
        working also on scores ...
        """
        interR = interp1d(self.get_raw_theta()[::-1], self.get_raw_r()[::-1],
                bounds_error=False,
                fill_value=1)

        return 1 - interR(3.0*N.pi/4.0)/N.sqrt(2)

    def get_eer_threshold(self):
        """Returns the thresold at the EER (previously computed).
        """

        interR = interp1d(x=self.get_raw_theta()[::-1],
                y=self.get_raw_thresholds()[::-1],
                bounds_error=True,
                kind='linear')

        return interR(3.0*N.pi/4.0)


if __name__ == "__main__":
    import sys
    sys.path.append('./../')
    import data.banca as banca
    source = banca.Banca()
    #db = source.get_data('g1', 'SURREY_face_nc_man_scale_200')
    #db = source.get_data('g1', 'IDIAP_voice_gmm_auto_scale_25_10_pca')
    db = source.get_data('g1', 'SURREY_face_svm_man_scale_2.00')
    rocscores = ROCScores(db)
    roccurve = rocscores.get_roc()

    polarroc = roccurve.get_polar_representation()

    import matplotlib.pyplot as plt

 #   plt.figure()
 #   polarroc.plot(color='red')
 #   roccurve.plot(color='blue')

    plt.figure()
    print rocscores.get_confidence_interval(alpha=0.1, repet=1000)
    plt.figure()
    rocscores.display_bootstraped_difference()

    plt.show()
    

