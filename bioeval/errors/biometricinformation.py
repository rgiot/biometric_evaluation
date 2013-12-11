# -*- coding: utf-8 -*-

"""Store common things about biometric information.

AUTHOR Romain Giot <giot.romain@gmail.com>
LICENCE GPL
"""

from traits.api import HasTraits, Array
import numpy as N
from matplotlib import pyplot as plt

class AbstractBiometricData(HasTraits):
    """Manage biometric data."""

    # Content of the object
    _data = Array()
    GALLERY = 0
    PROBE = 1
 
    def __init__(self, data):
        """Load the data from a filename or numpy array.

        @param data: if it is a string, this is the filename
                     of the file to load. Otherwise it is the numpy
                     array.
        """

        super(AbstractBiometricData, self).__init__()

        # Read from string, otherwise store data
        if isinstance(data, str):
            self._data = N.loadtxt(data)
        else:
            self._data = data

    def get_users_id(self):
        return self.get_raw_gallery_userid(unique=True)

    def get_raw_gallery_userid(self, unique=False):
        """REturns the list of gallery userid.
        @param unique: if True, only one instance.
        """

        userids = self._data[:,self.GALLERY]
        if unique:
            return N.unique(userids)
        else:
            return userids

    def get_raw_prob_userid(self, unique=False):
        """REturns the list of gallery userid.
        @param unique: if True, only one instance.
        """

        userids = self._data[:,self.PROBE]
        if unique:
            return N.unique(userids)
        else:
            return userids



    def get_number_of_users(self):
        """Returns the number of users in the dataset."""
        return len(self.get_users_id())

    def get_presentations_of_user(self, userid):
        """Returns the presentations of the user.
        The result is an array like that:
        user presentation #    error

        @param userid: Id of the required user
        """
        indicies = N.where(self._data[:,0] == userid)
        return self._data[indicies]



    def get_presentations_number_of_user(self, userid):
        """Returns the number of presentation for the required user.

        @param userid: Id of the required user
        """
        return len(self.get_presentations_of_user(userid))

    def get_mean_number_of_presentations(self):
        """Returns the mean number of presentations."""

        nb_presentations = \
                [self.get_presentations_number_of_user(user) \
                    for user in self.get_users_id() ]

        return N.mean(nb_presentations)

    def get_number_of_presentations(self):
        """Returns the number of samples."""
        return self._data.shape[0]


    def bootstrap(self):
        """Return a bootstraped view of the data """
        raise Exception('Not Implemented')

    def get_effective_sample_size(size):
        """Returns the effective sample size."""
        raise Exception('Not Implemented')

    def get_data(self):
        """Returns the data"""
        return self._data


    def __str__(self):
        return str(self.get_data())

class AbstractComparisonData(AbstractBiometricData):
    """Manage information created with sample comparison against a template.
    The data of of this type:
      * gallery_user
      * prob_user
      * presentation_number
      * score or decision error
    """

    GALLERY = 0
    PROBE = 1
    PRESENTATION = 2
    RESULTS = 3


    def __init__(self, data):
        super(AbstractComparisonData, self).__init__(data)

    def get_raw_results(self):
       """Return the results only."""
       return self._data[:,self.RESULTS]


class Decision(AbstractComparisonData):
    """Store biometric decision.
    Assume that it is considered as decision errors when comuting error rate.
    """

    def get_estimated_error(self):
        """Return the estimated errors"""
        return N.mean(self._data[:,self.RESULTS])

class Score(AbstractComparisonData):
    """The aim of this class it to store genuine and impostors scores.
    Data are stored like that:

      * gallery_user 
      * prob_user 
      * presentation_number
      * score
    """

    def __init__(self, data):
        super(Score, self).__init__(data)

    def filter_indicies(self, indicies):
        """Return only scores for the indicies"""
        return Score( self._data[indicies, :][0])

    def inferior_to_threhold(self, threshold):
        """Return an object of result of comparison.
        data <= threshold"""

        res = N.copy(self._data)
        res[:,self.RESULTS] = res[:,self.RESULTS] <= threshold
        return Decision(res)

    def superior_to_threhold(self, threshold):
        """Return an object of result of comparison.
        data > threshold"""

        res = N.copy(self._data)
        res[:,self.RESULTS] = res[:,self.RESULTS] > threshold
        return Decision(res)

    def min(self):
        """Return the min value of the set"""
        return N.min(self._data[:,self.RESULTS])

    def max(self):
        """Return the max value of the set"""
        return N.max(self._data[:,self.RESULTS])

    def get_raw_scores(self):
       """Return the scores only."""
       return self.get_raw_results()

    def get_genuine_presentations(self):
        """Returns the genuine presentations:
        $i==k$ for $Y_{ikl}.
        """
        indicies = N.where(self._data[:,self.GALLERY] == self._data[:,self.PROBE])
        return self.filter_indicies(indicies)

    def get_impostor_presentations(self):
        """Returns the impostor's presentations
        $i!=k$ for $Y_{ikl}.
        """
        indicies = N.where(self._data[:,self.GALLERY] != self._data[:,self.PROBE])
        return self.filter_indicies(indicies)

    def get_genuine_presentations_of_user(self, userid):
        """Return the list of genuine presentation for the required user.

        In this case GALLERY == PROBE == userid

        Parameters
        ----------
            - userid: int
                User identifier
        """

        indicies = N.where(
                N.logical_and(
                    self._data[:, self.GALLERY] == self._data[:,self.PROBE],
                    self._data[:, self.GALLERY] == userid
                    )
                )
        return self.filter_indicies(indicies)


    def get_impostor_presentations_of_user(self, userid):
        """Return the list of impostor presentation of the required user.
        It is the presentations done by the user to attack an other one.

        In this case GALLERY != PROBE  and PROBE == userid

        Parameters
        ----------
            - userid: int
                User identifier
        """

        indicies = N.where(
                N.logical_and(
                    self._data[:, self.GALLERY] != self._data[:,self.PROBE],
                    self._data[:, self.PROBE] == userid
                    )
                )
        return self.filter_indicies(indicies)

    def display_distributions(self, merged=True, nbbins=100, interpolation=True):
        """Display the distribution of the scores on a curve.
        @param merged: if true display all the scores together, otherwise,
        display a distribution per user
        """

        assert merged == True, 'Need to code for False'

        def get_histogram(scores):
            """Compute the probability of each value"""
            hist, edges = N.histogram(scores, nbbins)
            return hist/float(hist.sum()), edges

        # Get the scores
        genuine = self.get_genuine_presentations().get_raw_scores()
        impostor = self.get_impostor_presentations().get_raw_scores()


        if not interpolation:
            # Get the histograms
            genuine_hist, genuine_edges = get_histogram(genuine)
            impostor_hist, impostor_edges = get_histogram(impostor)

            
            # Display the histograms
            plt.plot( (genuine_edges[:-1] + genuine_edges[1:])/2.0,
                       genuine_hist,
                       label='genuine', color='blue',linewidth=2)
            plt.plot( (impostor_edges[:-1] + impostor_edges[1:])/2.0,
                       impostor_hist,
                       label='impostor', color='red',linewidth=2)

        else:
            from scipy.stats import gaussian_kde
            kdeg = gaussian_kde(genuine)
            kdei = gaussian_kde(impostor)

            scores = N.linspace(self.min(), self.max(), 500, True)
            probg =  kdeg(scores)
            probi =  kdei(scores)

            plt.plot(scores, probg, label='genuine (interpolated)',
                    color='blue', linewidth=2)
            plt.plot(scores, probi, label='impostor (interpolated)',
                    color='red', linewidth=2)

        plt.xlabel('Scores')
        plt.ylabel('Probability')
        plt.legend(loc=0)



