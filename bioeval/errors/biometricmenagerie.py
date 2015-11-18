#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AUTHOR Romain Giot <giot.romain@gmail.com>

Analyse scores to display the menagerie.
Each user is symbolised by the mean score intra or inter

TODO use more extensively traits api
"""

from traits.api import HasStrictTraits, Instance, \
        Float, Array, String, Enum, Int
import numpy as np
from scipy.stats.stats import pearsonr, scoreatpercentile
from matplotlib import pyplot as plt


from joblib import Parallel, delayed
import biometricinformation as bi
import roc

def _parallel_mean_score_helper(presentations):
    """Helper to get the genuine scores in a parallel way."""

    scores = presentations.get_raw_scores()
    mean_score = np.mean(scores)

    return mean_score

def _parallel_false_reject_helper(presentations, thr, _type):
    """Helper to get the false reject rate in a parallel way."""

    scores = presentations.get_raw_scores()
    if _type == 'distance':
        return np.sum(scores > thr)
    else:
        return np.sum(scores <= thr)

def _parallel_false_accept_helper(presentations, thr, _type):
    """Helper to get the false accept rate in a parallel way."""

    scores = presentations.get_raw_scores()
    if _type == 'distance':
        return np.sum(scores <= thr)
    else:
        return np.sum(scores > thr)

class BiometricMenagerie(bi.Score):
    """
    Categorize each user among the following categories:

        - Chameleons
        - Phantoms
        - Doves
        - Worms

    This is based on the work:
    Neil Yager and Ted Dunstone. "The Biometric Menagerie", IEEE TRANSACTIONS ON
    PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 32, NO. 2, FEBRUARY 2010.
    """

    _type = Enum('score', 'distance')
    _genuine_scores = Array
    _genuine_ranking = Array

    _impostor_scores = Array
    _impostor_ranking = Array

    _goat_scores = Array
    _wolf_scores = Array
    n_jobs=Int


    def __init__(self, data, scoretype='score'):
        """Build the biometric menagerie

        @param data: The data to use
        @param type: to type of score (distance or score)
        """

        super(BiometricMenagerie, self).__init__(data)

        self.n_jobs=1
        self._type = scoretype

        self._compute_mean_scores()


    def _compute_wolf_and_goat_scores(self):
        """Compute wolf and goat scores for each user.

        goat score: Mean of the 5\% worst scores
            (real definition cannot be applyed: comparaison of each test aginst
            each element of the gallery, and mean of worst score for each
            element of the gallery)

        wolf score: Mean of the best score for each individul impersonated
            (real definition cannot ve applyed: comparison of each elem of the
            gallery against other samples. mean of best comparison for ech elem)


        """
        nb5percent = max(len(self._users_id)*5/100, 1)

        # Get goat score
        goat_scores = []
        for userid in self._users_id:
            presentations = self.get_genuine_presentations_of_user(userid)
            scores = presentations.get_raw_scores()
            scores.sort()

            if self._type == 'score':
                selected_scores = scores[:nb5percent]
            else:
                selected_scores = scores[-nb5percent:]

            goat_scores.append(np.mean(selected_scores))
        self._goat_scores = np.array(goat_scores)


        # Get wolf scores
        wolf_scores = []
        for userid in self._users_id:
            presentations = self.get_impostor_presentations_of_user(userid)
            best_matchings = []

            # loop other the other user to get the best matching
            gallery_ids = presentations.get_raw_gallery_userid(unique=True)
            for gallery_id in gallery_ids:
                assert gallery_id != userid
                presentation_for_gallery = \
                    presentations.filter_indicies( \
                            presentations._data[:,self.GALLERY] == gallery_id)

                scores = presentations.get_raw_scores()
                scores.sort()

                if self._type == 'score':
                    selected_scores = scores[-1]
                else:
                    selected_scores = scores[1]


                best_matchings.append(selected_scores)
            wolf_scores.append(np.mean(best_matchings))
        self._wolf_scores = np.array(wolf_scores)


    def _compute_mean_scores(self):
        """Launch the computation of mean user's genuine or impostor match
        scores.
        """
        # Get users id for correspondance
        self._users_id = self.get_users_id()
        nb25percent = len(self._users_id)*25/100

        # Get genuine scores of each users
        self._genuine_scores = np.asarray(Parallel(n_jobs=self.n_jobs, verbose=1) \
                (delayed(_parallel_mean_score_helper)(self.get_genuine_presentations_of_user(userid)) \
                    for userid in self._users_id))


        # Get impostors scores of each users
        self._impostor_scores = np.asarray(Parallel(n_jobs=self.n_jobs, verbose=1) \
                (delayed(_parallel_mean_score_helper)(self.get_impostor_presentations_of_user(userid)) \
                    for userid in self._users_id))

        # Rank genuine scores by increasing statistics
        self._genuine_ranking = np.argsort(self._genuine_scores)
        if self._type == 'distance':
            self._genuine_ranking = self._genuine_ranking[::-1]

        # Get low and high parts
        self._genuine_low_users = self._genuine_ranking[:nb25percent]
        self._genuine_high_users = self._genuine_ranking[-nb25percent:]


        # Rank impostor scores by increasing statistics
        self._impostor_ranking = np.argsort(self._impostor_scores)
        if self._type == 'distance':
            self._impostor_ranking = self._impostor_ranking[::-1]


        # Get low and high parts
        self._impostor_low_users = self._impostor_ranking[:nb25percent]
        self._impostor_high_users = self._impostor_ranking[-nb25percent:]


    def get_chameleon_indices(self):
        """Return the indicies of chameleon users.
        Chameleon appear similar to others receibing high match scores for all
        verifications
        """

        return np.intersect1d(self._impostor_high_users,self._genuine_high_users)

    def get_phantom_indices(self):
        """Return the indicies of phantoms users.
        Phantoms have very low match scores in any case
        """

        return np.intersect1d(self._impostor_low_users,self._genuine_low_users)

    def get_dove_indices(self):
        """Return the indicies of doves users.
        Dove are the best users. They are pure and recognizable, matching well
        against themselves and poorly against aothers.
        """

        return np.intersect1d(self._impostor_low_users,self._genuine_high_users)

    def get_worm_indices(self):
        """Return the indicies of worms users.
        Worms are the worst users. 
        """

        return np.intersect1d(self._impostor_high_users,self._genuine_low_users)

        return np.intersect1d(self._impostor_high_users,self._genuine_low_users)


    def get_goat_indicies(self):
        """Return the indicies of goat user.
        Speration is done at 30th (or 70th) percentile.

        Source: Exploiting the "Doddington Zoo" Effect in Biometric Fusion
        """

        sorted_genuine = self._genuine_scores.copy()
        sorted_genuine.sort()

        if self._type == 'distance':
            score = scoreatpercentile(sorted_genuine, 70)
            indicies = np.where(self._genuine_scores>score)
        elif self._type == 'score':
            score = scoreatpercentile(sorted_genuine, 30)
            indicies = np.where(self._genuine_scores<score)

        return indicies[0]


    def get_lamb_indicies(self):
        """Return the indicies of lambs users.

        BUG: seems to not work
        """

        lambs_scores = []
        for userid in self._users_id:
            presentations = self.get_impostor_presentations_of_user(userid)
            best_matchings = []

            # loop other the other user to get the best matching
            gallery_ids = presentations.get_raw_gallery_userid(unique=True)
            for gallery_id in gallery_ids:
                assert gallery_id != userid

                # Select only one gallery user
                indicies = presentations._data[:,self.GALLERY] ==  gallery_id
                impscores = presentations.get_raw_scores()[indicies]

                if self._type == 'score':
                    selected_score = np.min(impscores)
                elif self._type == 'distance' :
                    selected_score = np.max(impscores)
                else:
                    raise "Impossible"


                best_matchings.append(selected_score)
            lambs_scores.append(np.mean(best_matchings))

        lamb_score_sorted = np.sort(lambs_scores)
        if self._type == 'distance':
            score = scoreatpercentile(lamb_score_sorted, 10)
            indicies = np.where(self._genuine_scores<score)
        elif self._type == 'score':
            score = scoreatpercentile(lamb_score_sorted, 90)
            indicies = np.where(self._genuine_scores>score)
        else:
            raise "Impossible"

        return indicies[0]




    def get_probability_existence(self, category=None):
        """Returns the probablity existence of each category or of the selected
        category.
        The null hypothesis states that the probability of belonging to one category is 1/16.

        Parameters
        ==========

         - category : str or None
            Name of the category if we want one.
            None for all the categories

        """
        from scipy.misc import comb

        # Method to call for each famly
        calls = {
                'chameleons' : self.get_chameleon_indices,
                'phantoms': self.get_phantom_indices,
                'doves': self.get_dove_indices,
                'worms': self.get_worm_indices,
                }

        # Build list of categories to search
        if category==None:
            categories = calls.keys()
        else:
            categories = [category]

        results = {}
        for category in categories:
            indices =  calls[category]()
            c = len(indices)
            n = len (self._users_id)
            p = 1.0/16.0

            prob = 0
            for i in range(c, n+1):
                prob = prob + comb(n, i)*(p**i)*((1-p)**(n-i))

            results[category] = {'prob': prob,
                                 'alpha':0.05,
                                 'H0 rejected':  prob<0.05,
                                 'sup?' : (p*n)<c,
                                 'percent': c*100.0/float(n)}

        return results

    def get_eer_and_threshold(self):
        """Compute the EER and the threshold to obtain it."""

        scores = roc.ROCScores(self._data, self._type)
        curve1 = scores.get_roc()
        curve = curve1.get_polar_representation()

        eer = curve.get_eer()
        thr = curve.get_eer_threshold()

        return eer, thr

    def get_correlation_between_mean_score_and_error(self):
        """Compute the correlation between:

         * mean genuine score and false reject count
         * mean impostor score and false acceptance count


        False reject count and flase reject count is computed thanks to a global threshold.
        This threshold is the threshold giving the EER.
        Correlation is computed using Pearson correlation factor.
        """

        # We need the EER threshold
        eer, thr = self.get_eer_and_threshold()

        # We need to compute error rate of each user
        # Get genuine reject of each users
        fr = np.asarray(Parallel(n_jobs=self.n_jobs, verbose=1) \
                (delayed(_parallel_false_reject_helper)(self.get_genuine_presentations_of_user(userid),
                    thr, self._type) \
                    for userid in self._users_id))


        # Get impostors accept of each users
        fa = np.asarray(Parallel(n_jobs=self.n_jobs, verbose=1) \
                (delayed(_parallel_false_accept_helper)(self.get_impostor_presentations_of_user(userid),
                    thr, self._type) \
                    for userid in self._users_id))



        #compute the correlations
        return pearsonr(fr, self._genuine_scores)[0], pearsonr(fa,
                self._impostor_scores)[0], eer

    def get_menagerie_members(self):
        """Returns a dictionnary contining, for each type of animals, the id of
        the users.
        Note that we use ID and not indice!
        """

        # Method to call for each famly
        calls = {
                'chameleons' : self.get_chameleon_indices,
                'phantoms': self.get_phantom_indices,
                'doves': self.get_dove_indices,
                'worms': self.get_worm_indices,
                }

        results = {}

        # Catch the user id of each family
        others = self._users_id
        for family in calls:
            members_indices = calls[family]()
            members_id = self._users_id[members_indices]
            others = np.setdiff1d(others, members_id)

            results[family] = members_id


        # Build the other family
        results['others'] = others
        return results


    def display_wolf_goat_plots(self):
        """Display the wolf and goat plot."""

        plt.scatter([1]*len(self._goat_scores),self._goat_scores)
        plt.scatter([2]*len(self._wolf_scores),self._wolf_scores)

        plt.title('Goat and Wolf plot')
        plt.xlabel('Animal')
        plt.ylabel('Mean worst genuine score/ Mean best impostor score')
        plt.xticks([1,2], ['Goat', 'Wolf'])

    def display_menagerie(self, legend=True):
        """Display meanagerie on screen"""

        chameleons = self.get_chameleon_indices()
        phantoms = self.get_phantom_indices()
        doves = self.get_dove_indices()
        worms = self.get_worm_indices()

        goats = self.get_goat_indicies()
        lambs = self.get_lamb_indicies()

        others = np.arange(len(self._users_id))
        others = np.setdiff1d(others, chameleons)
        others = np.setdiff1d(others, phantoms)
        others = np.setdiff1d(others, doves)
        others = np.setdiff1d(others, worms)

        if True:
            if len(chameleons)>0:
                plt.scatter(
                    self._genuine_scores[chameleons],
                    self._impostor_scores[chameleons],
                    label='chameleon',
                    color='green',
                    marker='^')

            if len(phantoms)>0:
                plt.scatter(
                    self._genuine_scores[phantoms],
                    self._impostor_scores[phantoms],
                    label='phantom',
                    color='red',
                    marker='<')

            if len(doves)>0:
                plt.scatter(
                    self._genuine_scores[doves],
                    self._impostor_scores[doves],
                    label='dove',
                    color='blue',
                    marker='>')


            if len(worms)>0:
                plt.scatter(
                    self._genuine_scores[worms],
                    self._impostor_scores[worms],
                    label='worm',
                    color='black',
                    marker='v')

            if len(others)>0:
                plt.scatter(
                    self._genuine_scores[others],
                    self._impostor_scores[others],
                    label='other',
                    color='gray',
                    marker='o')

        else:
            if len(goats)>0:
                plt.scatter(
                    self._genuine_scores[goats],
                    self._impostor_scores[goats],
                    label='goat',
                    facecolor='none',
                    edgecolor='black',
                    marker='o',
                    s= 70)
    
   
            if len(lambs)>0:
                print lambs
                plt.scatter(
                   self._genuine_scores[lambs],
                    self._impostor_scores[lambs],
                    label='lamb',
                    facecolor='none',
                    edgecolor='black',
                    marker='s',
                    s= 80)

        # Bordures
        if self._type == 'score':
            plt.axvline(linestyle=':', x=max(self._genuine_scores[self._genuine_low_users ]))
            plt.axvline(linestyle=':', x=min(self._genuine_scores[self._genuine_high_users ]))

            plt.axhline(linestyle=':', y=max(self._impostor_scores[self._impostor_low_users ]))
            plt.axhline(linestyle=':', y=min(self._impostor_scores[self._impostor_high_users ]))
        else:
            plt.axvline(linestyle=':', x=min(self._genuine_scores[self._genuine_low_users ]))
            plt.axvline(linestyle=':', x=max(self._genuine_scores[self._genuine_high_users ]))

            plt.axhline(linestyle=':', y=min(self._impostor_scores[self._impostor_low_users ]))
            plt.axhline(linestyle=':', y=max(self._impostor_scores[self._impostor_high_users ]))

        plt.xlabel('Average genuine match score')
        plt.ylabel('Average impostor score')
        plt.title('zooplot')

        if legend:
            plt.legend(loc=0)

if __name__ == "__main__":
    import sys
    sys.path.append('./../')
    import data.banca as banca
    source = banca.Banca()
    #db = source.get_data('g1', 'SURREY_face_nc_man_scale_200')
    #db = source.get_data('g1', 'IDIAP_voice_gmm_auto_scale_25_10_pca')
    db = source.get_data('g1', 'SURREY_face_svm_man_scale_2.00')

    bm = BiometricMenagerie(db)
    plt.figure()
    bm.display_menagerie()


    import data.xmvts
    xmvts = data.xmvts.XMVTS()
    db = xmvts.get_data('PI', 'dev', 1)

    bm = BiometricMenagerie(db)
    plt.figure()
    bm.display_menagerie()


    plt.show()
