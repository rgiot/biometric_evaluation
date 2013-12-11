#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AUTHOR Romain Giot <giot.romain@gmail.com>
LICENCE GPL

This module contains the code to compute EER statistic information

TODO include some parts of ROC.py
"""

import matplotlib.pyplot as plt
import numpy as np

from utils import get_confidence_limits

def confidence_interval_of_two_independant_eer(roc1, roc2, alpha, verbose=True):
    """Get the confidence interval of two independant eers'.
    The confidence interval of the EERS must already be computed.
    page 206

    @param roc1: First system
    @param roc2: Second system
    @param alpha: Confidence interval
    """

    assert roc1.estimated_eer and roc1.bootstraped_eers, \
            "You must call get_confidence_interval before for roc1"
    assert roc2.estimated_eer and roc2.bootstraped_eers, \
            "You must call get_confidence_interval before for roc2"

    e = np.array(roc1.bootstraped_eers)  - np.array(roc2.bootstraped_eers) \
            - (roc1.estimated_eer - roc2.estimated_eer )

    e_L, e_U = get_confidence_limits(e, alpha)
    base = roc1.estimated_eer - roc2.estimated_eer


    if verbose:
        print "Comparison of two independant EER"
        print "================================="
        print "Estimated difference EER1-EER2: %0.6f" % base
        print "Lower boundary: %0.6f" % ( base - e_U)
        print "Upper boundary: %0.6f" % ( base - e_L)

    return base, base - e_U, base - e_L, e

def hypothesis_test_for_difference_of_two_independant_eer(roc1, roc2, alpha, verbose = True):
    """Verifiy if the first eer is better than the second.
    page 191.

    HO: EER1=EER2
    H1: EER1<EER2
    """

    #alpha is unecessary in this case
    base, lower, upper, e = confidence_interval_of_two_independant_eer(roc1, roc2, alpha)

    p = ( 1 + np.sum(e <= (roc1.estimated_eer - roc2.estimated_eer ))) / float(len(e)+1)
    better = p < alpha

    if verbose:
        print "EER comparison"
        print "=============="
        print "Estimated EER difference: %0.6f"  % (roc1.estimated_eer-roc2.estimated_eer)
        print "  H0: EER1=EER2"
        print "  H1: EER1<EER2"
        print "p-value: %0.6f" % p
        if better:
            print "Rejection of H0 => EER < EER0"
            print "EER1 %0.6f is significantly less than EER2 %0.6f" \
                    % (roc1.estimated_eer, roc2.estimated_eer)
        else:
            print "No rejection of H0 => EER1 = EER2"
        print

    return better

def display_confidence_interval_of_two_independant_eer(roc1, roc2, alpha):
    """Display the distribution of the difference of EER between two
    systems."""

    base, lower, upper, e = confidence_interval_of_two_independant_eer(roc1, roc2, alpha)
    plt.figure()
    plt.hist(e)
    n, bins, patches = plt.hist(e, bins=100, fill=False)
    plt.vlines(upper, 0, max(n), label='upper', linestyle='dotted')
    plt.vlines(lower, 0, max(n), label='lower', linestyle='dashed')


def hypothesis_test_for_single_eer(roc, eer0, level, verbose = True):
    """Compare the EER to a value and returns true if it is better.
    page 187

    H0: EER=eer0
    H1: EER<eer0

    @param roc: The roc curve
    @param eer0: EER value we want to beat
    @param level: significance level
    """

    assert roc.estimated_eer and roc.bootstraped_eers, "You must call get_confidence_interval before"

    e = roc.bootstraped_eers - roc.estimated_eer + eer0


    p = (1 + np.sum(e<=roc.estimated_eer))/float(len(roc.bootstraped_eers)+1)
    better = p < level

    if verbose:
        print "EER comparison"
        print "=============="
        print "Estimated EER: %0.6f" % roc.estimated_eer
        print "Comparison EER0: %0.6f" % eer0
        print "  H0: EER=EER0"
        print "  H1: EER<EER0"
        print "p-value: %0.6f" % p
        if better:
            print "Rejection of H0 => EER < EER0"
            print "EER is significantly less than %0.6f" %eer0
        else:
            print "No rejection of H0 => EER = EER0"
        print

    return better



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

    roccurve1 = rocscores1.get_roc()
    roccurve2 = rocscores2.get_roc()

    plt.figure()
    rocscores1.get_confidence_interval(alpha=0.1, repet=1000)
    plt.figure()
    rocscores2.get_confidence_interval(alpha=0.1, repet=1000)

    hypothesis_test_for_single_eer(rocscores1, 0.05, 0.05)
    plt.figure()
    display_confidence_interval_of_two_independant_eer(rocscores1, rocscores2, 0.1)

    hypothesis_test_for_difference_of_two_independant_eer(rocscores1,
            rocscores2, 0.05)
    plt.show()
