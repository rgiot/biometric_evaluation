#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the FNMR compmutation.

AUTHOR Romain Giot <giot.romain@gmail.com>
"""

import sys
sys.path.append('./../')

import errors.roc as roc
import errors.fnmr as fnmr
import data.banca as banca

import unittest
import matplotlib.pylab as plt

class TestFNMR(unittest.TestCase):
    """Test the running of the FNMR curve utility."""

    def setUp(self):
        """Read the database."""

        source = banca.Banca()
        db1 = source.get_data('g1', 'SURREY_face_svm_man_scale_2.00') #-0.10
        db2 = source.get_data('g1', 'UC3M_voice_gmm_auto_scale_10_100') #-0.30
        self.roc1 = roc.ROCScores(db1)
        self.roc2 = roc.ROCScores(db2)
        self.curve1 = self.roc1.get_roc()
        self.curve2 = self.roc2.get_roc()

    def test_computation(self):
        """Test the computation."""
        fnmr1 = fnmr.FNMR(self.roc1.get_genuine_presentations().inferior_to_threhold(-0.10).get_data())
        error = fnmr1.get_error_estimation()

        print error

        fnmr1a = fnmr1.bootstrap()
        print fnmr1a.get_error_estimation()

        print fnmr1.get_correlation_estimation()
        print fnmr1.get_confidence_interval_largesample(alpha=0.05)
        print fnmr1.get_confidence_interval_bootstrap(alpha=0.05)


        fnmr2 = \
        fnmr.FNMR(self.roc2.get_genuine_presentations().inferior_to_threhold(-0.30).get_data())
        error = fnmr2.get_error_estimation()

        print error

if __name__ == '__main__':
    unittest.main()
