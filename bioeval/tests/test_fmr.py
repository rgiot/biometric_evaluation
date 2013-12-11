#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the FNMR compmutation.

AUTHOR Romain Giot <giot.romain@gmail.com>
"""

import sys
sys.path.append('./../')

import errors.roc as roc
import data.banca as banca

import unittest
import matplotlib.pylab as plt

class TestFMR(unittest.TestCase):
    """Test the running of the FNMR curve utility."""

    def setUp(self):
        """Read the database."""

        source = banca.Banca()
        db1 = source.get_data('g1', 'IDIAP_voice_gmm_auto_scale_25_10_pca')
        db2 = source.get_data('g1', 'SURREY_face_nc_man_scale_200')
        self.roc1 = roc.ROCScores(db1)
        self.roc2 = roc.ROCScores(db2)
        self.curve1 = self.roc1.get_roc()
        self.curve2 = self.roc2.get_roc()

    def test_threshold(self):
        """Test the computation of the threshold."""
        thr = self.curve1.get_threshold_for_FMR(0.7)
        print thr

if __name__ == '__main__':
    unittest.main()
