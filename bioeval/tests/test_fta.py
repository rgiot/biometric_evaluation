#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('./../')

import errors.fta 
import unittest
"""
Test the fta module
AUTHOR Romain Giot <giot.romain@gmail.com>
"""

class TestFTA(unittest.TestCase):
    """Test case for the Failure to Acquire Rate."""

    def test_a_trier(self):
        fta = errors.fta.FTA('../data/GREYC_FT.txt')
        boostrap = fta.get_confidence_interval(method='boostrap')
        print boostrap

        assert fta.get_error_estimation() == fta.get_error_estimation_slow()
        print fta.can_use_large_sample_approach()
        print fta.get_confidence_interval()
        largesample = fta.get_confidence_interval(method='largesample')


        print largesample
        print boostrap

if __name__ == '__main__':
    unittest.main()
