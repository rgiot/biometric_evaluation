#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.stats import scoreatpercentile

"""
AUTHOR Romain Giot <giot.romain@gmail.com>

This module embed various utility functions.
"""

def get_confidence_limits(values, confidence):
    """Returns the lower and upper limit for the required values at the confidence
    level.

    @param values: Values from which we want to extract an upper and lower limit
    @param confidence: Confidence level (0< confidence <1)
    @returns lower, upper
    """

    upper = scoreatpercentile(values, 100*(1.0-confidence/2.0))
    lower = scoreatpercentile(values, 100*(confidence/2.0))

    return lower, upper
