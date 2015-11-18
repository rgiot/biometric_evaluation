#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AUTHOR Romain Giot <giot.romain@gmail.com>
"""
import sys
import numpy as N

import errors.roc as roc
import pylab

def main():
    """Main application.
    @todo read parameters from args
    """

    M = 20
    nb_thresholds = 1000
    nb_angles = 40
    alpha = 0.01
    fname = sys.argv[1]

    print "M=%d\nthreshold=%d\nangles=%d\nalpha=%f\nfile=%s" % \
     (M, nb_thresholds, nb_angles, alpha, fname)

    data = None
    try:
        data = N.loadtxt(fname)
    except:
        data = N.loadtxt(fname, delimiter=',')
    rocScore = roc.ROCScores(data)
    rocScore.get_confidence_interval(alpha=alpha, repet=M)
    pylab.show()

if __name__ == "__main__":
    main()
    
