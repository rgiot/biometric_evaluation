#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Biometric evaluation

"""

# imports
from distutils.core import setup

setup(name='Bioeval',
      version='0.1',
      description='Evalaution of biometrics authentication methods',
      author='Romain Giot',
      author_email='romain.giot@u-bordeaux1.fr',
      url='https://github.com/rgiot/biometric_evaluation',
      packages=['bioeval', 
                'bioeval.data', 
                'bioeval.errors', 
                'bioeval.graph', 
                'bioeval.tests', 
                'bioeval.tools'],
     )


# metadata
__author__ = 'Romain Giot'
__copyright__ = 'Copyright 2013, LaBRI'
__credits__ = ['Romain Giot']
__licence__ = 'GPL'
__version__ = '0.1'
__maintainer__ = 'Romain Giot'
__email__ = 'romain.giot@u-bordeaux1.fr'
__status__ = 'Prototype'

