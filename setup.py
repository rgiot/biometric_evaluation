#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Biometric evaluation

"""

# imports
import os, sys
from distutils.core import setup



if not os.path.exists('bioeval/data/data.zip'):
    sys.stderr("You need to download the XMVTS databasefile and store it in bioeval/data/data.zip")
    sys.exit(-1)
if not os.path.exists('bioeval/data/snapshot1.zip'):
    sys.stderr("You need to download the BANCA databasefile and store it in bioeval/data/snapshot1.zip")
    sys.exit(-1)

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
      package_data={'bioeval.data': ['data.zip', 'snapshot1.zip']}
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

