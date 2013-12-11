#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AUTHOR Romain Giot <romain.giot@ensicaen.fr>

TODO:
     - Write a GUI component for selecting scores
     - Write a GUI component for selecting bootrsaping info
"""

from traits.api import  \
        HasTraits, Instance, \
        File, String, Enum, List, Int, Range, Bool,\
        on_trait_change


alpha_type = Range(0.0,
                  1.0,
                  value=0.05,
                  label='Alpha',
                  desc='Precision of the confidence interval',
                  )

nb_bootstrap_type = Int(10,
                      label='Boostraps',
                      desc='Number of time the boostrap must been done')

type_score_type =  Enum('Distance','Score',
                    label='Type',
                    desc='The type of measures (scores or distances)')
