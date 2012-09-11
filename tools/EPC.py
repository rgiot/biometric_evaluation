#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AUTHOR Romain Giot <romain.giot@ensicaen.fr>
LICENCE GPL
"""

from traits.api import  \
        HasTraits, Instance, \
        File, String, Enum, List, Int, Range, Bool,\
        on_trait_change
from traitsui.api import View, Item, VGroup, HGroup
from traitsui.menu import OKCancelButtons

import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append('./../')
from errors.roc import ROCScores
from errors.EPC import BootstrapedEPC, EPC

from common_gui import *



type_score_type =  Enum('Distance','Score',
                    label='Type',
                    desc='The type of measures (scores or distances)')



class EPCManager(HasTraits):
    """Manage computation off the EPC curve on two datasets
    """
    # Reference to the ROC curve 1
    rocfname1 = File('/home/romain/Travail/Code/Schuckers/tools/Scores_A_16573049.txt',
                    label='Devel set',
                    desc='The file containing the devel set')

    # Reference to the ROC curve 2
    rocfname2 = File('/home/romain/Travail/Code/Schuckers/tools/Scores_B_16573049.txt',
                    label='Test set',
                    desc='The file containing the test set')


    # Type of information
    typeinfo = type_score_type

    # Number of bootstrapq
    #nbbootstrap = nb_bootstrap_type

    # Alpha
    #alpha = alpha_type

    # Container of the ROC scores
    rocscores1 = Instance(ROCScores)

    # Container of the ROC scores
    rocscores2 = Instance(ROCScores)

    # Use bootstrap
    usebootstrap = Bool(True,
                        label= 'Use bootstrap',
                        desc='Check if you want to use bootstrap')

    # Number of bootstrapq
    nbbootstrap = nb_bootstrap_type

    # Alpha
    alpha = alpha_type

    # View of the object
    view = View(
            VGroup(
                'rocfname1',
                'rocfname2',
                'typeinfo',
                'usebootstrap',
                'nbbootstrap',
                'alpha',
                ),
                buttons=OKCancelButtons,
                title = 'EPC curve'
            )

    def _rocfname1_changed(self, value):
        #TODO manage erroneous files
        db = np.loadtxt(self.rocfname1)
        self.rocscores1 = ROCScores(db)

    def _rocfname2_changed(self, value):
        #TODO manage erroneous files
        db = np.loadtxt(self.rocfname2)
        self.rocscores2 = ROCScores(db)


    def compute(self):
        """Compute all the necessary things for the ROC curve."""
        if not self.rocscores1:
            self._rocfname1_changed(None)
        if not self.rocscores2:
            self._rocfname2_changed(None)

        if self.usebootstrap:
            epc = BootstrapedEPC(devel=self.rocscores1, test=self.rocscores2)
            epc.compute_epc(nb_bootstrap=self.nbbootstrap,
                    confidence=self.alpha,
                    alpha_step=100)
            
        else:
            epc = EPC(devel=self.rocscores1, test=self.rocscores2)
            epc.compute_epc()
            plt.figure()
            epc.display_epc()



        plt.show()



mana = EPCManager()
if mana.configure_traits():
    mana.compute()

