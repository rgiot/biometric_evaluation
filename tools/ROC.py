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
from traitsui.api import View, Item, VGroup, HGroup, Handler
from traitsui.menu import OKCancelButtons

import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append('./../')
from errors.roc import ROCScores
from errors.RocConfidenceRegionOfDifference import ROCConfidenceRegionOfDifference
from data import banca
from common_gui import *



banca_list = ["g1_%s" % val for val in banca._LABEL_TO_POS.keys()]
banca_list.extend(["g2_%s" % val for val in banca._LABEL_TO_POS.keys()])
banca_list.sort()
banca_choose = Enum(banca_list)
class RocManager(HasTraits):

    # Reference to the ROC curve
    rocfname = File(label='Score file',
                    desc='The file containing the ROC curve scores')
    bancascores = banca_choose

    # Type of information
    typeinfo = type_score_type

    # Number of bootstrapq
    nbbootstrap = nb_bootstrap_type

    # Alpha
    alpha = alpha_type

    # Container of the ROC scores
    rocscores = Instance(ROCScores)

    # Use bootstrap
    usebootstrap = Bool(True,
                        label= 'Use bootstrap',
                        desc='Check if you want to use bootstrap')



    @on_trait_change('rocfname', 'typeinfo', 'nbbootstrap', 'alpha',
            'usebootstrap', 'bancascores')
    def update(self, name, value):
        if name in [ 'rocfname', 'typeinfo', 'bancascores']:
            if self.rocfname:
                #TODO manage erroneous files
                db = np.loadtxt(self.rocfname)
                self.rocscores = ROCScores(db, self.typeinfo.lower())
            else:
                 bancadb = banca.Banca()
                 db = bancadb.get_data(self.bancascores[:2], self.bancascores[3:])
                 print self.bancascores[:2]
                 print self.bancascores[3:]
                 self.rocscores = ROCScores(db, 'score')


    def compute(self):
        """Compute all the necessary things for the ROC curve."""

        if self.usebootstrap:
            # We ask to use bootstrap
            self.rocscores.get_confidence_interval(\
                alpha=self.alpha,
                repet=self.nbbootstrap
                )
        else:
            roccurve = self.rocscores.get_roc()
            plt.figure()
            roccurve.plot()

            roccurve.get_polar_representation().plot()

            plt.figure()
            roccurve.plot_det()

            plt.figure()
            self.rocscores.display_distributions()

            plt.figure()
            roccurve.plot_FMR_FNMR_versus_threshold()
            plt.show()



class RocManagerHandler(Handler):
    """Handler of the ui of displaying ROC."""

    def setattr(self, info, object, name, value):
        Handler.setattr(self, info, object, name, value)
        info.object._updated=True

    def object__updated_changed(self, info):
        if info.initialized:
            info.ui.title += "*"


# View of the object
RocManagerView = View(
            VGroup(
                HGroup('rocfname', 'bancascores'),
                'typeinfo',
                'usebootstrap',
                'nbbootstrap',
                'alpha',
                ),
                buttons=OKCancelButtons,
                title = 'ROC curve plotter',
                handler=RocManagerHandler()
            )
def single_roc():
    mana = RocManager()
    if mana.configure_traits(view=RocManagerView):
        mana.compute()


class DoubleRocManager(HasTraits):
    """Manage computation on two different ROC curves
    """
    # Reference to the ROC curve 1
    rocfname1 = File('/home/romain/Travail/Code/Schuckers/tools/Scores_A_16573049.txt',
                    label='File name 1',
                    desc='The file containing the first ROC curve')

    # Reference to the ROC curve 2
    rocfname2 = File('/home/romain/Travail/Code/Schuckers/tools/Scores_B_16573049.txt',
                    label='File name 2',
                    desc='The file containing the second ROC curve')


    # Type of information
    typeinfo = type_score_type

    # Number of bootstrapq
    nbbootstrap = nb_bootstrap_type

    # Alpha
    alpha = alpha_type

    # Container of the ROC scores
    rocscores1 = Instance(ROCScores)
                  

    # Container of the ROC scores
    rocscores2 = Instance(ROCScores)

    # View of the object
    view = View(
            VGroup(
                'rocfname1',
                'rocfname2',
                'typeinfo',
                'nbbootstrap',
                'alpha',
                ),
                buttons=OKCancelButtons
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


        region = ROCConfidenceRegionOfDifference(\
                self.rocscores1,
                self.rocscores2)
        region.get_confidence_region(nb_iter=self.nbbootstrap, alpha=self.alpha)
        plt.show()





def multiple_roc():
    mana = DoubleRocManager()
    if mana.configure_traits():
        mana.compute()

single_roc()
multiple_roc()

plt.show()
