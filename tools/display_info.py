#!/usr/bin/env python
# -*- coding: utf-8 -*-

from traits.api import *
from traitsui.api import View, Item, CustomEditor
from chaco.chaco_plot_editor import ChacoPlotItem
from chaco.api import Plot, ArrayPlotData, BarPlot, ArrayDataSource
from enable.component_editor import ComponentEditor

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

import numpy as N
import wx
import sys
sys.path.append('./../')


import data.banca as banca
from errors.roc import ROCScores

"""
The aim of this application is to display various information
an a system.

AUTHOR Romain Giot <romain.giot@ensicaen.fr>
"""

def MakePlot(parent, editor):
    """
    Builds the Canvas window for displaying the mpl-figure
    """
    fig = editor.object.plot2
    panel = wx.Panel(parent, -1)
    canvas = FigureCanvasWxAgg(panel, -1, fig)
    toolbar = NavigationToolbar2Wx(canvas)
    toolbar.Realize()

    sizer = wx.BoxSizer(wx.VERTICAL)
    sizer.Add(canvas,1,wx.EXPAND|wx.ALL,1)
    sizer.Add(toolbar,0,wx.EXPAND|wx.ALL,1)
    panel.SetSizer(sizer)
    return panel



class PerformanceInfo(HasTraits):
    """Get and show the performances of a biometric application."""

    plot1 = Instance(Plot) #FMR/FNMR depending on threshold
    plot2 = Instance(Figure, ())
    axes = Instance(Axes)
    line = Instance(Line2D)

    _draw_pending = Bool(False)

    traits_view = View(ChacoPlotItem('FMR', 'TMR',
        x_label='FMR',
        y_label='TMR',
        color='black',
        title='ROC curve:',
        resizable=True),
        

        Item('plot2', editor=CustomEditor(MakePlot), show_label=False, resizable=True),
        Item('plot1', editor=ComponentEditor(), show_label=False, resizable=True),

        Item('decisionthreshold'),

        width = 900,
        height = 500,
        resizable=True
    )
#    view = View( ('decisionthreshold'),
#            title = 'Performances'
#            )

    def __init__(self, scores):
        """Change the scores."""

        intra = scores.get_genuine_presentations()
        inter = scores.get_impostor_presentations()
        roc = scores.get_roc()

        scores = scores.get_raw_scores()
        self.decisionthreshold = Range(low=float(min(scores)), high=float(max(scores)),
                value=float(scores[len(scores)/2]))

        self.intrascores = intra.get_raw_scores()
        self.interscores = inter.get_raw_scores()

        self.FMR = roc.get_raw_FMR()
        self.TMR = roc.get_raw_TMR()
        self.FNMR = roc.get_raw_FNMR()
        self.thresholds = roc.get_raw_thresholds()


        plotdata = ArrayPlotData(thr=self.thresholds, \
                fmr=self.FMR,
                fnmr=self.FNMR)
        plot1 = Plot(plotdata)
        plot1.plot( ('thr', 'fmr', 'fnmr'), type='line' )
        self.plot1 = plot1


        (nintra, binsintra) = N.histogram(self.intrascores, bins=100)
        (ninter, binsinter) = N.histogram(self.interscores, bins=100)
        plotdata = ArrayPlotData(\
                nintra=nintra/float(len(self.intrascores)),
                ninter=ninter/float(len(self.interscores)),
                binsintra=.5*(binsintra[1:]+binsintra[:-1]), \
                binsinter=.5*(binsinter[1:]+binsinter[:-1]))
        #plot2 = BarPlot(index=ArrayDataSource(.5*(binsintra[1:]+binsintra[:-1])), 
        #                value=ArrayDataSource(nintra/float(len(self.intrascores))))
        #plot2.plot( ('binsintra', 'nintra') )
        #plot2.plot( ('binsinter', 'ninter') )
        #self.plot2 = plot2

        super(PerformanceInfo, self).__init__()

    def _axes_default(self):
        return self.figure.add_subplot(111)
    
    def _line_default(self):
        return self.axes.plot(self.x, self.y)[0]
    
    @cached_property
    def _get_y(self):
        return numpy.sin(self.scale * self.x)
    
    @on_trait_change("x, y")
    def update_line(self, obj, name, val):
        attr = {'x': "set_xdata", 'y': "set_ydata"}[name]
        getattr(self.line, attr)(val)
        self.redraw()
        
    def redraw(self):
        if self._draw_pending:
            return
        canvas = self.figure.canvas
        if canvas is None:
            return
        def _draw():
            canvas.draw()
            self._draw_pending = False
        wx.CallLater(50, _draw).Start()
        self._draw_pending = True
if __name__ == "__main__":
    import data.banca as banca

    source = banca.Banca()
    #TODO do not use ROC as name
    scores = ROCScores(source.get_data('g1', 'SURREY_face_nc_man_scale_200'))

    pi = PerformanceInfo(scores)
    pi.configure_traits()
