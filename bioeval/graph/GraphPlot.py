#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph based representation of biometric performance.

Represents the users and their relations in a directed graph.
Each node represents a user.
Each edge represents a relation between two users.
There is a relation between two users, when the mean score between the two users is higher to a threshold
(or lower for a distance based system).
Such kind of graph allow to visually analyse the relations between users.

"""

# imports
from enthought.traits.api import HasStrictTraits, Instance, \
        Float, Array, String, Enum
import matplotlib.pyplot as plt
import networkx
import numpy as np
import errors.biometricinformation as bi
import errors.roc as roc

# code
class GraphPlot(bi.Score):

    _type = Enum('score', 'distance')
    
    def __init__(self, data, scoretype='score'):
        """Build the biometric menagerie

        @param data: The data to use
        @param type: to type of score (distance or score)
        """

        super(GraphPlot, self).__init__(data)

        self._type = scoretype



    def display_graph(self, thr):
        """Compute all the things for the graph for a givben threshold"""

        DG = networkx.DiGraph()

        # Add each user and compute its HTER
        for user_source in self.get_users_id():
            scores = self.filter_indicies(np.where(self.get_data()[:,self.GALLERY] == user_source))
                                      
            # Compute the individual HTER
            HTER = roc.ROCScores(scores.get_data(), self._type).get_weighted_error_at_threshold(thr, 0.5)
            if np.isnan(HTER):
                HTER = 1
            DG.add_node(int(user_source), HTER=HTER, node_color=HTER)

        #TODO compute in parallel
        for user_source in self.get_users_id():
            for user_dest in self.get_users_id():
                # Do not compare the same user
                if user_dest == user_source:
                    continue
                print "Manage  %d => %d" % (user_source, user_dest)

                # Get scores when user_source want to impersonate user_dest
                try:
                    # Select the right samples
                    scores = self.filter_indicies(np.where(np.logical_and(
                                                    self.get_data()[:,self.GALLERY] == user_source,
                                                    self.get_data()[:,self.PROBE] == user_dest))
                                              )

                    # Compute the mean score
                    data = scores.get_data()
                    data.shape = (-1,4)
                    mean_scores = np.mean(data[:,-1])

                    if self._type == 'score':
                        decision = mean_scores >= thr
                        weight = 1/mean_scores
                    else:
                        decision = mean_scores <= thr
                        weight = mean_scores

                    if decision:
                        DG.add_edge(int(user_source), int(user_dest), weight=weight, mean_scores=mean_scores)

                except Exception, e:
                    print e
                    print 'No comparison between %d and %d' % (user_source, user_dest)

        labels = dict([(edge, "%0.3f" %DG.edge[edge[0]][edge[1]]['mean_scores']) for edge in DG.edges_iter()])
        print labels
        plt.subplot(111)
        pos=networkx.spring_layout(DG, iterations=200, weighted=True)
        networkx.draw_networkx(DG, pos=pos,with_labels=True, node_color=[DG.node[node]['node_color'] for node in DG.nodes()], prog='neato')
        networkx.draw_networkx_edge_labels(DG, pos=pos, edge_labels=labels)
        plt.colorbar()
        HTER = roc.ROCScores(self.get_data(), self._type).get_weighted_error_at_threshold(thr, 0.5)
        plt.title('Connected ratio: %d/%d\nWeakly connected: %d/%d\nNb attracting components: %d\nHTER: %f'% (
                           networkx.number_connected_components(DG.to_undirected()),
                           DG.number_of_nodes(),
                           networkx.number_weakly_connected_components(DG),
                           DG.number_of_nodes(),
                           networkx.number_connected_components(DG.to_undirected()),
                           HTER))
if __name__ == '__main__':

    import sys
    sys.path.append('./../')

    db  = np.loadtxt('data/Scores_A_16573049.txt')
    bm = GraphPlot(db, 'distance')
    bi.Score(db).display_distributions()
    plt.figure()
    plt.hot()
    bm.display_graph(12)
    plt.show()

    import data.banca as banca
    source = banca.Banca()
    #db = source.get_data('g1', 'SURREY_face_nc_man_scale_200')
    #db = source.get_data('g1', 'IDIAP_voice_gmm_auto_scale_25_10_pca')
    db = source.get_data('g1', 'SURREY_face_svm_man_scale_2.00')

    plt.figure()
    bm = GraphPlot(db)
    bi.Score(db).display_distributions()
    plt.figure()
    bm.display_graph(0.01)
    plt.show()

    import data.xmvts
    xmvts = data.xmvts.XMVTS()
    db = xmvts.get_data('PI', 'dev', 1)
    bm = GraphPlot(db)
    plt.figure()
    bi.Score(db).display_distributions()
    plt.figure()
    bm.display_graph(1)


    plt.show()
    pass


# metadata
__author__ = 'Romain Giot'
__copyright__ = 'Copyright 2012'
__credits__ = ['Romain Giot']
__licence__ = 'GPL'
__version__ = '0.1'
__maintainer__ = 'Romain Giot'
__email__ = 'giot.romain@gmail.com'
__status__ = 'Prototype'

