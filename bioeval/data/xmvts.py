#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AUTHOR Romain Giot <giot.romain@gmail.com>

TODO give a string instead of a method number
"""
import zipfile
import numpy as N
import os

import cPickle as pickle

CLAIMED = 0
IDENTITY = 1
BEGINNING = 2

class XMVTS(object):
    """XMVTS database loader.
    You need to download the database at:
    http://info.ee.surrey.ac.uk/Personal/Norman.Poh/web/fusion/main.php?bodyfile=entry_page.html
    """

    def __init__(self, path=None):
        if path is None:
               path =  os.path.join(os.path.dirname(os.path.abspath( __file__ )), 'data.zip')

        self._file = zipfile.ZipFile(path)


    def get_data(self, protocol,setname, experiment):
        """Read one data from xmvts"""
        assert protocol in ['PI', 'PII']
        assert setname in ['dev', 'eva']


        fname = 'Data/multi_modal/%s/%s.label' % (protocol, setname)
        pickle_fname = '/tmp/%s%s.pickle' % (protocol, setname)

        try:
            fp = open(pickle_fname, 'r')
            result = pickle.load(fp)
            fp.close()
        except:

            #Open the archive
            f = self._file.open(fname)
            data = N.loadtxt(f)
            f.close()

            #Keep only one kind of scores
            position = BEGINNING + experiment
            data = data [:, [CLAIMED, IDENTITY, position] ]

            result = self._cleanup_data(data)

            fp = open(pickle_fname, 'w')
            pickle.dump(result, fp)
            fp.close()

        return result


    def _cleanup_data(self, data):
        """Cleanup the data :
          * Add the presentation column
          * Order everithing
        """
        results = None
        
        for i in N.unique(data[:,CLAIMED]):
            for k in N.unique(data[:,IDENTITY]):
                indicies = N.where(N.logical_and(\
                    data[:,IDENTITY] == k,
                    data[:,CLAIMED] == i
                ))

                #Assemble the tries
                presentations = data[indicies,BEGINNING].flatten()
                if len(presentations) == 0: continue

                presentation_order = range(1, len(presentations) + 1)
                tmp = N.empty( (len(presentations),4))
                tmp[:,0] = i
                tmp[:,1] = k
                tmp[:,2] = presentation_order
                tmp[:,3] = presentations

                if results is None:
                    results = tmp
                else:
                    results = N.vstack( (tmp, results) )


        return results
if __name__ == "__main__":
    xmvts = XMVTS()
    print xmvts.get_data('PI', 'dev', 1)
