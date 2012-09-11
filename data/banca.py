"""Banca database loader

giot.romain@gmail.com
"""

import zipfile
import numpy as N
import os

CLAIMED = 0
IDENTITY = 1
BEGINNING = 2


_LABEL_TO_POS = { \
'IDIAP_voice_gmm_auto_scale_25_100_pca': 0,
'IDIAP_voice_gmm_auto_scale_25_100' : 1,
'IDIAP_voice_gmm_auto_scale_25_10_pca' : 2,
'IDIAP_voice_gmm_auto_scale_25_10' : 3,
'IDIAP_voice_gmm_auto_scale_25_200_pca' : 4,
'IDIAP_voice_gmm_auto_scale_25_200' : 5,
'IDIAP_voice_gmm_auto_scale_25_25_pca' : 6,
'IDIAP_voice_gmm_auto_scale_25_25' : 7,
'IDIAP_voice_gmm_auto_scale_25_300_pca' : 8,
'IDIAP_voice_gmm_auto_scale_25_300' : 9,
'IDIAP_voice_gmm_auto_scale_25_50_pca' : 10,
'IDIAP_voice_gmm_auto_scale_25_50' : 11,
'IDIAP_voice_gmm_auto_scale_25_75_pca' : 12,
'IDIAP_voice_gmm_auto_scale_25_75' : 13,
'IDIAP_voice_gmm_auto_scale_33_100_pca' : 14,
'IDIAP_voice_gmm_auto_scale_33_100' : 15,
'IDIAP_voice_gmm_auto_scale_33_10_pca' : 16,
'IDIAP_voice_gmm_auto_scale_33_10' : 17,
'IDIAP_voice_gmm_auto_scale_33_200_pca' : 18,
'IDIAP_voice_gmm_auto_scale_33_200' : 19,
'IDIAP_voice_gmm_auto_scale_33_25_pca' : 20,
'IDIAP_voice_gmm_auto_scale_33_25' : 21,
'IDIAP_voice_gmm_auto_scale_33_300_pca' : 22,
'IDIAP_voice_gmm_auto_scale_33_300' : 23,
'IDIAP_voice_gmm_auto_scale_33_50_pca' : 24,
'IDIAP_voice_gmm_auto_scale_33_50' : 25,
'IDIAP_voice_gmm_auto_scale_33_75_pca' : 26,
'IDIAP_voice_gmm_auto_scale_33_75' : 27,
'SURREY_face_nc_man_scale_100' : 28,
'SURREY_face_nc_man_scale_200' : 29,
'SURREY_face_nc_man_scale_25' : 30,
'SURREY_face_nc_man_scale_50' : 31,
'SURREY_face_svm_auto' : 32,
'SURREY_face_svm_man_scale_0.13' : 33,
'SURREY_face_svm_man_scale_0.18' : 34,
'SURREY_face_svm_man_scale_0.25' : 35,
'SURREY_face_svm_man_scale_0.35' : 36,
'SURREY_face_svm_man_scale_0.50' : 37,
'SURREY_face_svm_man_scale_0.71' : 38,
'SURREY_face_svm_man_scale_1.00' : 39,
'SURREY_face_svm_man_scale_1.41' : 40,
'SURREY_face_svm_man_scale_2.00' : 41,
'SURREY_face_svm_man_scale_2.83' : 42,
'SURREY_face_svm_man_scale_4.00' : 43,
'SURREY_face_svm_man' : 44,
'UC3M_voice_gmm_auto_scale_10_100' : 45,
'UC3M_voice_gmm_auto_scale_10_200' : 46,
'UC3M_voice_gmm_auto_scale_10_300' : 47,
'UC3M_voice_gmm_auto_scale_10_32' : 48,
'UC3M_voice_gmm_auto_scale_10_400' : 49,
'UC3M_voice_gmm_auto_scale_10_500' : 50,
'UC3M_voice_gmm_auto_scale_10_64' : 51,
'UC3M_voice_gmm_auto_scale_18_100' : 52,
'UC3M_voice_gmm_auto_scale_18_200' : 53,
'UC3M_voice_gmm_auto_scale_18_300' : 54,
'UC3M_voice_gmm_auto_scale_18_32' : 55,
'UC3M_voice_gmm_auto_scale_18_400' : 56,
'UC3M_voice_gmm_auto_scale_18_500' : 57,
'UC3M_voice_gmm_auto_scale_18_64' :58,
'UC3M_voice_gmm_auto_scale_26_100' : 59,
'UC3M_voice_gmm_auto_scale_26_200' : 60,
'UC3M_voice_gmm_auto_scale_26_300' : 61,
'UC3M_voice_gmm_auto_scale_26_32' : 62,
'UC3M_voice_gmm_auto_scale_26_400' : 63,
'UC3M_voice_gmm_auto_scale_26_500' : 64,
'UC3M_voice_gmm_auto_scale_26_64' : 65,
'UC3M_voice_gmm_auto_scale_34_100' : 66,
'UC3M_voice_gmm_auto_scale_34_200' : 67,
'UC3M_voice_gmm_auto_scale_34_300' : 68,
'UC3M_voice_gmm_auto_scale_34_32' : 69,
'UC3M_voice_gmm_auto_scale_34_400' : 70,
'UC3M_voice_gmm_auto_scale_34_500' : 71,
'UC3M_voice_gmm_auto_scale_34_64' : 72,
'UCL_face_lda_man' : 73,

    }
class Banca(object):
    """Banca database loader.
    You need to download the database at:
    http://personal.ee.surrey.ac.uk/Personal/Norman.Poh/web/banca_multi/main.php?bodyfile=welcome.html
    """

    def __init__(self, path=None):    
        if path is None:
               path =  os.path.join(os.path.dirname(os.path.abspath( __file__ )) , 'snapshot1.zip')

        self._file = zipfile.ZipFile(path)

    def get_data(self, group, data):
        """Returns the data in the correct format.
        
        @param group: group of users
        @param data: name of the modality
        """
        assert group in ['g1', 'g2']
        assert data in _LABEL_TO_POS.keys()

        if group == 'g1':
            fname = 'snapshot1/G_g1.scores'
        else:
            fname = 'snapshot1/G_g2.scores'

        position = _LABEL_TO_POS[data] + BEGINNING

        #Open the archive
        f = self._file.open(fname)
        data = N.loadtxt(f)
        f.close()

        #Keep only one kind of scores
        data = data [:, [CLAIMED, IDENTITY, position] ]

        return self._cleanup_data(data)

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
    banca = Banca()
    print banca.get_data('g1', 'SURREY_face_nc_man_scale_200')
