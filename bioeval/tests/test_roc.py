import sys
sys.path.append('./../')

import errors.roc as roc
import data.banca as banca

import unittest
import matplotlib.pylab as plt

class TestROC(unittest.TestCase):
    """Test the running of the ROC curve utility."""

    def setUp(self):
        """Read the database."""
        source = banca.Banca()
        #db = source.get_data('g1', 'IDIAP_voice_gmm_auto_scale_25_10_pca')
        db = source.get_data('g1', 'SURREY_face_nc_man_scale_200')
        self.roc = roc.ROCScores(db)
        self.curve = self.roc.get_roc()
        
    def test_intra_inter_splitting(self):
        """Test if we read correctly the intra nad inter values."""


        intra = self.roc.get_genuine_presentations()
        inter = self.roc.get_impostor_presentations()


        intra_size = intra.get_number_of_presentations()
        inter_size = inter.get_number_of_presentations()
        total_size = self.roc.get_number_of_presentations()

        self.assertEquals( total_size, intra_size + inter_size)

    def test_bootstrap(self):
        """Test if bootsrtapping works."""
        bootstrap = self.roc.bootstrap()

        self.assertNotEqual( bootstrap, None)

        intra = bootstrap.get_genuine_presentations()
        inter = bootstrap.get_impostor_presentations()


        intra_size = intra.get_number_of_presentations()
        inter_size = inter.get_number_of_presentations()
        total_size = bootstrap.get_number_of_presentations()

        self.assertEquals( total_size, intra_size + inter_size)


    def test_roc_computing(self):
        """Test if launch ROC with failing."""
        self.assertNotEqual(self.curve, None)

    def test_auc(self):
        """TODO give the answer"""
        auc = self.curve.get_auc()
        self.assertTrue(auc >= 0)
        #self.assertAlmostEqual(auc, 0) #TODO set the value

    def test_eer(self):
        """Test if EER computation is correct and verify
        if polar and cartesian coordinate returns the same thing
        """
        eer1 = self.curve.get_eer()
        self.assertTrue(eer1 >=0)
        self.assertAlmostEqual(eer1, 0.0595, 2) #TODO verifier avec plus de precision

        eer2 = self.curve.get_polar_representation().get_eer()
        self.assertTrue(eer2 >=0)
        self.assertAlmostEqual(eer1, eer2, 2)

    
    def test_plot(self):
        plt.figure()
        self.curve.plot()
        #plt.show()

if __name__ == '__main__':
    unittest.main()
