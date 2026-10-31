import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_ecg_classification import (Classifier, LABELS, binary_metrics,
                                      choose_thresholds, normalization, targets)


class ClassificationTests(unittest.TestCase):
    def test_confusion_and_rank_metrics(self):
        m = binary_metrics([0, 0, 1, 1], [.1, .7, .4, .9], [0, 1, 0, 1])
        self.assertEqual([m[k] for k in ['tn', 'fp', 'fn', 'tp']], [1, 1, 1, 1])
        for k in ['precision', 'recall', 'specificity', 'f1']:
            self.assertAlmostEqual(m[k], .5)
        self.assertAlmostEqual(m['auroc'], .75)
        self.assertAlmostEqual(m['auprc'], 5/6)

    def test_undefined_metrics_have_reasons(self):
        m = binary_metrics([0, 0], [.2, .3], [0, 0])
        for key in ['auroc', 'auprc', 'precision', 'recall', 'f1']:
            self.assertTrue(np.isnan(m[key]))
            self.assertTrue(m[key + '_na_reason'])
        self.assertEqual(m['specificity'], 1)

    def test_mixed_retained_and_unclassified_excluded(self):
        frame = pd.DataFrame([[1,0,0,0,0],[1,0,0,1,0],[0,0,0,0,0],[0,1,0,0,0]], columns=LABELS)
        y, eligible = targets(frame, 'multilabel')
        np.testing.assert_array_equal(eligible, [True, True, False, True])
        np.testing.assert_array_equal(y[1], [1,0,0,1,0])
        binary, mask = targets(frame, 'binary')
        np.testing.assert_array_equal(binary[:,0], [0,1,0,1])
        np.testing.assert_array_equal(mask, eligible)

    def test_normalization_uses_train_mask_only(self):
        x = np.array([[[1.,3.]], [[1000.,2000.]]])
        mean, sd, constant = normalization(x, np.array([True,False]))
        np.testing.assert_allclose(mean, [2])
        np.testing.assert_allclose(sd, [1])
        self.assertFalse(constant.item())

    def test_threshold_tie_and_degenerate_validation(self):
        y=np.array([[0],[0],[1],[1]])
        t, details=choose_thresholds(y,np.array([[.1],[.2],[.8],[.9]]))
        self.assertAlmostEqual(t.item(), .5)
        self.assertEqual(details[0]['validation_f1'], 1)
        t, details=choose_thresholds(np.zeros((3,1)),np.array([[.1],[.2],[.3]]))
        self.assertEqual(t.item(), .5)
        self.assertIn('fallback', details[0]['reason'])

    def test_deeper_initialization_shared_across_inputs(self):
        a,b,c=Classifier(1,5,42),Classifier(12,5,42),Classifier(12,5,42)
        for key,value in a.body.state_dict().items():
            self.assertTrue(torch.equal(value,b.body.state_dict()[key]))
        for key,value in a.head.state_dict().items():
            self.assertTrue(torch.equal(value,b.head.state_dict()[key]))
        for key,value in b.state_dict().items():
            self.assertTrue(torch.equal(value,c.state_dict()[key]))

    def test_output_shapes(self):
        for channels,outputs in [(1,5),(12,5),(1,1),(12,1)]:
            model=Classifier(channels,outputs,42)
            self.assertEqual(tuple(model(torch.zeros(2,channels,1000)).shape),(2,outputs))


if __name__=='__main__':
    unittest.main()
