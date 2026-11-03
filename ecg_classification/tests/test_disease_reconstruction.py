"""Meaningful edge-case and audit tests; uses synthetic data, never trains."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_disease_reconstruction as analysis


class MetricsTest(unittest.TestCase):
    def test_known_errors_and_correlations(self):
        a = np.array([0., 1., 2., 3.])[None, :, None]
        b = a + 2
        m = analysis.metric_arrays(a, b)
        self.assertAlmostEqual(m["mae"].item(), 2)
        self.assertAlmostEqual(m["rmse"].item(), 2)
        self.assertAlmostEqual(m["pearson_r"].item(), 1)
        reverse = analysis.metric_arrays(a, -a)
        self.assertAlmostEqual(reverse["pearson_r"].item(), -1)

    def test_constants_near_constants_and_nonfinite(self):
        a = np.tile(np.arange(4.)[None, :, None], (6, 1, 1))
        b = a.copy()
        a[0] = .1
        b[1] = .1
        a[2] = b[2] = .1
        b[3, 0, 0] = np.nan
        a[4, :, 0] = 1e9 + np.arange(4) * 1e-7
        a[5, 0, 0] = np.inf
        m = analysis.metric_arrays(a, b)
        self.assertEqual(m["correlation_status"][:, 0].tolist(), ["constant_actual", "constant_reconstructed", "constant_both", "nonfinite_reconstructed", "near_constant", "nonfinite_actual"])
        self.assertTrue(np.isnan(m["pearson_r"]).all())
        self.assertTrue(np.isfinite(m["mae"][:3]).all())
        self.assertTrue(np.isnan(m["mae"][3]).all())

    def test_excludes_input_i_with_permuted_leads(self):
        leads = ["V6", "I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5"]
        a = np.broadcast_to(np.arange(4.)[None, :, None], (1, 4, 12)).copy()
        b = a + 1
        b[:, :, 1] += 10000
        joined = pd.DataFrame([{"ecg_id": 10, "split": "test", "npz_index": 0, "analysis_classes": "NORM", "analysis_groups": "normal_class_only"}])
        m, r = analysis.make_metrics(joined, a, b, leads)
        self.assertEqual(len(m), 11)
        self.assertNotIn("I", m.lead.tolist())
        self.assertEqual(m.lead.tolist()[0], "V6")
        self.assertAlmostEqual(r.mae.iloc[0], 1)
        self.assertAlmostEqual(r.rmse.iloc[0], 1)

    def test_pooled_rmse_differs_from_mean_rmse_and_empty_groups_remain(self):
        leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        a = np.broadcast_to(np.arange(4.)[None, :, None], (2, 4, 12)).copy()
        b = a.copy()
        b[1] += 4
        joined = pd.DataFrame({"ecg_id": [1, 2], "split": ["test", "test"], "npz_index": [0, 1], "analysis_classes": ["NORM", "NORM"], "analysis_groups": ["normal_class_only", "normal_class_only"]})
        m, r = analysis.make_metrics(joined, a, b, leads)
        counts, overlap, summary = analysis.summarize(m, r, leads[1:])
        overall = summary[(summary.split == "test") & (summary.group == "normal_class_only") & (summary.lead == "OVERALL")].iloc[0]
        self.assertAlmostEqual(overall.rmse_mean, 2)
        self.assertAlmostEqual(overall.pooled_rmse, np.sqrt(8))
        self.assertEqual(len(counts), 14)
        self.assertEqual(len(summary), 14 * 12)
        self.assertTrue(summary[summary.n_records == 0].mae_mean.isna().all())


class LabelsTest(unittest.TestCase):
    def test_multilabel_mixed_and_unclassified(self):
        self.assertEqual(set(analysis.group_memberships(analysis.parse_classes("MI;STTC"))), {"MI", "STTC"})
        self.assertEqual(analysis.group_memberships({"NORM", "CD"}), ["normal_disease_mixed"])
        self.assertEqual(analysis.group_memberships(set()), ["unclassified"])
        self.assertEqual(analysis.group_memberships({"NORM"}), ["normal_class_only"])
        with self.assertRaises(ValueError):
            analysis.parse_classes("AMI")

    def test_inclusion_flag_does_not_assign_or_drop(self):
        frame = pd.DataFrame({"diagnostic_classes": ["NORM", "MI", np.nan], "include_in_analysis": [False, True, True]})
        assigned, _ = analysis.assign_classes(frame)
        self.assertEqual(assigned.analysis_groups.tolist(), ["normal_class_only", "MI", "unclassified"])
        frame["NORM"], frame["MI"], frame["STTC"], frame["CD"], frame["HYP"] = [1, 0, 0], [0, 0, 0], 0, 0, 0
        with self.assertRaisesRegex(ValueError, "disagrees"):
            analysis.assign_classes(frame)


class InputAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.actual = np.arange(72, dtype=np.float32).reshape(3, 2, 12)
        np.savez(self.root / "reference.npz", X=self.actual[:, :, :1], Y=self.actual[:, :, 1:], ecg_id=[1, 2, 3], patient_id=[11, 12, 13], strat_fold=[1, 9, 10], split=["train", "validation", "test"], lead_names=self.leads, sampling_rate_hz=100)
        config = {"dataset_sha256": analysis.sha256(self.root / "reference.npz"), "split_ecg_ids": {"train": [1], "validation": [2], "test": [3]}}
        (self.root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        self.frame = pd.DataFrame({"ecg_id": [3, 1, 2], "split": ["test", "train", "validation"], "diagnostic_classes": ["MI;CD", "NORM", "NORM"]})
        self.arrays = dict(actual_12lead=self.actual, reconstructed_12lead=self.actual + 1, ecg_id=np.array([1, 2, 3]), split=np.array(["train", "validation", "test"]), lead_names=np.array(self.leads), sampling_rate_hz=100)
        self.args = analysis.parse_args(["--labels", str(self.root / "labels.csv"), "--reconstructions", str(self.root / "recon.npz"), "--reference-dataset", str(self.root / "reference.npz"), "--split-config", str(self.root / "config.json")])

    def tearDown(self):
        self.tmp.cleanup()

    def load(self):
        self.frame.to_csv(self.args.labels, index=False)
        np.savez(self.args.reconstructions, **self.arrays)
        return analysis.load_inputs(self.args, self.root, {})

    def test_join_uses_id_not_row_order(self):
        joined, *_ = self.load()
        self.assertEqual(joined.ecg_id.tolist(), [1, 2, 3])
        self.assertEqual(joined.analysis_classes.tolist(), ["NORM", "NORM", "CD|MI"])

    def test_duplicate_and_missing_ids_rejected_and_audited(self):
        self.frame.loc[0, "ecg_id"] = 1
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load()
        self.assertTrue((self.root / "duplicate_ids.csv").is_file())
        self.frame = self.frame.iloc[1:].copy()
        with self.assertRaisesRegex(ValueError, "Missing/extra"):
            self.load()

    def test_split_mismatch_rejected(self):
        self.frame.loc[0, "split"] = "validation"
        with self.assertRaisesRegex(ValueError, "split mismatch"):
            self.load()
        self.assertEqual(len(pd.read_csv(self.root / "split_mismatches.csv")), 1)

    def test_changed_actual_and_reference_hash_rejected(self):
        self.arrays["actual_12lead"] = self.actual + .01
        with self.assertRaisesRegex(ValueError, "Actual waveforms differ"):
            self.load()
        self.args.reference_dataset.write_bytes(b"changed reference")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.load()


if __name__ == "__main__":
    unittest.main()
