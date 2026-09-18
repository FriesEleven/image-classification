"""Tests of ID leakage, selection objectives, and corrected MAC cost."""
import unittest
import numpy as np
from scripts.analysis.minimal_core_evidence import split_by_id, metrics, select, bootstrap_pair


class MinimalCoreTests(unittest.TestCase):
    def test_overlap_alignment_with_different_target_pool(self):
        cohorts={}
        for dataset,names,classes in [('cifar10',['cifar10_source','cifar10_target'],10),
            ('cifar100',['cifar100_source','cifar100_target','cifar100_confirmation'],100)]:
            ids=np.arange(5000);labels=np.repeat(np.arange(classes),5000//classes)
            for name in names:
                altered=ids.copy()
                if name.endswith('confirmation'):altered[::2]+=10000
                order=np.arange(4999,-1,-1)
                cohorts[name]=[dict(sample_ids=altered[order],labels=labels[order])]
        result=split_by_id(cohorts)
        for part in result.values():
            self.assertEqual(len(part['fit_ids']),2500);self.assertEqual(len(part['audit_ids']),2500)
            self.assertFalse(set(part['fit_ids'])&set(part['audit_ids']))
        self.assertFalse(set(result['cifar100_source']['fit_ids'])&set(result['cifar100_confirmation']['audit_ids']))
        self.assertEqual(result,split_by_id(cohorts))

    def test_bad_id_labels_fail(self):
        cohorts={'cifar10_source':[dict(sample_ids=np.arange(5000),labels=np.repeat(np.arange(10),500))]*2}
        cohorts['cifar10_source'][1]=dict(sample_ids=np.arange(5000),labels=np.zeros(5000,dtype=int))
        with self.assertRaises(ValueError):split_by_id(cohorts)

    def test_fallback_overhead_and_risk_constraint(self):
        # Full prediction correct; exit prediction wrong on one of two images.
        row=dict(labels=np.array([0,1]),final_logits=np.array([[3.,0.],[0.,3.]]),
            exit8_logits=np.array([[0.,3.],[0.,3.]]),exit_cost=.4,head_cost=.01)
        self.assertAlmostEqual(metrics(row,np.array([False,False]))['cost_saving_fraction'],-.01)
        chosen=select([row],np.array([0.,1.01]),'S',0.)
        self.assertEqual(chosen['threshold'],1.01)

    def test_removing_class_constraint_keeps_failure_visible(self):
        row=dict(labels=np.array([0,0,1,1]),final_logits=np.array([[3.,0.],[3.,0.],[3.,0.],[0.,3.]]),
            exit8_logits=np.array([[0.,3.],[3.,0.],[0.,3.],[0.,3.]]),exit_cost=.4,head_cost=.01)
        grid=np.array([0.,1.01])
        self.assertEqual(select([row],grid,'S',0.)['threshold'],1.01)
        self.assertEqual(select([row],grid,'C',0.)['threshold'],0.)
        result=metrics(row,np.ones(4,dtype=bool))
        self.assertEqual(result['accuracy_drop'],0.)
        self.assertEqual(result['worst_class_accuracy_drop'],.5)

    def test_bootstrap_preserves_fixed_model_image_pairing(self):
        result=bootstrap_pair(np.array([0,0,1,1]),[np.ones(4)*.5]*3)
        self.assertEqual(result['paired_accuracy_difference'],.5)
        self.assertEqual(result['paired_ci95_low'],.5)
        self.assertEqual(result['paired_ci95_high'],.5)


if __name__=='__main__':unittest.main()
