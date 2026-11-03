"""Verify saved classification artifacts without retraining or selecting models."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
import torch

from train_ecg_classification import (DEFAULT_OUT, LABELS, PROJECT, PROTOCOL,
                                      choose_thresholds, normalization, targets)
from analyze_disease_reconstruction import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.experiment
    config = json.loads((out/'experiment_config.json').read_text(encoding='utf-8'))
    completion = json.loads((out/'completion.json').read_text(encoding='utf-8'))
    lock = json.loads((out/'selection_frozen_before_test.json').read_text(encoding='utf-8'))
    assert config['code_sha256'] == sha256(PROJECT/'train_ecg_classification.py')
    assert completion['models_trained'] == len(lock['models']) == 18
    assert completion['selection_lock_sha256'] == sha256(out/'selection_frozen_before_test.json')
    assert lock['config_sha256'] == sha256(out/'experiment_config.json')
    frame = pd.read_csv(out/'cohort.csv')
    metrics = pd.read_csv(out/'metrics_by_seed.csv')
    long = pd.read_csv(out/'predictions_long.csv')
    assert len(long) == 998 * 9 * 6 and len(metrics) == 216
    assert frame.groupby('split').size().to_dict() == {'test':100,'train':798,'validation':100}
    with np.load(PROJECT/'data/received/ecg_for_classification.npz',allow_pickle=False) as z:
        actual, reconstructed = z['actual_12lead'], z['reconstructed_12lead']
        assert np.array_equal(frame.ecg_id.to_numpy(),z['ecg_id'])
        assert np.array_equal(actual[:,:,0],reconstructed[:,:,0])
        values={'A': actual[:,:,:1].transpose(0,2,1),'B':actual.transpose(0,2,1),'C':reconstructed.transpose(0,2,1)}
    orders = {}
    metric_groups = 0
    for selection in lock['models']:
        folder=out/selection['path']
        assert sha256(folder/'best_model.pth')==selection['checkpoint_sha256']
        assert sha256(folder/'thresholds.json')==selection['thresholds_sha256']
        history=pd.read_csv(folder/'history.csv')
        assert history.epoch.tolist()==list(range(1,41))
        assert int(history.loc[history.validation_loss.idxmin(),'epoch'])==selection['best_epoch']
        np.testing.assert_allclose(history.validation_loss.min(),selection['best_validation_loss'],rtol=1e-10)
        cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        y,eligible=targets(frame,selection['task'])
        train=(frame.split=='train').to_numpy() & eligible
        val=(frame.split=='validation').to_numpy() & eligible
        assert int(train.sum())==796 and int(val.sum())==96
        assert cfg['train_ids']==frame.ecg_id[train].tolist()
        assert cfg['validation_ids']==frame.ecg_id[val].tolist()
        assert set(cfg['train_ids']).isdisjoint(frame.ecg_id[frame.split=='test'])
        mean,sd,_=normalization(values[selection['input']],train)
        np.testing.assert_allclose(cfg['normalization_mean'],mean,rtol=1e-12)
        np.testing.assert_allclose(cfg['normalization_sd'],sd,rtol=1e-12)
        np.testing.assert_allclose(cfg['pos_weight'],(len(y[train])-y[train].sum(axis=0))/y[train].sum(axis=0),rtol=1e-7)
        orders[(selection['task'],selection['input'],selection['seed'])]=pd.read_csv(folder/'train_order_digests.csv').train_ecg_id_order_sha256.tolist()
        with np.load(folder/'predictions.npz',allow_pickle=False) as z:
            p, predicted, stored_y = z['probability'],z['predicted'],z['target']
            np.testing.assert_array_equal(z['eligible'],eligible)
            np.testing.assert_array_equal(z['ecg_id'],frame.ecg_id)
            assert np.isnan(stored_y[~eligible]).all()
            np.testing.assert_array_equal(stored_y[eligible],y[eligible])
            np.testing.assert_array_equal(predicted,p>=np.asarray(selection['thresholds']))
            expected,_=choose_thresholds(y[val],p[val])
            np.testing.assert_array_equal(expected,selection['thresholds'])
        for split in ['train','validation','test']:
            mask=(frame.split==split).to_numpy() & eligible
            for k,label in enumerate(cfg['labels']):
                yy,pp,pred=y[mask,k],p[mask,k],predicted[mask,k]
                r=metrics[(metrics.task==selection['task'])&(metrics.input==selection['input'])&(metrics.seed==selection['seed'])&(metrics.split==split)&(metrics.label==label)].iloc[0]
                independent={'auroc':roc_auc_score(yy,pp),'auprc':average_precision_score(yy,pp),
                             'precision':precision_score(yy,pred,zero_division=np.nan),
                             'recall':recall_score(yy,pred,zero_division=np.nan),'f1':f1_score(yy,pred,zero_division=np.nan)}
                tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel()
                for key,v in independent.items():
                    np.testing.assert_allclose(r[key],v,rtol=1e-10,atol=1e-10,equal_nan=True)
                np.testing.assert_array_equal([r.tn,r.fp,r.fn,r.tp],[tn,fp,fn,tp])
                metric_groups+=1
    for task in ['multilabel','binary']:
        for seed in [42,43,44]:
            assert orders[(task,'A',seed)]==orders[(task,'B',seed)]==orders[(task,'C',seed)]
    # Recompute macro/micro from stored per-class rows and raw predictions, not training code.
    for task in ['multilabel','binary']:
        for inp in ['A','B','C']:
            for seed in [42,43,44]:
                q=long[(long.task==task)&(long.input==inp)&(long.seed==seed)]
                assert len(q)==998*(5 if task=='multilabel' else 1)
                assert q[~q.eligible].target.isna().all()
                if task=='multilabel':
                    for split in ['train','validation','test']:
                        rows=metrics[(metrics.task==task)&(metrics.input==inp)&(metrics.seed==seed)&(metrics.split==split)]
                        macro=rows[rows.label=='macro'].iloc[0]
                        for key in ['auroc','auprc','precision','recall','f1']:
                            np.testing.assert_allclose(macro[key],rows[rows.label.isin(LABELS)][key].mean(),rtol=1e-10)
                        micro=rows[rows.label=='micro'].iloc[0]
                        allpairs=q[(q.split==split)&q.eligible]
                        for key,v in [('auroc',roc_auc_score(allpairs.target,allpairs.probability)),('auprc',average_precision_score(allpairs.target,allpairs.probability)),('f1',f1_score(allpairs.target,allpairs.predicted_label))]:
                            np.testing.assert_allclose(micro[key],v,rtol=1e-10)
    # Source/hash evidence for the numerically matching reconstruction checkpoint.
    provenance=json.loads((out/'reconstruction_provenance.json').read_text(encoding='utf-8'))
    evidence=[]
    for match in provenance['numerically_consistent_checkpoints']:
        checkpoint=torch.load(PROJECT/match['checkpoint'],map_location='cpu',weights_only=True)
        metadata=checkpoint['experiment']
        for name,digest in metadata.get('source_sha256',{}).items():
            path=PROJECT/name
            evidence.append({'source':name,'exists':path.is_file(),'matches_saved_hash':path.is_file() and sha256(path)==digest})
        name='model_dilated.py' if match['model_class']=='model_dilated.ECGDilatedModel' else 'model.py'
        evidence.append({'source':name,'exists':True,'matches_saved_hash':sha256(PROJECT/name)==metadata['model_structure']['model_source_sha256']})
    (out/'reconstruction_source_hash_verification.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    result={'status':'passed','trained_models':18,'epochs_per_model':40,'prediction_rows':len(long),'metric_rows':len(metrics),
            'independently_verified_per_label_split_seed_groups':metric_groups,'macro_micro_recalculation':'passed',
            'all_checkpoints_are_first_minimum_validation_BCE':True,'all_thresholds_recomputed_from_validation_only':True,
            'all_normalizers_and_weights_recomputed_from_train_only':True,'same_seed_minibatch_order_shared_by_ABC':True,
            'original_split_preserved':True,'unclassified_targets_saved_as_NA':True,
            'matching_reconstruction_source_hashes':evidence}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
