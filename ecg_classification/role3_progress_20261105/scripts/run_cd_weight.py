import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cd_training_source as base

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('results/CD_weight_experiment'))
    args=ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    d=np.load(args.npz, allow_pickle=False)
    frame=pd.read_csv(Path(__file__).with_name('cohort.csv'))
    ids=d['ecg_id'].astype(int)
    assert len(set(ids.tolist()))==len(ids)==998
    frame=frame.set_index('ecg_id').loc[ids].reset_index()
    assert np.array_equal(frame.patient_id.to_numpy().astype(int),d['patient_id'].astype(int))
    assert np.array_equal(frame.split.to_numpy().astype(str),d['split'].astype(str))
    expected=['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
    assert d['lead_names'].tolist()==expected
    assert float(d['sampling_rate_hz'])==100
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not(set(frame.loc[frame.split==a,'patient_id']) & set(frame.loc[frame.split==b,'patient_id']))
    # Remove test rows before any signal normalization or forward pass.
    keep=frame.split.isin(['train','validation']).to_numpy()
    values=d['reconstructed_12lead'][keep].copy()
    values[:,:,0]=d['lead_i'][keep,:,0]
    values=values.transpose(0,2,1).copy()
    frame=frame.loc[keep].reset_index(drop=True)
    assert np.isfinite(values).all() and values.shape[1:]==(12,1000)
    eligible=frame[base.LABELS].sum(axis=1)>0
    assert int(((frame.split=='train')&eligible).sum())==796
    assert int(((frame.split=='validation')&eligible).sum())==96
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plan={'task':'multilabel','input':'C','seeds':[42,43,44], 'epochs':40,
          'variants':{'baseline':1.0,'cd_sqrt_weight':0.5},
          'change':'CD train positive weight only: neg/pos -> sqrt(neg/pos)',
          'checkpoint_selection':'both variants use original neg/pos weighted validation BCE',
          'threshold_selection':base.PROTOCOL['threshold_selection'],
          'evaluation':'validation only; exploratory; no test forward pass',
          'npz_sha256':hashlib.sha256(args.npz.read_bytes()).hexdigest()}
    base.write_json(args.output/'experiment_plan.json',plan)
    rows=[]
    for variant,power in plan['variants'].items():
        base.CD_POWER=power
        for seed in plan['seeds']:
            out=args.output/variant
            base.train_one(out,'multilabel','C',seed,frame,values,device)
            folder=out/'models/multilabel/C'/f'seed_{seed}'
            probs=pd.read_csv(folder/'validation_probabilities.csv')
            thresholds={x['label']:x['threshold'] for x in json.loads((folder/'thresholds.json').read_text())}
            for lab,g in probs.groupby('label'):
                r=base.binary_metrics(g.target.to_numpy(),g.probability.to_numpy(),g.probability.to_numpy()>=thresholds[lab])
                rows.append(dict(variant=variant,seed=seed,label=lab,threshold=thresholds[lab],**r))
            pd.DataFrame(rows).to_csv(args.output/'validation_by_seed.csv',index=False)
    result=pd.DataFrame(rows)
    result.groupby(['variant','label'])[['f1','precision','recall','auroc','auprc','fp','fn']].agg(['mean','std']).to_csv(args.output/'validation_summary.csv')
    print('COMPLETE:',args.output.resolve())

if __name__=='__main__':main()
