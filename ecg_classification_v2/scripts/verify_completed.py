"""Independent checks for newly trained runs; requires real completed outputs."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--policy',choices=['v1','v2'],default='v2');args=p.parse_args()
 run=R/'runs'/args.policy;assert json.loads((run/'completion.json').read_text())['status']=='complete'
 lock=json.loads((run/'selection_frozen_before_test.json').read_text());assert len(lock['models'])==18
 m=pd.read_csv(run/'metrics_by_seed.csv');count=0
 for s in lock['models']:
  folder=run/s['path']
  for name,key in [('best_model.pth','checkpoint_sha256'),('thresholds.json','thresholds_sha256')]:assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==s[key]
  h=pd.read_csv(folder/'history.csv');assert len(h)==40
  assert int(h.loc[h.validation_loss.idxmin(),'epoch'])==s['best_epoch']
  with np.load(folder/'predictions.npz',allow_pickle=False) as z:
   config=json.loads((folder/'config.json').read_text());assert config['n_train']==796 and config['n_validation']==96
   assert np.array_equal(z['predicted'],(z['probability']>=z['thresholds']).astype(int))
   assert np.isnan(z['target'][~z['eligible']]).all()
   assert {s0:int(((z['split']==s0)&z['eligible']).sum()) for s0 in ['train','validation','test']}=={'train':796,'validation':96,'test':98}
   for split in ['train','validation','test']:
    mask=(z['split']==split)&z['eligible']
    for j,label in enumerate(z['labels']):
     y=z['target'][mask,j];p=z['predicted'][mask,j];tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum())
     vals={'precision':tp/(tp+fp) if tp+fp else np.nan,'recall':tp/(tp+fn) if tp+fn else np.nan,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan}
     row=m[(m.task==s['task'])&(m.input==s['input'])&(m.seed==s['seed'])&(m.split==split)&(m.label==label)].iloc[0]
     for k,v in vals.items():assert np.isclose(row[k],v,atol=1e-10,equal_nan=True)
     count+=1
 # Same shuffled training record ordering for A/B/C with each seed/task.
 for task in ['multilabel','binary']:
  for seed in [42,43,44]:
   ds=[pd.read_csv(run/'models'/task/inp/f'seed_{seed}/train_order_digests.csv') for inp in ['A','B','C']]
   assert ds[0].equals(ds[1]) and ds[1].equals(ds[2])
 result={'status':'passed','trained_models':18,'epochs_each':40,'verified_per_label_metric_rows':count,'checkpoint_and_threshold_hashes_match_lock':True,'first_minimum_validation_checkpoint':True,'excluded_records_targets_NA':True,'training_order_shared_by_inputs':True}
 (R/'reports'/('new_run_verification.json' if args.policy=='v2' else 'v1_cpu_control_verification.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
