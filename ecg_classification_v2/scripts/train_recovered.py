"""Frozen classification functions, recovered input audit, v1/v2 targets only."""
from pathlib import Path
import sys,json,hashlib,argparse,shutil
import numpy as np,pandas as pd,torch
R=Path(__file__).resolve().parents[1];O=R/'preserved/role3';W=R/'working'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--policy',choices=['v1','v2'],default='v2');p.add_argument('--output',type=Path);a=p.parse_args()
 out=(a.output or R/'runs'/a.policy).resolve()
 for protected in [R/'preserved',R/'recovery',R/'working',R/'scripts',R/'reports']:
  if out==protected or protected in out.parents or out in protected.parents:raise ValueError('Output overlaps a protected directory')
 out.mkdir(parents=True,exist_ok=False)
 for f in O.glob('*.py'):shutil.copy2(f,W/f.name)
 for folder in ['checkpoints','results/dilated_1000']:
  if not (W/folder).exists():shutil.copytree(O/folder,W/folder)
 (W/'data/processed').mkdir(parents=True,exist_ok=True)
 ref=R/'recovery/processed/ptbxl_100hz_1000records.npz'
 if not (W/'data/processed'/ref.name).exists():shutil.copy2(ref,W/'data/processed'/ref.name)
 sys.path.insert(0,str(W));import train_ecg_classification as t
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 original_config=O/'results/dilated_1000/run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636/experiment_config.json'
 cfg=json.loads(original_config.read_text());cfg['historical_dataset_sha256']=cfg['dataset_sha256'];cfg['dataset_sha256']=sha(ref)
 cfg['recovery_note']='Derived audit reference only; historical config preserved separately; new serialization hash not claimed identical'
 (out/'recovered_split_config.json').write_text(json.dumps(cfg,indent=2))
 parsed=t.audit_tools.parse_args
 def parse_override(argv=None):
  return parsed(['--labels',str(R/'preserved/labels_v1/ecg_labels_analysis_998.csv'),'--reconstructions',str(R/'recovery/ecg_for_classification.npz'),'--reference-dataset',str(ref),'--split-config',str(out/'recovered_split_config.json')])
 t.audit_tools.parse_args=parse_override
 frame,values,audit=t.prepare(out)
 t.provenance(out,frame,values)
 v=pd.read_csv(R/'preserved/labels_v2/tables/ecg_labels_v2_classifier_998.csv').set_index('ecg_id')
 assert v.index.is_unique and set(v.index)==set(frame.ecg_id)
 v=v.loc[frame.ecg_id]
 for key in ['patient_id','strat_fold','split']:assert np.array_equal(v[key].to_numpy(),frame[key].to_numpy()),key
 for k in t.LABELS:assert np.array_equal(v['source_'+k].to_numpy(),frame[k].to_numpy()),k
 oldtarget=frame[t.LABELS].to_numpy().copy();eligible=oldtarget.sum(1)>0
 assert np.array_equal(eligible,v.supervised_eligible_v2.to_numpy()==1)
 target=v[['target_v2_'+k for k in t.LABELS]].to_numpy()
 assert np.isnan(target[~eligible]).all()
 expect=oldtarget.astype(float);expect[expect[:,1:].sum(1)>0,0]=0;expect[~eligible]=np.nan
 assert np.array_equal(target,expect,equal_nan=True)
 if a.policy=='v2':
  for k in t.LABELS:frame['source_'+k]=frame[k];frame[k]=v[k].to_numpy(dtype=int)
  t.PROTOCOL=dict(t.PROTOCOL,multilabel_mixed='v2 disease positive forces NORM=0; disease labels unchanged')
 frame['label_policy']=a.policy
 t.csv(out/'cohort.csv',frame)
 support=[]
 for split in ['train','validation','test']:
  m=(frame.split==split)&eligible
  for label in t.LABELS:support.append({'split':split,'label':label,'n_eligible':int(m.sum()),'positive':int(frame.loc[m,label].sum())})
 t.csv(out/'label_support.csv',support)
 config={'started_at_kst':t.stamp(),'protocol':t.PROTOCOL,'policy':a.policy,'device':'cpu','versions':{'torch':str(torch.__version__),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':t.sklearn.__version__},'code_sha256':sha(W/'train_ecg_classification.py'),'adapter_sha256':sha(Path(__file__)),'recovered_data_sha256':sha(R/'recovery/ecg_for_classification.npz'),'reference_sha256':sha(ref),'v2_label_sha256':sha(R/'preserved/labels_v2/tables/ecg_labels_v2_classifier_998.csv'),'historical_config_unchanged':True}
 t.write_json(out/'experiment_config.json',config)
 selections=[]
 for task in ['multilabel','binary']:
  for seed in t.PROTOCOL['seeds']:
   for inp in t.INPUTS:selections.append(t.train_one(out,task,inp,seed,frame,values[inp],torch.device('cpu')))
 t.write_json(out/'selection_frozen_before_test.json',{'frozen_at_kst':t.stamp(),'config_sha256':sha(out/'experiment_config.json'),'models':selections})
 t.csv(out/'selected_models.csv',selections)
 metrics=t.evaluate(out,frame,values,selections,torch.device('cpu'));t.summarize(out,metrics)
 t.write_json(out/'completion.json',{'status':'complete','models_trained':18,'finished_at_kst':t.stamp(),'all_selections_locked_before_test':True})
if __name__=='__main__':main()
