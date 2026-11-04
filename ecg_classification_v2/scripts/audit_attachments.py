"""Check supplied metadata and stored predictions without inventing new training."""
from pathlib import Path
import json,hashlib,zipfile
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'preserved/role3';out=R/'reports'
labels=['NORM','MI','STTC','CD','HYP']
def main():
 old=pd.read_csv(O/'results/cohort.csv').set_index('ecg_id')
 supplied=pd.read_csv(R/'preserved/labels_v1/ecg_labels_analysis_998.csv').set_index('ecg_id').loc[old.index]
 new=pd.read_csv(R/'preserved/labels_v2/tables/ecg_labels_v2_classifier_998.csv').set_index('ecg_id').loc[old.index]
 for k in ['patient_id','strat_fold','split']+labels:assert old[k].equals(supplied[k]),k
 assert old.index.is_unique and new.index.is_unique
 for k in ['patient_id','strat_fold','split']:assert old[k].equals(new[k]),k
 for k in labels:assert np.array_equal(old[k],new['source_'+k])
 eligible=new.supervised_eligible_v2.to_numpy()==1
 y=old[labels].to_numpy();expect=y.astype(float);expect[y[:,1:].sum(1)>0,0]=0;expect[~eligible]=np.nan
 assert np.array_equal(expect,new[['target_v2_'+k for k in labels]].to_numpy(),equal_nan=True)
 assert np.array_equal(eligible,y.sum(1)>0)
 assert all(not set(new[new.split==a].patient_id)&set(new[new.split==b].patient_id) for a,b in [('train','validation'),('train','test'),('validation','test')])
 changes=[];support=[]
 for e in old.index:
  for k in labels:
   if new.loc[e,'supervised_eligible_v2'] and old.loc[e,k]!=new.loc[e,k]:changes.append({'ecg_id':e,'patient_id':int(old.loc[e,'patient_id']),'split':old.loc[e,'split'],'label':k,'v1':int(old.loc[e,k]),'v2':int(new.loc[e,k])})
 for split in ['train','validation','test']:
  d=new[new.split==split];m=d.supervised_eligible_v2==1
  for k in labels:support.append({'split':split,'label':k,'all_records':len(d),'supervised':int(m.sum()),'v1_positive':int(old.loc[d.index,k].sum()),'v2_positive':int(d.loc[m,k].sum())})
 pd.DataFrame(changes).to_csv(out/'label_changes.csv',index=False);pd.DataFrame(support).to_csv(out/'label_distribution.csv',index=False)
 metrics=pd.read_csv(O/'results/metrics_by_seed.csv');checks=[]
 for path in (O/'results/classification_abc/experiment_20261101/models').glob('*/*/seed_*/predictions.npz'):
  task,inp,seed=path.parts[-4:-1];seed=int(seed.split('_')[1])
  with np.load(path,allow_pickle=False) as z:
   assert np.array_equal(old.loc[z['ecg_id'],'split'],z['split'])
   expected=old.loc[z['ecg_id'],labels].to_numpy()
   assert np.array_equal(expected.sum(1)>0,z['eligible'])
   if task=='binary':expected=(expected[:,1:].sum(1,keepdims=True)>0).astype(int)
   assert np.array_equal(z['target'][z['eligible']],expected[z['eligible']])
   for split in ['train','validation','test']:
    m=(z['split']==split)&z['eligible']
    for j,k in enumerate(z['labels']):
     y=z['target'][m,j];p=z['predicted'][m,j]
     tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());tn=int(((y==0)&(p==0)).sum())
     vals={'tp':tp,'fp':fp,'tn':tn,'fn':fn,'precision':tp/(tp+fp) if tp+fp else np.nan,'recall':tp/(tp+fn) if tp+fn else np.nan,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan}
     row=metrics[(metrics.task==task)&(metrics.input==inp)&(metrics.seed==seed)&(metrics.split==split)&(metrics.label==k)].iloc[0]
     for key,val in vals.items():assert np.isclose(row[key],val,atol=1e-10,equal_nan=True)
     checks.append({'task':task,'input':inp,'seed':seed,'split':split,'label':str(k),'matches':True})
 pd.DataFrame(checks).to_csv(out/'historical_metric_checks.csv',index=False)
 (out/'attachment_audit.json').write_text(json.dumps({'stored_metric_rows_verified':len(checks),'label_changes':len(changes),'disease_label_changes':sum(c['label']!='NORM' for c in changes),'patient_overlap':0,'unclassified_excluded':int((~eligible).sum())},indent=2))
 print('ATTACHMENT AUDIT PASSED',len(checks))
if __name__=='__main__':main()
