"""Fair label-policy contrast: both 18-model runs trained on the same restored CPU inputs."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'preserved/role3'));import train_ecg_classification as t

def main():
 for policy in ['v1','v2']:assert json.loads((R/'runs'/policy/'completion.json').read_text())['models_trained']==18
 pair_checks=[]
 for cfgpath in (R/'runs/v1/models').glob('*/*/seed_*/config.json'):
  rel=cfgpath.relative_to(R/'runs/v1')
  a=json.loads(cfgpath.read_text());b=json.loads((R/'runs/v2'/rel).read_text())
  for key in ['task','input','seed','labels','parameter_count','structure','n_train','n_validation','train_ids','validation_ids','normalization_mean','normalization_sd']:
   assert a[key]==b[key],(rel,key)
  for key in a['protocol']:
   if key!='multilabel_mixed':assert a['protocol'][key]==b['protocol'][key],(rel,key)
  ha=pd.read_csv(cfgpath.parent/'train_order_digests.csv');hb=pd.read_csv((R/'runs/v2'/rel).parent/'train_order_digests.csv');assert ha.equals(hb)
  pair_checks.append(str(rel))
 (R/'reports/controlled_comparison_checks.json').write_text(json.dumps({'status':'passed','paired_models':len(pair_checks),'same_architecture_hyperparameters_ids_normalizers_training_order':True,'differing_training_factor':'label policy only; resulting class weights/checkpoints/thresholds may change'},indent=2))
 rows=[];frame=pd.read_csv(R/'runs/v2/cohort.csv').set_index('ecg_id')
 for path in (R/'runs/v1/models').glob('*/*/seed_*/predictions.npz'):
  task,inp,seed=path.parts[-4:-1];seed=int(seed.split('_')[1])
  with np.load(path,allow_pickle=False) as z:
   ordered=frame.loc[z['ecg_id']].reset_index();y,eligible=t.targets(ordered,task)
   assert np.array_equal(z['eligible'],eligible)
   mask=(z['split']=='test')&eligible;per=[]
   for k,label in enumerate(z['labels']):
    m=t.binary_metrics(y[mask,k],z['probability'][mask,k],z['predicted'][mask,k]);per.append(m)
    rows.append({'task':task,'input':inp,'seed':seed,'split':'test','label':str(label),**m})
   if task=='multilabel':
    rows.append({'task':task,'input':inp,'seed':seed,'split':'test','label':'micro',**t.binary_metrics(y[mask].ravel(),z['probability'][mask].ravel(),z['predicted'][mask].ravel())})
    rows.append({'task':task,'input':inp,'seed':seed,'split':'test','label':'macro',**{k:float(np.nanmean([m[k] for m in per])) for k in ['precision','recall','f1']}})
 old=pd.DataFrame(rows);old.to_csv(R/'reports/v1_cpu_on_common_v2_targets.csv',index=False)
 new=pd.read_csv(R/'runs/v2/metrics_by_seed.csv');new=new[new.split=='test']
 keys=['task','input','seed','split','label'];fields=['precision','recall','f1']
 c=old[keys+fields].merge(new[keys+fields],on=keys,how='outer',suffixes=('_baseline','_v2'),validate='one_to_one',indicator=True)
 assert (c._merge=='both').all()
 for k in fields:c[k+'_delta']=c[k+'_v2']-c[k+'_baseline']
 c.to_csv(R/'reports/comparison_same_cpu_common_v2_by_seed.csv',index=False)
 nums=[k+s for k in fields for s in ['_baseline','_v2','_delta']]
 c.groupby(['task','input','label'])[nums].agg(['mean','std','count']).to_csv(R/'reports/comparison_same_cpu_common_v2_mean_sd.csv')
 # Binary targets are identical: check whether CPU deterministic repetitions match.
 binary=[]
 for path in (R/'runs/v1/models/binary').glob('*/seed_*/predictions.npz'):
  rel=path.relative_to(R/'runs/v1')
  with np.load(path,allow_pickle=False) as a,np.load(R/'runs/v2'/rel,allow_pickle=False) as b:
   binary.append({'input':path.parts[-3],'seed':path.parts[-2],'max_probability_difference':float(np.abs(a['probability']-b['probability']).max()),'prediction_differences':int((a['predicted']!=b['predicted']).sum())})
 pd.DataFrame(binary).to_csv(R/'reports/binary_same_cpu_repeat_check.csv',index=False)
 f=c[(c.task=='multilabel')&c.label.isin(t.LABELS)]
 fig,axes=plt.subplots(3,3,figsize=(14,11),layout='constrained')
 for i,inp in enumerate(t.INPUTS):
  for j,k in enumerate(fields):
   d=f[f.input==inp].groupby('label')[[k+'_baseline',k+'_v2']].agg(['mean','std']).reindex(t.LABELS)
   ix=np.arange(5);ax=axes[i,j]
   for shift,col,color,label in [(-.18,k+'_baseline','#64748b','CPU retrained v1 labels'),(.18,k+'_v2','#2563eb','CPU retrained v2 labels')]:
    ax.bar(ix+shift,d[col]['mean'],yerr=d[col]['std'],width=.36,capsize=2,label=label,color=color)
   ax.set_xticks(ix,t.LABELS);ax.set_ylim(0,1.05);ax.set_title(inp+' / '+k);ax.grid(axis='y',alpha=.2)
 axes[0,0].legend(fontsize=8)
 fig.suptitle('Same CPU / restored inputs / protocol; common v2 test targets\nMean +/- sample SD over seeds 42/43/44; 98 eligible test ECGs')
 fig.savefig(R/'reports/same_cpu_common_v2_per_class.png',dpi=160);plt.close(fig)
 print(c[(c.task=='multilabel')&(c.input=='C')].groupby('label')[nums].mean().to_string())
if __name__=='__main__':main()
