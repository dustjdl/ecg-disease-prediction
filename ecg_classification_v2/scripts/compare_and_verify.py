"""Evaluate preserved checkpoints on restored inputs, and compare on common v2 targets."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'preserved/role3'
sys.path.insert(0,str(O));import train_ecg_classification as t

def main():
 torch.set_num_threads(4)
 run=R/'runs/v2'
 assert json.loads((run/'completion.json').read_text())['models_trained']==18
 with np.load(R/'recovery/ecg_for_classification.npz',allow_pickle=False) as z:
  ids=z['ecg_id'];actual=z['actual_12lead'];recon=z['reconstructed_12lead'];splits=z['split']
 values={'A':actual[:,:,:1].transpose(0,2,1),'B':actual.transpose(0,2,1),'C':recon.transpose(0,2,1)}
 v1=pd.read_csv(O/'results/cohort.csv').set_index('ecg_id').loc[ids].reset_index()
 v2=pd.read_csv(run/'cohort.csv').set_index('ecg_id').loc[ids].reset_index()
 rows=[];consistency=[]
 historical=O/'results/classification_abc/experiment_20261101/models'
 for task in ['multilabel','binary']:
  for inp in t.INPUTS:
   for seed in [42,43,44]:
    folder=historical/task/inp/f'seed_{seed}'
    ck=torch.load(folder/'best_model.pth',weights_only=True,map_location='cpu');cfg=ck['config']
    model=t.Classifier(values[inp].shape[1],len(cfg['labels']),seed);model.load_state_dict(ck['model_state_dict']);model.eval()
    mean=np.asarray(cfg['normalization_mean']);sd=np.asarray(cfg['normalization_sd'])
    computed_mean,computed_sd,_=t.normalization(values[inp],(splits=='train')&(v1[t.LABELS].sum(axis=1).to_numpy()>0))
    x=np.ascontiguousarray((values[inp]-mean[None,:,None])/sd[None,:,None],dtype=np.float32)
    prob=t.probabilities(model,x,torch.device('cpu'))
    threshold=np.array([q['threshold'] for q in json.loads((folder/'thresholds.json').read_text())]);pred=prob>=threshold
    with np.load(folder/'predictions.npz',allow_pickle=False) as saved:
     assert np.array_equal(ids,saved['ecg_id'])
     consistency.append({'task':task,'input':inp,'seed':seed,'max_normalization_mean_difference':float(np.abs(computed_mean-mean).max()),'max_normalization_sd_difference':float(np.abs(computed_sd-sd).max()),'max_abs_probability_difference':float(np.abs(prob-saved['probability']).max()),'predicted_cell_differences':int((pred!=saved['predicted']).sum())})
    np.savez_compressed(R/'reports'/f'v1_reinference_{task}_{inp}_{seed}.npz',ecg_id=ids,probability=prob,predicted=pred,thresholds=threshold)
    for policy,frame in [('v1',v1),('v2',v2)]:
     y,eligible=t.targets(frame,task)
     for split in ['validation','test']:
      mask=(splits==split)&eligible;per=[]
      for k,label in enumerate(cfg['labels']):
       m=t.binary_metrics(y[mask,k],prob[mask,k],pred[mask,k]);per.append(m)
       rows.append({'model':'v1_preserved_weights_recovered_input','target_policy':policy,'task':task,'input':inp,'seed':seed,'split':split,'label':label,**m})
      if task=='multilabel':
       rows.append({'model':'v1_preserved_weights_recovered_input','target_policy':policy,'task':task,'input':inp,'seed':seed,'split':split,'label':'micro',**t.binary_metrics(y[mask].ravel(),prob[mask].ravel(),pred[mask].ravel())})
       rows.append({'model':'v1_preserved_weights_recovered_input','target_policy':policy,'task':task,'input':inp,'seed':seed,'split':split,'label':'macro',**{k:float(np.nanmean([m[k] for m in per])) for k in ['precision','recall','f1']}})
 old=pd.DataFrame(rows);old.to_csv(R/'reports/v1_reinference_metrics.csv',index=False)
 pd.DataFrame(consistency).to_csv(R/'reports/restored_vs_historical_predictions.csv',index=False)
 new=pd.read_csv(run/'metrics_by_seed.csv');historical_metrics=pd.read_csv(O/'results/metrics_by_seed.csv')
 keys=['task','input','seed','split','label'];fields=['precision','recall','f1']
 for mode,baseline in [('common_v2_targets',old[old.target_policy=='v2']),('historical_v1_targets',historical_metrics)]:
  base=baseline[baseline.split=='test'];v=new[new.split=='test']
  comp=base[keys+fields].merge(v[keys+fields],on=keys,how='outer',suffixes=('_baseline','_v2'),validate='one_to_one',indicator=True)
  assert (comp._merge=='both').all()
  for k in fields:comp[k+'_delta']=comp[k+'_v2']-comp[k+'_baseline']
  comp['comparison']=mode
  comp.to_csv(R/'reports'/f'comparison_{mode}_by_seed.csv',index=False)
  nums=[k+s for k in fields for s in ['_baseline','_v2','_delta']]
  comp.groupby(['task','input','label'])[nums].agg(['mean','std','count']).to_csv(R/'reports'/f'comparison_{mode}_mean_sd.csv')
 comp=pd.read_csv(R/'reports/comparison_common_v2_targets_by_seed.csv')
 f=comp[(comp.task=='multilabel')&comp.label.isin(t.LABELS)]
 fig,axes=plt.subplots(3,3,figsize=(14,11),layout='constrained')
 for i,inp in enumerate(t.INPUTS):
  for j,k in enumerate(fields):
   g=f[f.input==inp].groupby('label')[[k+'_baseline',k+'_v2']].agg(['mean','std']).reindex(t.LABELS)
   ax=axes[i,j];ix=np.arange(5)
   for shift,col,color,label in [(-.18,k+'_baseline','#64748b','Preserved v1 weights'),(.18,k+'_v2','#2563eb','Retrained v2')]:
    ax.bar(ix+shift,g[col]['mean'],yerr=g[col]['std'],width=.36,capsize=2,label=label,color=color)
   ax.set_xticks(ix,t.LABELS);ax.set_ylim(0,1.05);ax.set_title(inp+' / '+k);ax.grid(axis='y',alpha=.2)
 axes[0,0].legend(fontsize=8)
 fig.suptitle('Common v2 test targets; means +/- sample SD over seeds 42/43/44\nRecovered inputs and CPU; same 98 eligible test ECGs')
 fig.savefig(R/'reports/common_v2_per_class.png',dpi=160);plt.close(fig)
 print(pd.DataFrame(consistency).to_string(index=False))
 print('COMPARISON COMPLETE')
if __name__=='__main__':main()
