"""Independently recompute validation metrics and produce aligned comparisons."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
METRICS=['mse_mV2','mae_mV','pooled_rmse_mV']

def metrics(y,p):
    e=p.astype(np.float64)-y.astype(np.float64)
    mse=np.mean(e*e,axis=(0,1)); mae=np.mean(np.abs(e),axis=(0,1))
    return mse,mae,np.sqrt(mse)

def load_predictions(folder,ids,leads):
    with np.load(folder/'validation_predictions.npz',allow_pickle=False) as z:
        assert np.array_equal(z['ecg_id'],ids), 'ECG ID/order mismatch'
        assert np.array_equal(z['lead_names'],leads), 'Lead order mismatch'
        assert int(z['sampling_rate_hz'])==100 and str(z['voltage_unit'])=='mV'
        p=z['predicted_11lead'].copy()
    assert p.shape==(100,1000,11) and np.isfinite(p).all()
    return p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--unet',type=Path,default=ROOT/'results/unet');ap.add_argument('--cnn',type=Path,default=ROOT/'reference/cnn_baseline');ap.add_argument('--output',type=Path,default=ROOT/'results/comparison');args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False);figs=out/'figures';figs.mkdir()
    with np.load(ROOT/'data/ecg_actual_100hz_998.npz',allow_pickle=False) as d:
        mask=d['split']=='validation';actual=d['actual_12lead'][mask];ids=d['ecg_id'][mask];leads=d['lead_names'][1:]
    configs=[json.loads((p/'experiment_config.json').read_text()) for p in [args.cnn,args.unet]]
    for k in ['sampling_rate_hz','duration_seconds','train','validation','test','test_evaluated','seeds','epochs','batch_size','learning_rate','optimizer','loss','preprocessing','checkpoint_selection','labels_sha256','device','threads']:
        assert configs[0][k]==configs[1][k],f'Protocol mismatch: {k}'
    assert configs[1]['seeds']==[42,43,44] and configs[1]['epochs']==30
    frames=[pd.read_csv(p/'cohort.csv') for p in [args.cnn,args.unet]];pd.testing.assert_frame_equal(*frames)
    rows=[];preds={};histories={};verification=[]
    for name,path in [('CNN',args.cnn),('UNet',args.unet)]:
        reported=pd.read_csv(path/'validation_metrics_by_seed.csv').set_index(['seed','lead'])
        for seed in [42,43,44]:
            folder=path/f'seed_{seed}';p=load_predictions(folder,ids,leads);preds[name,seed]=p
            hist=pd.read_csv(folder/'training_history.csv');assert hist.epoch.tolist()==list(range(1,31));histories[name,seed]=hist
            run=json.loads((folder/'run_summary.json').read_text());assert run['best_epoch']==int(hist.loc[hist.validation_mse.idxmin(),'epoch'])
            m,a,r=metrics(actual[:,:,1:],p)
            for j,lead in enumerate(leads): rows.append(dict(model=name,seed=seed,lead=lead,mse_mV2=m[j],mae_mV=a[j],pooled_rmse_mV=r[j]))
            rows.append(dict(model=name,seed=seed,lead='ALL_11',mse_mV2=m.mean(),mae_mV=a.mean(),pooled_rmse_mV=np.sqrt(m.mean())))
            for row in rows[-12:]:
                np.testing.assert_allclose([row[k] for k in METRICS],reported.loc[(seed,row['lead']),METRICS].to_numpy(dtype=float),rtol=1e-10,atol=1e-12)
            verification.append(dict(model=name,seed=seed,predictions_aligned=True,metrics_recomputed=True,best_epoch_checked=True))
    df=pd.DataFrame(rows);df.to_csv(out/'metrics_recomputed.csv',index=False)
    summary=df.groupby(['model','lead'])[METRICS].agg(['mean','std']);summary.to_csv(out/'metrics_mean_std.csv')
    paired=df[df.model=='CNN'].merge(df[df.model=='UNet'],on=['seed','lead'],suffixes=('_cnn','_unet'))
    for k in METRICS:
        paired[k+'_delta_unet_minus_cnn']=paired[k+'_unet']-paired[k+'_cnn']
        paired[k+'_improvement_percent']=100*(paired[k+'_cnn']-paired[k+'_unet'])/paired[k+'_cnn']
    paired.to_csv(out/'paired_seed_lead_comparison.csv',index=False)
    means=df.groupby(['model','lead'])[METRICS].mean();table=[]
    for lead in ['ALL_11']+list(leads):
        row={'lead':lead}
        for k in METRICS:
            c,u=means.loc[('CNN',lead),k],means.loc[('UNet',lead),k]
            row.update({f'cnn_{k}':c,f'unet_{k}':u,f'{k}_improvement_percent':100*(c-u)/c})
        table.append(row)
    pd.DataFrame(table).to_csv(out/'comparison_by_lead.csv',index=False)
    plt.rcParams.update({'font.size':10,'figure.dpi':140})
    colors={'CNN':'#e09a37','UNet':'#367dcc'}
    fig,axs=plt.subplots(1,3,figsize=(15,4.3))
    for ax,k,title in zip(axs,METRICS,['MSE (mV²)','MAE (mV)','Pooled RMSE (mV)']):
        for x,name in enumerate(['CNN','UNet']):
            vals=df[(df.model==name)&(df.lead=='ALL_11')][k];ax.bar(x,vals.mean(),yerr=vals.std(),color=colors[name],alpha=.75,capsize=5);ax.scatter(np.full(3,x),vals,color='black',s=18,zorder=4)
        ax.set_xticks([0,1],['CNN','U-Net']);ax.set_ylabel(title);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
    fig.suptitle('100 Hz validation | 3 seeds, mean ± sample SD | observed Lead I excluded');fig.tight_layout();fig.savefig(figs/'overall_metrics.png');plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(12,10))
    x=np.arange(11)
    for ax,k,label in zip(axs,METRICS,['MSE (mV²)','MAE (mV)','Pooled RMSE (mV)']):
        for offset,name in [(-.19,'CNN'),(.19,'UNet')]:
            vals=[summary.loc[(name,l),(k,'mean')] for l in leads];sd=[summary.loc[(name,l),(k,'std')] for l in leads]
            ax.bar(x+offset,vals,width=.38,label=name,color=colors[name],yerr=sd,capsize=2)
        ax.set_xticks(x,leads);ax.set_ylabel(label);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Validation error by reconstructed lead | mean ± SD across 3 seeds');fig.tight_layout();fig.savefig(figs/'lead_errors.png');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for ax,seed in zip(axs,[42,43,44]):
        for name in ['CNN','UNet']:
            h=histories[name,seed];ax.plot(h.epoch,h.validation_mse,color=colors[name],label=f'{name} val');ax.plot(h.epoch,h.train_mse,color=colors[name],ls='--',alpha=.65,label=f'{name} train')
        ax.set_title(f'Seed {seed}');ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axs[0].set_ylabel('MSE (mV²)');axs[-1].legend(fontsize=8);fig.suptitle('Identical training budget: 30 epochs, batch 8, Adam 0.001');fig.tight_layout();fig.savefig(figs/'learning_curves.png');plt.close(fig)
    # Fixed, model-independent examples: three smallest validation ECG IDs, seed 42.
    chosen=np.argsort(ids)[:3];pd.DataFrame({'ecg_id':ids[chosen],'seed':42,'selection':'three smallest validation ECG IDs; no performance selection'}).to_csv(out/'waveform_selection.csv',index=False)
    for idx in chosen:
        fig,axs=plt.subplots(6,2,figsize=(15,13),sharex=True);t=np.arange(1000)/100
        for j,ax in enumerate(axs.flat):
            name='I' if j==0 else str(leads[j-1]);ax.plot(t,actual[idx,:,j],color='#252525',lw=.8,label='Actual')
            if j:
                for model in ['CNN','UNet']:ax.plot(t,preds[model,42][idx,:,j-1],color=colors[model],lw=.75,alpha=.85,label=model)
            ax.set_title(f'{name}'+(' (input; not scored)' if j==0 else ''),fontsize=10);ax.set_ylabel('mV');ax.grid(alpha=.15)
        axs[0,1].legend(ncol=3,fontsize=8);axs[-1,0].set_xlabel('Time (s)');axs[-1,1].set_xlabel('Time (s)')
        fig.suptitle(f'Validation ECG {ids[idx]} | seed 42 | actual vs reconstruction | 100 Hz');fig.tight_layout();fig.savefig(figs/f'waveform_ecg_{ids[idx]}.png');plt.close(fig)
    # Per-record error distribution across seed averages (not ensemble predictions).
    record=[]
    for i,eid in enumerate(ids):
        row={'ecg_id':int(eid)}
        for name in ['CNN','UNet']:
            vals=[metrics(actual[i:i+1,:,1:],preds[name,s][i:i+1])[0].mean() for s in [42,43,44]];row[name+'_mse']=float(np.mean(vals))
        row['delta_unet_minus_cnn']=row['UNet_mse']-row['CNN_mse'];record.append(row)
    records=pd.DataFrame(record);records.to_csv(out/'per_record_seed_mean_mse.csv',index=False)
    fig,ax=plt.subplots(figsize=(5.5,5));ax.scatter(records.CNN_mse,records.UNet_mse,alpha=.7,s=20);lim=max(records.CNN_mse.max(),records.UNet_mse.max())*1.05;ax.plot([0,lim],[0,lim],'k--',lw=1);ax.set(xlabel='CNN MSE (mV²)',ylabel='U-Net MSE (mV²)',title='100 validation records | mean over 3 seeds');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figs/'per_record_mse.png');plt.close(fig)
    (out/'verification.json').write_text(json.dumps({'checks':verification,'identical_protocol':True,'identical_cohort':True,'test_evaluated':False,'waveforms_not_cherry_picked':True,'records_improved_mse':int((records.delta_unet_minus_cnn<0).sum())},indent=2))
    print(pd.DataFrame(table).head(1).to_string(index=False));print('Saved',out)
if __name__=='__main__':main()
