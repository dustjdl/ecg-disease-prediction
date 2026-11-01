"""Validation-only postprocessing comparison. No fitting, test scoring or label changes."""
from pathlib import Path
import argparse, ast, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):
    with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def derive(i,ii):
    return {'III':ii-i,'AVR':-(i+ii)/2,'AVL':i-ii/2,'AVF':ii-i/2}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--npz',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    labels=pd.read_csv(a.labels); meta=pd.read_csv(a.metadata)
    with np.load(a.npz,allow_pickle=False) as d:
        leads=d['lead_names'].tolist(); splits=d['split']; ids=d['ecg_id']; patients=d['patient_id']
        assert len(set(leads))==12 and len(set(ids))==len(ids)
        assert int(d['sampling_rate_hz'])==100 and str(d['voltage_unit'])=='mV'
        label=labels.set_index('ecg_id').loc[ids]
        assert np.array_equal(label.patient_id,patients) and np.array_equal(label.split,splits)
        for sa,sb in [('train','validation'),('train','test'),('validation','test')]:
            assert set(patients[splits==sa]).isdisjoint(patients[splits==sb])
        mask=splits=='validation'
        # Deliberately slice validation BEFORE waveform computations. No test metrics.
        actual=d['actual_12lead'][mask].astype(np.float64)
        pred=d['reconstructed_12lead'][mask].astype(np.float64)
        val_ids=ids[mask];val_patient=patients[mask]
        assert actual.shape==pred.shape==(100,1000,12)
        assert np.isfinite(actual).all() and np.isfinite(pred).all()
        j={n:leads.index(n) for n in leads}
        assert np.array_equal(actual[:,:,j['I']],pred[:,:,j['I']])
        hybrid=pred.copy();derived=derive(actual[:,:,j['I']],pred[:,:,j['II']])
        for k,v in derived.items():hybrid[:,:,j[k]]=v
        unchanged=['I','II','V1','V2','V3','V4','V5','V6']
        assert np.array_equal(pred[:,:,[j[k] for k in unchanged]],hybrid[:,:,[j[k] for k in unchanged]])
        # Independent formula fixtures and exact limb-relation checks.
        assert derive(np.array([2.]),np.array([3.]))=={'III':np.array([1.]),'AVR':np.array([-2.5]),'AVL':np.array([.5]),'AVF':np.array([2.])}
        assert np.allclose(hybrid[:,:,j['II']],hybrid[:,:,j['I']]+hybrid[:,:,j['III']],atol=1e-12)
        true_derived=derive(actual[:,:,j['I']],actual[:,:,j['II']])
        rows=[]; per=[];quality=[]
        for name,idx in [(k,[j[k]]) for k in leads if k!='I']+[('ALL_11',[j[k] for k in leads if k!='I']),('DERIVED_4',[j[k] for k in derived])]:
            eb=pred[:,:,idx]-actual[:,:,idx];eh=hybrid[:,:,idx]-actual[:,:,idx]
            bm=np.mean(eb**2,axis=(1,2));hm=np.mean(eh**2,axis=(1,2))
            ba=np.mean(np.abs(eb),axis=(1,2));ha=np.mean(np.abs(eh),axis=(1,2))
            rows.append({'scope':name,'n_records':len(val_ids),'direct_MAE_mV':ba.mean(),'formula_MAE_mV':ha.mean(),'MAE_reduction_percent':100*(ba.mean()-ha.mean())/ba.mean(),'direct_MSE_mV2':bm.mean(),'formula_MSE_mV2':hm.mean(),'MSE_reduction_percent':100*(bm.mean()-hm.mean())/bm.mean(),'direct_pooled_RMSE_mV':np.sqrt(bm.mean()),'formula_pooled_RMSE_mV':np.sqrt(hm.mean()),'MSE_improved_records':int((hm<bm-1e-12).sum()),'MSE_worsened_records':int((hm>bm+1e-12).sum()),'MSE_tied_records':int((abs(hm-bm)<=1e-12).sum())})
            for n,e,pa in zip(range(len(val_ids)),val_ids,val_patient):per.append({'ecg_id':e,'patient_id':pa,'split':'validation','scope':name,'direct_MAE_mV':ba[n],'formula_MAE_mV':ha[n],'direct_MSE_mV2':bm[n],'formula_MSE_mV2':hm[n],'MSE_delta_formula_minus_direct':hm[n]-bm[n]})
        for k,v in true_derived.items():
            residual=v-actual[:,:,j[k]]
            quality.append({'lead':k,'n_records':len(val_ids),'actual_formula_residual_MAE_mV':abs(residual).mean(),'actual_formula_residual_max_abs_mV':abs(residual).max()})
        summary=pd.DataFrame(rows);summary.to_csv(a.output/'validation_comparison.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(per).to_csv(a.output/'validation_record_comparison.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(quality).to_csv(a.output/'actual_limb_formula_residual.csv',index=False,encoding='utf-8-sig')
        # Fixed min-ID waveform example; not chosen using performance.
        ni=int(np.argmin(val_ids));time=np.arange(1000)/100
        fig,axs=plt.subplots(4,1,figsize=(11,8),sharex=True)
        for ax,k in zip(axs,derived):
            ax.plot(time,actual[ni,:,j[k]],color='black',lw=1,label='Actual')
            ax.plot(time,pred[ni,:,j[k]],color='#527ab2',lw=.9,alpha=.8,label='Direct AI')
            ax.plot(time,hybrid[ni,:,j[k]],color='#df8b36',lw=.9,alpha=.8,label='Formula')
            ax.set_ylabel(k+' (mV)');ax.grid(alpha=.15)
        axs[0].legend(ncol=3);axs[-1].set_xlabel('Time (s)');fig.suptitle(f'Validation ECG {val_ids[ni]}: smallest ID, not selected by error')
        fig.tight_layout();fig.savefig(a.output/'waveform_fixed_example.png',dpi=160);plt.close(fig)
        s=summary.set_index('scope').loc[list(derived)];xx=np.arange(4)
        fig,axs=plt.subplots(1,2,figsize=(10,4.5))
        for ax,metric,unit in zip(axs,['MAE','MSE'],['mV','mV²']):
            suffix='mV' if metric=='MAE' else 'mV2'
            ax.bar(xx-.18,s[f'direct_{metric}_{suffix}'],.36,color='#527ab2',label='Direct AI')
            ax.bar(xx+.18,s[f'formula_{metric}_{suffix}'],.36,color='#df8b36',label='I + predicted II formula')
            ax.set_xticks(xx,s.index);ax.set_ylabel(f'{metric} ({unit})');ax.spines[['top','right']].set_visible(False)
        axs[0].legend(fontsize=8);fig.suptitle('Validation only: same 100 ECGs, unchanged trained model');fig.tight_layout();fig.savefig(a.output/'formula_comparison.png',dpi=170);plt.close(fig)
    mixed=labels[(labels.NORM==1)&(labels[['MI','STTC','CD','HYP']].sum(axis=1)>0)].copy()
    src=meta[['ecg_id','patient_id','report','scp_codes','validated_by_human']].rename(columns={'patient_id':'source_patient_id','scp_codes':'source_scp_codes'})
    mixed=mixed.merge(src,on='ecg_id',validate='one_to_one')
    assert np.array_equal(mixed.patient_id,mixed.source_patient_id)
    assert mixed.scp_codes.eq(mixed.source_scp_codes).all()
    mixed['review_status']='pending_source_review_not_confirmed_error'
    mixed['recommended_next_step']='Review original SCP codes, likelihood and report; agree a target policy before any relabel/retraining.'
    mixed['reviewer_decision']='';mixed['reviewer_reason']=''
    mixed.to_csv(a.output/'norm_mixed_review_16.csv',index=False,encoding='utf-8-sig')
    codes=[]
    for _,r in mixed.iterrows():
        for code,likelihood in ast.literal_eval(r.scp_codes).items():codes.append({'ecg_id':r.ecg_id,'split':r.split,'scp_code':code,'likelihood':likelihood,'note':'0 means likelihood unknown, not absence' if likelihood==0 else ''})
    pd.DataFrame(codes).to_csv(a.output/'norm_mixed_code_details.csv',index=False,encoding='utf-8-sig')
    info={'source_sha256':{str(f.name):sha(f) for f in [a.npz,a.labels,a.metadata]},'analysis':'validation_only_postprocessing_no_training','validation_records':100,'test_waveforms_scored':False,'changed_leads':list(derived),'unchanged_leads':unchanged,'labels_changed':False,'mixed_records_by_split':{str(k):int(v) for k,v in mixed.split.value_counts().items()},'fixed_waveform_example_ecg_id':int(val_ids[ni]),'metrics_definition':'MSE and MAE pooled across same length ECGs/time/leads; RMSE=sqrt(pooled MSE).'}
    (a.output/'audit.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary[summary.scope.isin(list(derived)+['ALL_11','DERIVED_4'])].to_string(index=False))
    print('Mixed',len(mixed),mixed.split.value_counts().to_dict())

if __name__=='__main__':main()
