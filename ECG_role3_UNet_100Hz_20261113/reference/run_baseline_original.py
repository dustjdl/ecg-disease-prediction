"""Existing CNN, 100 Hz, v3 groups; validation-only evaluation, no test forward pass."""
import argparse, hashlib, json, random, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from model import ECGModel

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--npz',required=True); ap.add_argument('--labels',required=True); ap.add_argument('--output',required=True); ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--validate-only',action='store_true'); args=ap.parse_args()
    torch.set_num_threads(4)
    labels=pd.read_csv(args.labels); assert labels.ecg_id.is_unique
    with np.load(args.npz,allow_pickle=False) as d:
        assert int(d['sampling_rate_hz'])==100 and str(d['voltage_unit'])=='mV'
        ids=d['ecg_id'].copy(); patients=d['patient_id'].copy(); split=d['split'].copy(); leads=d['lead_names'].astype(str).tolist()
        assert len(ids)==998 and len(set(ids))==998
        assert leads==['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
        assert d['actual_12lead'].shape==(998,1000,12)
        assert np.array_equal(d['actual_12lead'][:,:,:1],d['lead_i'])
        # Actual measured waveforms only. Never use reconstructed_12lead as a training target.
        train=d['actual_12lead'][split=='train'].copy(); val=d['actual_12lead'][split=='validation'].copy()
    assert np.isfinite(train).all() and np.isfinite(val).all()
    assert {s:int((split==s).sum()) for s in ['train','validation','test']}=={'train':798,'validation':100,'test':100}
    assert not ({421,710}&set(ids))
    for a,b in [('train','validation'),('train','test'),('validation','test')]: assert not (set(patients[split==a])&set(patients[split==b]))
    labels=labels.set_index('ecg_id').loc[ids].reset_index()
    assert np.array_equal(labels.patient_id.to_numpy(),patients) and np.array_equal(labels.split.to_numpy(),split)
    assert (labels.sampling_rate_hz==100).all() and (labels.v3_reconstruction_eligible==1).all()
    for n in [1000,5000]:
        with torch.no_grad(): assert ECGModel()(torch.zeros(1,1,n)).shape==(1,11,n)
    print('VALIDATED: 998 IDs, patient split, actual signals, lead order, v3 labels; synthetic shape checks passed.',flush=True)
    if args.validate_only:return
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    config={'model':'model.ECGModel (original basic CNN, not Dilated)','sampling_rate_hz':100,'duration_seconds':10,'train':798,'validation':100,'test':100,'test_evaluated':False,'seeds':args.seeds,'epochs':args.epochs,'batch_size':8,'learning_rate':0.001,'optimizer':'Adam','loss':'mean MSE, all 11 estimated leads','preprocessing':'none; mV; no filtering, clipping, normalization or augmentation','checkpoint_selection':'lowest validation MSE, strict improvement','label_role':'v3 diagnosis groups for validation analysis only; no diagnostic labels in reconstruction loss','npz_sha256':sha(args.npz),'labels_sha256':sha(args.labels),'model_sha256':sha(Path(__file__).with_name('model.py')),'script_sha256':sha(__file__),'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'device':'cpu','threads':4,'note':'Exploratory 100Hz baseline for future matched U-Net / 500Hz comparisons; shared protocol not yet confirmed. No claimed improvement from v3 labels.'}
    save(out/'experiment_config.json',config)
    pd.DataFrame({'ecg_id':ids,'patient_id':patients,'split':split}).to_csv(out/'cohort.csv',index=False)
    tx=torch.from_numpy(train[:,:,:1].transpose(0,2,1).copy()); ty=torch.from_numpy(train[:,:,1:].transpose(0,2,1).copy()); vx=torch.from_numpy(val[:,:,:1].transpose(0,2,1).copy()); vy=torch.from_numpy(val[:,:,1:].transpose(0,2,1).copy())
    vmeta=labels.loc[split=='validation'].reset_index(drop=True); allmetrics=[]; allrecords=[]; grouprows=[]
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
        m=ECGModel(); opt=torch.optim.Adam(m.parameters(),lr=.001); loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx,ty),batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(seed))
        folder=out/f'seed_{seed}'; folder.mkdir(); best=float('inf'); hist=[]; start=time.perf_counter()
        for epoch in range(1,args.epochs+1):
            m.train(); total=0
            for x,y in loader:
                opt.zero_grad(); loss=nn.functional.mse_loss(m(x),y); assert torch.isfinite(loss); loss.backward(); opt.step(); total+=loss.item()*len(x)
            m.eval()
            with torch.no_grad():
                vl=sum(nn.functional.mse_loss(m(x),y).item()*len(x) for x,y in zip(vx.split(8),vy.split(8)))/len(vx)
            hist.append({'epoch':epoch,'train_mse':total/len(tx),'validation_mse':vl})
            if vl<best:
                best=vl; bestepoch=epoch; torch.save({'model_state_dict':m.state_dict(),'epoch':epoch,'seed':seed,'experiment':'cnn_100hz_v3_baseline','config':config},folder/'best_model.pth')
            if epoch%5==0: print(f'seed={seed} epoch={epoch}/{args.epochs} train={total/len(tx):.6f} val={vl:.6f}',flush=True)
        pd.DataFrame(hist).to_csv(folder/'training_history.csv',index=False)
        m.load_state_dict(torch.load(folder/'best_model.pth',map_location='cpu',weights_only=True)['model_state_dict']); m.eval()
        with torch.no_grad(): pred=torch.cat([m(x) for x in vx.split(8)]).numpy()
        np.savez_compressed(folder/'validation_predictions.npz',ecg_id=vmeta.ecg_id.to_numpy(),predicted_11lead=pred.transpose(0,2,1),lead_names=np.array(leads[1:]),sampling_rate_hz=np.array(100),voltage_unit=np.array('mV'))
        err=pred.astype('float64')-vy.numpy().astype('float64'); mse=(err**2).mean(axis=2); mae=np.abs(err).mean(axis=2)
        def metric(mask,lead):
            a=mse[mask] if lead is None else mse[mask,lead]; b=mae[mask] if lead is None else mae[mask,lead]
            return {'mse_mV2':float(a.mean()),'mae_mV':float(b.mean()),'pooled_rmse_mV':float(np.sqrt(a.mean()))}
        for j,name in [(None,'ALL_11')]+list(enumerate(leads[1:])):
            allmetrics.append(dict(seed=seed,lead=name,n=100,best_epoch=bestepoch,**metric(np.ones(100,dtype=bool),j)))
        for group in ['NORM','MI','STTC','CD','HYP','UNCLASSIFIED']:
            mask=(vmeta[[f'target_v3_{k}' for k in ['NORM','MI','STTC','CD','HYP']]].isna().all(axis=1).to_numpy() if group=='UNCLASSIFIED' else vmeta[f'target_v3_{group}'].eq(1).to_numpy())
            if mask.any():
                for j,name in [(None,'ALL_11')]+list(enumerate(leads[1:])): grouprows.append(dict(seed=seed,group=group,lead=name,n=int(mask.sum()),**metric(mask,j)))
        for i,row in vmeta.iterrows():
            for j,name in enumerate(leads[1:]): allrecords.append(dict(seed=seed,ecg_id=int(row.ecg_id),lead=name,mse_mV2=float(mse[i,j]),mae_mV=float(mae[i,j]),rmse_mV=float(np.sqrt(mse[i,j]))))
        save(folder/'run_summary.json',{'seed':seed,'best_epoch':bestepoch,'seconds':time.perf_counter()-start,'validation_mse':float(mse.mean()),'test_evaluated':False})
        print(f'FINISHED seed={seed} best_epoch={bestepoch} val_MSE={mse.mean():.6f} seconds={time.perf_counter()-start:.1f}',flush=True)
    pd.DataFrame(allmetrics).to_csv(out/'validation_metrics_by_seed.csv',index=False); pd.DataFrame(grouprows).to_csv(out/'validation_groups_by_seed.csv',index=False); pd.DataFrame(allrecords).to_csv(out/'validation_record_lead_metrics.csv',index=False)
    for rows,keys,name in [(allmetrics,['lead'],'validation_metrics_summary.csv'),(grouprows,['group','lead'],'validation_groups_summary.csv')]:
        f=pd.DataFrame(rows); f.groupby(keys)[['mse_mV2','mae_mV','pooled_rmse_mV']].agg(['mean','std']).to_csv(out/name)
    print('COMPLETE',out.resolve(),flush=True)
if __name__=='__main__':main()
