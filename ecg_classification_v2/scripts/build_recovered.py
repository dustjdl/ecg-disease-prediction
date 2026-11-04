"""Restore historical shapes/IDs; fixed saved reconstructor, no reconstructor training."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,torch,wfdb
R=Path(__file__).resolve().parents[1];O=R/'preserved/role3'
sys.path.insert(0,str(O))
from model_dilated import ECGDilatedModel
sys.path.insert(0,str(R/'preserved/repository/ecg_reconstruction'))
import prepare_dataset_1000 as prep
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4)
 d=pd.read_csv(R/'preserved/labels_v1/ecg_labels_all_1000.csv')
 assert len(d)==1000 and d.ecg_id.is_unique and d.patient_id.is_unique
 assert d.groupby('strat_fold').size().to_dict()=={i:100 for i in range(1,11)}
 expected=d.strat_fold.map({**{i:'train' for i in range(1,9)},9:'validation',10:'test'})
 assert expected.equals(d.split)
 x=[];y=[]
 for row in d.itertuples():
  a,b=prep.split_record(wfdb.rdrecord(str(R/'recovery/raw'/row.filename_lr)));x.append(a);y.append(b)
 x=np.stack(x);y=np.stack(y)
 paths=prep.output_paths(R/'recovery/processed')
 prep.write_outputs_atomically(paths,x,y,d[['ecg_id','patient_id','strat_fold','split','filename_lr']],pd.DataFrame(columns=['ecg_id','patient_id','strat_fold','reason']),False)
 config=json.loads((O/'results/dilated_1000/run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636/experiment_config.json').read_text())
 desired=[i for s in ['train','validation','test'] for i in config['split_ecg_ids'][s]]
 cohort=pd.read_csv(O/'results/cohort.csv')
 assert desired==cohort.ecg_id.tolist(),'frozen record ordering differs'
 assert set(d.ecg_id)-set(desired)=={421,710}
 ix=d.reset_index().set_index('ecg_id').loc[desired]['index'].to_numpy()
 actual=np.concatenate([x[ix],y[ix]],axis=2)
 checkpoint=O/'checkpoints/dilated_1000/run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636/best_model.pth'
 assert sha(checkpoint)=='e6cfd066ab8c88f1e37d5635beff7df2c3606509d0e1455392dab6642d8c6a64'
 ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
 assert ck['experiment']['split_ecg_ids']==config['split_ecg_ids']
 model=ECGDilatedModel();model.load_state_dict(ck['model_state_dict']);model.eval()
 parts=[]
 with torch.inference_mode():
  for i in range(0,998,32):parts.append(model(torch.from_numpy(x[ix][i:i+32].transpose(0,2,1))).numpy().transpose(0,2,1))
 reconstructed=np.concatenate([actual[:,:,:1],np.concatenate(parts)],axis=2)
 output=R/'recovery/ecg_for_classification.npz'
 np.savez_compressed(output,ecg_id=np.array(desired,dtype=np.int64),patient_id=d.patient_id.to_numpy()[ix],split=d.split.to_numpy(dtype='U10')[ix],lead_names=np.array(prep.EXPECTED_LEADS,dtype='U4'),sampling_rate_hz=np.array(100),voltage_unit=np.array('mV'),actual_12lead=actual,reconstructed_12lead=reconstructed,lead_i=actual[:,:,:1])
 expected_hash=json.loads((O/'DATA_REQUIREMENTS.json').read_text())['data_files']
 status={'records':1000,'analysis_records':998,'excluded':[421,710],'sampling_rate_hz':100,'X_shape':list(x.shape),'Y_shape':list(y.shape),'analysis_shape':list(actual.shape),'checkpoint_sha256':sha(checkpoint),'recovery_device':'cpu','torch':str(torch.__version__),'numpy':np.__version__,'files':[]}
 for path,key in [(paths[0],'reference_dataset'),(output,'reconstructions')]:
  old=next(v['sha256'] for v in expected_hash if v['role']==key)
  status['files'].append({'role':key,'path':str(path.relative_to(R)),'sha256':sha(path),'historical_sha256':old,'byte_identical':sha(path)==old})
 example_checks=[]
 for example in sorted((O/'examples/inputs').glob('*.npz')):
  with np.load(example,allow_pickle=False) as z:
   e=int(z['record_ids'][0]);idx=desired.index(e);kind=str(z['input_kind'].item())
   recovered={'A':actual[idx,:,:1],'B':actual[idx],'C':reconstructed[idx]}[kind]
   diff=float(np.abs(recovered-z['signal']).max())
   assert diff <= (2e-5 if kind=='C' else 0), (example,diff)
   example_checks.append({'file':example.name,'ecg_id':e,'max_abs_difference_mV':diff})
 status['historical_example_checks']=example_checks
 (R/'reports/recovery_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))
if __name__=='__main__':main()
