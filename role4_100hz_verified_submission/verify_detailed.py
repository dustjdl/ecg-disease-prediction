from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).parent
src=ROOT/'source/ecg-disease-prediction-main'
cnn=src/'ECG_role2_100Hz_v3_baseline_20261111/role2_100hz_v3/results/baseline'
unet=src/'ECG_role3_UNet_100Hz_20261113/results/unet'
control=src/'ECG_role3_UNet_100Hz_20261113/results/cnn_control'
out=ROOT/'output';out.mkdir(exist_ok=True)
models={'CNN':cnn,'U-Net':unet}
frames={m:pd.read_csv(p/'validation_record_lead_metrics.csv') for m,p in models.items()}
assert all(len(d)==3300 and not d.duplicated(['seed','ecg_id','lead']).any() for d in frames.values())
assert set(map(tuple,frames['CNN'][['seed','ecg_id','lead']].to_numpy()))==set(map(tuple,frames['U-Net'][['seed','ecg_id','lead']].to_numpy()))
assert all(not (d.lead=='I').any() for d in frames.values())
for m,d in frames.items():
 assert np.allclose(d.rmse_mV**2,d.mse_mV2,atol=1e-12)
 assert (d.mae_mV>=0).all() and (d.mse_mV2>=0).all()
 assert (d.mae_mV**2<=d.mse_mV2+1e-12).all()
cohorts=[pd.read_csv(p/'cohort.csv').sort_values('ecg_id').reset_index(drop=True) for p in [cnn,unet,control]]
assert cohorts[0].equals(cohorts[1]) and cohorts[0].equals(cohorts[2])
assert cohorts[0].groupby('patient_id').split.nunique().max()==1
assert set(frames['CNN'].ecg_id)==set(cohorts[0].query("split=='validation'").ecg_id)
# independently aggregate per seed and lead
leads=[]
for m,d in frames.items():
 for (seed,lead),g in d.groupby(['seed','lead']):
  leads.append({'model':m,'seed':seed,'lead':lead,'n_ecg':len(g),'mae_mV':g.mae_mV.mean(),'mse_mV2':g.mse_mV2.mean(),'pooled_rmse_mV':np.sqrt(g.mse_mV2.mean())})
lead=pd.DataFrame(leads)
lead.to_csv(out/'lead_metrics_by_seed_recomputed.csv',index=False)
summary=lead.groupby(['model','lead'],as_index=False)[['mae_mV','mse_mV2','pooled_rmse_mV']].mean()
wide=summary.pivot(index='lead',columns='model',values=['mae_mV','mse_mV2','pooled_rmse_mV']);wide.columns=[f'{a}_{b}' for a,b in wide.columns];wide=wide.reset_index()
wide['MAE_CNN_minus_UNet']=wide['mae_mV_CNN']-wide['mae_mV_U-Net'];wide['MAE_winner']=np.where(wide.MAE_CNN_minus_UNet<0,'CNN','U-Net')
wide.to_csv(out/'lead_comparison_independent.csv',index=False)
# validate against each supplied lead summary per seed
checks=[]
for m,p in models.items():
 supplied=pd.read_csv(p/'validation_metrics_by_seed.csv')
 print('columns metrics',m,supplied.columns.tolist())
 merged=lead[lead.model==m].merge(supplied,on=['seed','lead'],suffixes=('_calc','_supplied'),validate='one_to_one')
 for metric in ['mae_mV','mse_mV2','pooled_rmse_mV']:
  checks.append({'model':m,'table':'validation_metrics_by_seed','metric':metric,'max_abs_difference':float((merged[f'{metric}_calc']-merged[f'{metric}_supplied']).abs().max()),'matched_rows':len(merged)})
# Group aggregate: verify reported group aggregates against record-level and labels
# The group memberships are read from v3 labels; multi-label rows can belong to multiple groups.
labels=pd.read_csv(src/'ECG_role2_100Hz_v3_baseline_20261111/role2_100hz_v3/labels/ptbxl_100hz_v3_labels_998.csv')
print('label cols',labels.columns.tolist())
print('group rows',pd.read_csv(cnn/'validation_groups_by_seed.csv').groupby('group').n.first().to_dict())
# compare group tables between models, report summaries without inferring label membership
for m,p in models.items():
 d=pd.read_csv(p/'validation_groups_by_seed.csv')
 assert (d.n>0).all()
 assert np.allclose(d.pooled_rmse_mV**2,d.mse_mV2,atol=1e-12)
 d['model']=m
 if m=='CNN': groups=d
 else: groups=pd.concat([groups,d],ignore_index=True)
groups.to_csv(out/'disease_group_metrics_reported.csv',index=False)
g=groups[groups.lead=='ALL_11'].groupby(['model','group'],as_index=False)[['mae_mV','mse_mV2','pooled_rmse_mV']].mean()
gw=g.pivot(index='group',columns='model',values=['mae_mV','mse_mV2']);gw.columns=[f'{a}_{b}' for a,b in gw.columns];gw=gw.reset_index();gw['MAE_winner']=np.where(gw['mae_mV_CNN']<gw['mae_mV_U-Net'],'CNN','U-Net');gw.to_csv(out/'disease_group_comparison.csv',index=False)
# verify group ALL_11 consistency with 11 lead rows: group aggregates based on same number of ECGs per group and seed
for (model,seed,group),d in groups.groupby(['model','seed','group']):
 allrow=d[d.lead=='ALL_11']
 if len(allrow)!=1:continue
 others=d[d.lead!='ALL_11']
 if len(others)!=11:continue
 for metric in ['mae_mV','mse_mV2']:
  checks.append({'model':model,'table':'validation_groups_by_seed ALL_11','metric':metric,'max_abs_difference':float(abs(allrow[metric].iloc[0]-others[metric].mean())),'matched_rows':11})
# independently recompute disease group metrics from v3 label membership and record-lead errors
membership=[]
for _,r in labels[labels.split=='validation'].iterrows():
 for group in ['NORM','MI','STTC','CD','HYP']:
  if r['target_v3_'+group]==1: membership.append((int(r.ecg_id),group))
 if r.v3_supervised_eligible==0: membership.append((int(r.ecg_id),'UNCLASSIFIED'))
member=pd.DataFrame(membership,columns=['ecg_id','group'])
ind=[]
for model,d in frames.items():
 joined=d.merge(member,on='ecg_id',validate='many_to_many')
 for (seed,group,lead_name),part in joined.groupby(['seed','group','lead']):
  ind.append({'model':model,'seed':seed,'group':group,'lead':lead_name,'n':len(part),'mae_mV':part.mae_mV.mean(),'mse_mV2':part.mse_mV2.mean(),'pooled_rmse_mV':np.sqrt(part.mse_mV2.mean())})
 for (seed,group),part in joined.groupby(['seed','group']):
  ind.append({'model':model,'seed':seed,'group':group,'lead':'ALL_11','n':part.ecg_id.nunique(),'mae_mV':part.mae_mV.mean(),'mse_mV2':part.mse_mV2.mean(),'pooled_rmse_mV':np.sqrt(part.mse_mV2.mean())})
ind=pd.DataFrame(ind);ind.to_csv(out/'disease_group_metrics_independent.csv',index=False)
merged=ind.merge(groups,on=['model','seed','group','lead'],suffixes=('_calc','_supplied'),validate='one_to_one')
assert len(merged)==len(groups)
assert (merged.n_calc==merged.n_supplied).all()
for metric in ['mae_mV','mse_mV2','pooled_rmse_mV']:
 checks.append({'model':'CNN and U-Net','table':'independent v3 disease group calculation','metric':metric,'max_abs_difference':float((merged[metric+'_calc']-merged[metric+'_supplied']).abs().max()),'matched_rows':len(merged)})
check=pd.DataFrame(checks);check.to_csv(out/'aggregate_consistency_checks.csv',index=False)
print('lead winners',wide[['lead','MAE_winner']].to_string(index=False));print('groups',gw.to_string(index=False));print('max diff',check.groupby(['table','metric']).max_abs_difference.max().to_string())
