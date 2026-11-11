from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'results'/'baseline'; figs=out/'figures'; figs.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
f,ax=plt.subplots(figsize=(8,4))
for s in [42,43,44]:
 h=pd.read_csv(out/f'seed_{s}'/'training_history.csv'); ax.plot(h.epoch,h.validation_mse,label=f'Validation, seed {s}')
ax.set(xlabel='Epoch',ylabel='MSE (mV²)',title='100 Hz basic CNN: validation learning curves');ax.legend();f.tight_layout();f.savefig(figs/'validation_learning_curves.png',dpi=180);plt.close(f)
m=pd.read_csv(out/'validation_metrics_by_seed.csv'); m=m[m.lead!='ALL_11']; order=['II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']; a=m.groupby('lead').mae_mV.agg(['mean','std']).loc[order]
f,ax=plt.subplots(figsize=(8,4));ax.bar(order,a['mean'],yerr=a['std'],capsize=3,color='#3569b8');ax.set(ylabel='MAE (mV)',title='Validation lead error: mean ± SD across 3 seeds');f.tight_layout();f.savefig(figs/'validation_lead_mae.png',dpi=180);plt.close(f)
g=pd.read_csv(out/'validation_groups_by_seed.csv');g=g[g.lead=='ALL_11'];order=['NORM','MI','STTC','CD','HYP','UNCLASSIFIED'];a=g.groupby('group').mae_mV.agg(['mean','std']).loc[order]
f,ax=plt.subplots(figsize=(8,4));ax.bar(order,a['mean'],yerr=a['std'],capsize=3,color='#38867c');ax.set(ylabel='MAE (mV)',title='Validation v3 diagnosis groups (overlapping groups)');f.tight_layout();f.savefig(figs/'validation_group_mae.png',dpi=180);plt.close(f)
