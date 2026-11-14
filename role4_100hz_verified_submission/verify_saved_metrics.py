"""Independently aggregate saved per-record, per-lead validation metrics."""
import argparse
import io
import zipfile
import numpy as np
import pandas as pd

BASE = 'ecg-disease-prediction-main/ECG_role3_UNet_100Hz_20261113/results/'
LEADS = {'II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('zip_path')
    parser.add_argument('--output', default='independent_seed_metrics.csv')
    args = parser.parse_args()
    rows = []
    keys_by_model = {}
    with zipfile.ZipFile(args.zip_path) as archive:
        for model in ('cnn_control', 'unet'):
            path = BASE + model + '/validation_record_lead_metrics.csv'
            df = pd.read_csv(io.BytesIO(archive.read(path)))
            if set(df['lead']) != LEADS or set(df['seed']) != {42,43,44}:
                raise ValueError(f'Unexpected leads/seeds: {model}')
            if df.duplicated(['seed','ecg_id','lead']).any():
                raise ValueError(f'Duplicate metric keys: {model}')
            keys_by_model[model] = set(map(tuple, df[['seed','ecg_id','lead']].itertuples(index=False,name=None)))
            if not np.isfinite(df[['mse_mV2','mae_mV','rmse_mV']].to_numpy()).all():
                raise ValueError(f'Non-finite metric: {model}')
            for seed, group in df.groupby('seed'):
                if len(group) != 1100 or group.ecg_id.nunique() != 100:
                    raise ValueError(f'Unexpected cohort size: {model}, seed {seed}')
                if not (group.groupby('ecg_id')['lead'].nunique() == 11).all():
                    raise ValueError(f'Incomplete leads: {model}, seed {seed}')
                sq_error = float(np.max(np.abs(group.rmse_mV.to_numpy()**2-group.mse_mV2.to_numpy())))
                if sq_error > 1e-10:
                    raise ValueError(f'RMSE inconsistent: {model}, seed {seed}')
                rows.append({'model':model,'seed':int(seed),'n_ecg':100,'n_rows':1100,
                             'leads_ok':True,'duplicate_rows':0,'all_11_leads':True,
                             'rmse_sq_error':sq_error,'mae':float(group.mae_mV.mean()),
                             'mse':float(group.mse_mV2.mean()),
                             'pooled_rmse':float(np.sqrt(group.mse_mV2.mean()))})
    if keys_by_model['cnn_control'] != keys_by_model['unet']:
        raise ValueError('CNN/U-Net evaluation keys do not match')
    result = pd.DataFrame(rows)
    result.to_csv(args.output,index=False)
    print(result.to_string(index=False))
    print('PASS: same ECG IDs and lead keys across models')

if __name__ == '__main__':
    main()
