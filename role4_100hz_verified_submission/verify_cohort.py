"""Verify shared cohort and patient-level split isolation from the team ZIP."""
import argparse
import io
import zipfile
import pandas as pd
BASE = 'ecg-disease-prediction-main/ECG_role3_UNet_100Hz_20261113/results/'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('zip_path')
    args=parser.parse_args()
    with zipfile.ZipFile(args.zip_path) as z:
        cohorts={m:pd.read_csv(io.BytesIO(z.read(BASE+m+'/cohort.csv'))) for m in ('cnn_control','unet')}
        left=cohorts['cnn_control'].sort_values('ecg_id').reset_index(drop=True)
        right=cohorts['unet'].sort_values('ecg_id').reset_index(drop=True)
        assert left.equals(right), 'CNN/U-Net cohorts differ'
        assert not left.ecg_id.duplicated().any(), 'Duplicate ECG ID'
        assert left.groupby('patient_id').split.nunique().max()==1, 'Patient split leakage'
        assert left.split.value_counts().to_dict()=={'train':798,'validation':100,'test':100}, 'Unexpected split sizes'
        for model in cohorts:
            metrics=pd.read_csv(io.BytesIO(z.read(BASE+model+'/validation_record_lead_metrics.csv')))
            assert set(metrics.ecg_id)==set(left.loc[left.split=='validation','ecg_id']), f'{model} validation IDs mismatch'
        print('PASS: identical 998-row cohort, no duplicated ECG IDs, no patient-level split overlap')
        print('PASS: 798 train / 100 validation / 100 test; metrics IDs match validation split')
if __name__=='__main__': main()
