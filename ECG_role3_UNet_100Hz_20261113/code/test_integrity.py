"""Real-data provenance and meaningful shape/gradient/metric checks; no test inference."""
import hashlib,json,sys,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model import ECGModel
from compare_results import metrics,load_predictions
ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(2)
    checks={}
    src=ROOT/'reference/cnn_baseline'
    cfg=json.loads((src/'experiment_config.json').read_text())
    packaged=json.loads((src/'verification.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(ROOT/'data/ecg_actual_100hz_998.npz')==packaged['packaged_npz_sha256']
    assert sha(ROOT/'labels/ptbxl_100hz_v3_labels_998.csv')==cfg['labels_sha256']
    assert sha(ROOT/'reference/model.py')==cfg['model_sha256']
    assert sha(ROOT/'reference/run_baseline_original.py')==cfg['script_sha256']
    checks['packaged_data_labels_and_original_cnn_code_hashes_match']=True
    with np.load(ROOT/'data/ecg_actual_100hz_998.npz',allow_pickle=False) as d:
        ids=d['ecg_id'];patients=d['patient_id'];split=d['split'];leads=d['lead_names'][1:]
        assert len(set(ids))==998 and d['actual_12lead'].shape==(998,1000,12)
        assert int(d['sampling_rate_hz'])==100 and str(d['voltage_unit'])=='mV'
        assert np.isfinite(d['actual_12lead']).all()
        assert np.array_equal(d['lead_i'],d['actual_12lead'][:,:,:1])
        for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not set(patients[split==a])&set(patients[split==b])
        assert [sum(split==s) for s in ['train','validation','test']]==[798,100,100]
        vx=torch.from_numpy(d['lead_i'][split=='validation'].transpose(0,2,1).copy())
    checks['patient_split_disjoint_and_actual_targets_valid']=True
    torch.manual_seed(20261113);m=ECGModel();x=torch.randn(2,1,1000,requires_grad=True);y=m(x)
    assert y.shape==(2,11,1000);y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    for shape in [(2,1,5000),(2,2,1000),(2,1,999)]:
        try:m(torch.zeros(shape))
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    checks['forward_backward_and_input_guard']=True
    checks['unet_parameters']=sum(p.numel() for p in m.parameters())
    # Errors [1,-3] give MSE=5, MAE=2, pooled RMSE=sqrt(5).
    mse,mae,rmse=metrics(np.zeros((1,2,1)),np.array([[[1.],[-3.]]]))
    np.testing.assert_allclose([mse.item(),mae.item(),rmse.item()],[5,2,np.sqrt(5)])
    checks['metric_known_answer']=True
    spec=importlib.util.spec_from_file_location('cnn_reference',ROOT/'reference/model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    cnn=mod.ECGModel();checks['cnn_parameters']=sum(p.numel() for p in cnn.parameters())
    control=ROOT/'results/cnn_control';repro=[]
    for seed in [42,43,44]:
        archived=load_predictions(src/f'seed_{seed}',ids[split=='validation'],leads)
        cnn.load_state_dict(torch.load(src/f'seed_{seed}/best_model.pth',weights_only=True,map_location='cpu')['model_state_dict']);cnn.eval()
        with torch.no_grad():new=torch.cat([cnn(x) for x in vx.split(8)]).numpy().transpose(0,2,1)
        np.testing.assert_allclose(new,archived,rtol=1e-5,atol=1e-6)
        rerun=load_predictions(control/f'seed_{seed}',ids[split=='validation'],leads)
        np.testing.assert_allclose(rerun,archived,rtol=1e-5,atol=1e-6)
        h1=pd.read_csv(src/f'seed_{seed}/training_history.csv');h2=pd.read_csv(control/f'seed_{seed}/training_history.csv')
        np.testing.assert_allclose(h1,h2,rtol=1e-6,atol=1e-8)
        repro.append({'seed':seed,'archived_checkpoint_forward_max_abs_diff':float(np.max(np.abs(new-archived))),'retrained_prediction_max_abs_diff':float(np.max(np.abs(rerun-archived))),'training_history_max_abs_diff':float(np.max(np.abs(h1.to_numpy()-h2.to_numpy())))})
    checks['cnn_reproduction']=repro;checks['test_evaluated']=False;checks['status']='passed'
    (ROOT/'results/integrity_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
