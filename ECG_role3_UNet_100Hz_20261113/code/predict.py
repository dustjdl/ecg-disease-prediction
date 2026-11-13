"""Reconstruct 11 leads from 100Hz Lead I using a completed checkpoint."""
import argparse
from pathlib import Path
import numpy as np
import torch
from model import ECGModel

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--npz',required=True,type=Path);ap.add_argument('--checkpoint',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--split',choices=['train','validation'],default='validation');args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4)
    with np.load(args.npz,allow_pickle=False) as z:
        assert int(z['sampling_rate_hz'])==100 and str(z['voltage_unit'])=='mV'
        mask=z['split']==args.split;ids=z['ecg_id'][mask];x=z['lead_i'][mask].copy()
    assert x.shape[1:]==(1000,1) and np.isfinite(x).all() and len(x)>0
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    assert checkpoint['experiment']=='unet_100hz_matched'
    m=ECGModel();m.load_state_dict(checkpoint['model_state_dict']);m.eval()
    tx=torch.from_numpy(x.transpose(0,2,1).copy()).float()
    with torch.no_grad():pred=torch.cat([m(b) for b in tx.split(8)]).numpy().transpose(0,2,1)
    assert np.isfinite(pred).all()
    # Keep the measured input unchanged; metrics must exclude this copied lead.
    full=np.concatenate([x,pred],axis=2)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:
        np.savez_compressed(f,ecg_id=ids,predicted_11lead=pred,reconstructed_12lead=full,lead_names=np.array(['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']),sampling_rate_hz=np.array(100),voltage_unit=np.array('mV'))
    print('Saved',args.output)
if __name__=='__main__':main()
