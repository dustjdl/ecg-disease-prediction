"""Export matched Lead I, real and reconstructed ECGs using the original model loader."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from predict_ecg import model_from_checkpoint


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _, _ = model_from_checkpoint(checkpoint)
    model.eval()
    with np.load(args.dataset, allow_pickle=False) as data:
        keep = ~((data['split'] == 'train') & np.isin(data['ecg_id'], [421, 710]))
        x, y = data['X'][keep], data['Y'][keep]
        if x.shape != (998, 1000, 1) or y.shape != (998, 1000, 11):
            raise ValueError('Expected the original 1000-record dataset, excluding train ECGs 421 and 710.')
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(x), 16):
                batch = torch.from_numpy(x[start:start + 16]).transpose(1, 2)
                predictions.append(model(batch).transpose(1, 2).numpy())
                print(f'Progress: {min(start + 16, len(x))}/{len(x)}', flush=True)
        pred = np.concatenate(predictions)
        assert pred.shape == y.shape and np.isfinite(pred).all()
        split, patient = data['split'][keep], data['patient_id'][keep]
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            assert set(patient[split == a]).isdisjoint(patient[split == b])
        output = args.output / 'ecg_for_classification.npz'
        np.savez_compressed(output, lead_i=x, actual_12lead=np.concatenate([x, y], axis=2),
                            reconstructed_12lead=np.concatenate([x, pred], axis=2),
                            ecg_id=data['ecg_id'][keep], patient_id=patient, split=split,
                            lead_names=data['lead_names'], sampling_rate_hz=data['sampling_rate_hz'],
                            voltage_unit=np.array('mV'))
    info = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'checkpoint_sha256': sha256(args.checkpoint), 'dataset_sha256': sha256(args.dataset),
            'output_sha256': sha256(output), 'torch_version': torch.__version__,
            'numpy_version': np.__version__, 'device': 'cpu', 'batch_size': 16,
            'checkpoint_epoch': checkpoint['epoch'], 'excluded_train_ecg_ids': [421, 710]}
    (args.output / 'provenance.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'Saved: {output.resolve()}')


if __name__ == '__main__':
    main()
