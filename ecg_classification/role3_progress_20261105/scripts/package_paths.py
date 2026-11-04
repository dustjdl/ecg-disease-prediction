"""Check unpacked package paths/hashes; never reads host-specific original paths."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT/'results/classification_abc/experiment_20261101'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def check(mode):
    expected=[]
    for task in ['multilabel','binary']:
        for inp in ['A','B','C']:
            for seed in [42,43,44]:
                model=ROOT/f'models/{task}/{inp}/seed_{seed}'
                for name in ['best_model.pth','thresholds.json','selection.json']:
                    expected.append((model/name,None))
    if mode=='full':
        requirements=json.loads((ROOT/'DATA_REQUIREMENTS.json').read_text(encoding='utf-8'))
        expected.extend((ROOT/r['relative_path'],r['sha256']) for r in requirements['data_files'])
        frozen=json.loads((EXPERIMENT/'selection_frozen_before_test.json').read_text(encoding='utf-8'))
        for row in frozen['models']:
            model=EXPERIMENT/row['path'].replace('\\','/')
            expected.extend((model/name,None) for name in ['config.json','predictions.npz','history.csv','train_order_digests.csv'])
        expected.extend((EXPERIMENT/name,None) for name in ['experiment_config.json','metrics_by_seed.csv','predictions_long.csv','completion.json'])
        expected.append((ROOT/'train_ecg_classification.py',json.loads((EXPERIMENT/'experiment_config.json').read_text(encoding='utf-8'))['code_sha256']))
        expected.append((ROOT/'results/dilated_1000/run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636/experiment_config.json',None))
        expected.extend((ROOT/r['checkpoint'].replace('\\','/'),r['checkpoint_sha256']) for r in json.loads((EXPERIMENT/'reconstruction_provenance.json').read_text(encoding='utf-8'))['numerically_consistent_checkpoints'])
    missing=[];mismatches=[]
    for path,sha in expected:
        if not path.is_file():missing.append(str(path.relative_to(ROOT)))
        elif sha is not None and digest(path)!=sha:mismatches.append(str(path.relative_to(ROOT)))
    return {'mode':mode,'status':'passed' if not missing and not mismatches else 'blocked',
            'package_root':str(ROOT),'missing_files':missing,'hash_mismatches':mismatches,
            'data_manifest':'DATA_REQUIREMENTS.json','checked_paths':len(expected)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['inference','full'],default='inference')
    args=p.parse_args()
    result=check(args.mode)
    print(json.dumps(result,ensure_ascii=True,indent=2))
    raise SystemExit(0 if result['status']=='passed' else 2)


if __name__=='__main__':main()
