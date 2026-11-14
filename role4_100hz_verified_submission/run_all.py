"""Run all three independent checks from one original GitHub ZIP."""
import argparse, subprocess, sys, tempfile, zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Role4 common validation of 100Hz CNN vs U-Net')
    p.add_argument('github_zip',help='Original ecg-disease-prediction-main ZIP')
    args=p.parse_args()
    here=Path(__file__).resolve().parent
    src=Path(args.github_zip).resolve()
    if not src.is_file(): raise FileNotFoundError(src)
    output=here/'output'; output.mkdir(exist_ok=True)
    def run(cmd):
        print('RUN:', ' '.join(map(str,cmd)),flush=True)
        subprocess.run([str(x) for x in cmd],check=True,cwd=here)
    run([sys.executable,here/'verify_saved_metrics.py',src,'--output',here/'independent_seed_metrics.csv'])
    run([sys.executable,here/'verify_cohort.py',src])
    # Detailed verifier expects a local source tree. Extract only the files needed by its checks.
    prefixes=[
      'ecg-disease-prediction-main/ECG_role2_100Hz_v3_baseline_20261111/role2_100hz_v3/labels/ptbxl_100hz_v3_labels_998.csv',
      'ecg-disease-prediction-main/ECG_role2_100Hz_v3_baseline_20261111/role2_100hz_v3/results/baseline/',
      'ecg-disease-prediction-main/ECG_role3_UNet_100Hz_20261113/results/unet/',
      'ecg-disease-prediction-main/ECG_role3_UNet_100Hz_20261113/results/cnn_control/'
    ]
    with tempfile.TemporaryDirectory() as tmp:
        # Isolate temporary extracted inputs from submitted outputs.
        source=here/'source'
        if source.exists(): raise RuntimeError('source/ already exists; remove it before running')
        try:
            with zipfile.ZipFile(src) as z:
                for item in z.infolist():
                    name=item.filename
                    if not name.endswith('.csv'): continue
                    if not any(name==prefix or name.startswith(prefix) for prefix in prefixes): continue
                    dest=source/name
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_bytes(z.read(item))
            # Detailed verifier assumes source/ecg-disease-prediction-main.
            run([sys.executable,here/'verify_detailed.py'])
        finally:
            import shutil
            if source.exists(): shutil.rmtree(source)
    print('PASS: All three saved-result audits completed. No raw-waveform inference or held-out test evaluation was performed.')
if __name__=='__main__': main()
