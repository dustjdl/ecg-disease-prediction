# 체크포인트 별도 전달

모델 가중치는 `.gitignore`의 `/checkpoints/`, `*.pth` 규칙으로 Git에서 제외한다. 팀이 정한 접근 권한이 있는 파일 공유 채널로 전달한다. 이 작업에서는 업로드·파일 전송을 수행하지 않았다.

## 전달 대상과 검증값

아래는 최종 test 평가에 실제 사용한 여섯 best-validation 체크포인트다. 파일 이름·원래 run_id 경로를 유지한다. 경로는 프로젝트 루트 기준이며 SHA-256은 저장된 평가 메타데이터를 근거로 한다.

| seed | 모델 | best epoch | 복원할 경로 | SHA-256 |
|---:|---|---:|---|---|
| 42 | Baseline | 16 | `checkpoints/baseline_1000/run_20261010_230450_b4bed3d513ff42ddba68f2d7a6647034/best_model.pth` | `4d601e90c2225ca18f8a9a6e2f25ce5273c4e52bd9337e998bf872db4319de79` |
| 42 | Dilated | 28 | `checkpoints/dilated_1000/run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636/best_model.pth` | `e6cfd066ab8c88f1e37d5635beff7df2c3606509d0e1455392dab6642d8c6a64` |
| 43 | Baseline | 22 | `checkpoints/baseline_multiseed/run_20261011_002652_a8cd2ab69a3d479db661a33430b4ecf9/best_model.pth` | `966e50bd07ace25bf987104915d3197f6620c947470e03cd2235406975b97fd2` |
| 43 | Dilated | 23 | `checkpoints/dilated_multiseed/run_20261011_002749_0d998b6396074cfe8f18ad6df4d0cb5e/best_model.pth` | `f114c133ba7a13749676d109f413d807fac17175741aafa4854fc494f7620b87` |
| 44 | Baseline | 24 | `checkpoints/baseline_multiseed/run_20261011_003041_c5722aac03554995923061bf4806a705/best_model.pth` | `484f89817affb57c96d93784b3e0847981fcf6b1b7e4220677615c968550df4f` |
| 44 | Dilated | 24 | `checkpoints/dilated_multiseed/run_20261011_003125_fcf2094d5c794ae7a87618edaa16aafc/best_model.pth` | `e7eba253914d9b697e4bf316995a11f71d5fd8a964c7beaf058a0ec72d78e734` |

받은 파일은 다음처럼 해시를 확인한다.

```powershell
Get-FileHash -Algorithm SHA256 .\checkpoints\dilated_1000\run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636\best_model.pth
```

평가 프로그램은 state_dict shape뿐 아니라 모델 클래스, dilation/padding, 전처리, Lead 순서, 주파수, 데이터 해시·ID 순서와 소스 해시를 검사한다. 후보는 같은 seed Baseline의 파일 해시·run_id를 참조한다. 파일을 재저장하거나 체크포인트의 메타데이터를 수정하지 않는다.

## 함께 필요한 항목

- 여섯 `best_model.pth`와 이 표의 해시.
- 이번 Git 공유본의 **바이트가 유지된 Python 소스**. `.gitattributes`의 Python `-text` 규칙을 유지하며 자동 포맷을 적용하지 않는다.
- 최종 test 평가에는 원본 NPZ `data/processed/ptbxl_100hz_1000records.npz`가 필요하다. SHA-256은 `2828f1949a6c5205de243cf304afc47faabc52d55021c7b9120f19d6309662d6`이다. 동일 내용으로 재생성한 다른 바이트의 파일은 기존 해시 검증을 통과하지 못할 수 있다.
- 각 run의 `results/<실험그룹>/<run_id>/training_history.csv`. 최종 평가 코드가 선택 epoch를 이력의 최저 validation loss와 대조한다. 공유 폴더의 동일 경로 앞에 `reports/representative/`를 붙이면 이력 사본을 찾을 수 있다. 필요하면 기존 파일이 없는지 확인한 뒤 원래 로컬 경로에 복사한다. 기존 파일은 덮어쓰지 않는다.
- 단일 `predict_ecg.py` 추론에는 사용할 체크포인트와 사용자가 준비한 Lead I NPY만 필요하며 원본 전체 데이터나 학습 이력은 필요하지 않다.

공유 결과 JSON은 설명을 위한 사본이며 일부 로컬 절대 경로를 바꿨다. 실제 로더에는 원본 체크포인트를 사용한다. 원본·사본 파일 해시는 [MANIFEST.json](../reports/representative/MANIFEST.json)에 있다. 대표 결과는 Git에 포함하지만 원본 파형 NPZ·NPY와 가중치는 별도로 전달한다.

## 최종 test 평가 명령

아래 명령은 실제 test 추론을 수행한다. 여섯 체크포인트·원본 NPZ·대응 학습 이력을 준비한 뒤 필요할 때만 실행한다. 검증만 하려면 마지막에 `--validate-only`를 추가한다. 같은 test 100개를 세 seed로 평가하며 test로 모델·seed·체크포인트를 다시 선택하지 않는다. 기존 seed 42 Baseline test 결과를 이미 확인했다는 사실도 기록한다.

```powershell
.\.venv\Scripts\python.exe -B .\evaluate_test_multiseed.py `
  --baseline-checkpoint-42 .\checkpoints\baseline_1000\run_20261010_230450_b4bed3d513ff42ddba68f2d7a6647034\best_model.pth `
  --candidate-checkpoint-42 .\checkpoints\dilated_1000\run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636\best_model.pth `
  --baseline-checkpoint-43 .\checkpoints\baseline_multiseed\run_20261011_002652_a8cd2ab69a3d479db661a33430b4ecf9\best_model.pth `
  --candidate-checkpoint-43 .\checkpoints\dilated_multiseed\run_20261011_002749_0d998b6396074cfe8f18ad6df4d0cb5e\best_model.pth `
  --baseline-checkpoint-44 .\checkpoints\baseline_multiseed\run_20261011_003041_c5722aac03554995923061bf4806a705\best_model.pth `
  --candidate-checkpoint-44 .\checkpoints\dilated_multiseed\run_20261011_003125_fcf2094d5c794ae7a87618edaa16aafc\best_model.pth `
  --device cpu
```

출력은 `results/test_comparison_multiseed/<새 run_id>/`에 생성된다. 기존 결과를 덮어쓰지 않는다. 데이터 해시, 실행 환경, 원본 결과 근거는 [최종 test 공유 설정](../reports/representative/results/test_comparison_multiseed/run_20261011_010820_9c3607152b144c24b82e590f7c67f94a/evaluation_config.json)에 있다.
