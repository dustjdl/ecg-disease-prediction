# Lead I 기반 12유도 ECG 추정

PTB-XL의 단일 Lead I(100 Hz, 10초)으로 나머지 11개 유도를 추정하는 연구 프로젝트다. Baseline CNN과 중간 convolution의 dilation만 확대한 Dilated CNN을 비교한다. 출력 12유도 중 I은 관측한 입력이며 나머지는 모델의 추정값이다.

이 문서는 팀 저장소의 `ecg_reconstruction/` 공유본을 안내한다. 저장소 루트에서 `Set-Location .\ecg_reconstruction`으로 이동한 뒤 아래 명령을 실행한다. `data/`, `checkpoints/`, `examples/`, `results/`는 모두 이 폴더 기준이다. 원본 `ecg-capstone/`은 변경하지 않았다. 저장소 전체 복사 목록과 원본 대비 해시는 [파일 목록](../SHARE_FILES.txt), [복사 해시 목록](../SHARE_MANIFEST.json)에 있다.

- [실험 요약과 근거](EXPERIMENT_SUMMARY.md)
- [입출력 규격](ECG_INTERFACE.md)
- [대표 결과·파형](reports/representative/README.md)
- [체크포인트 전달 및 최종 평가 명령](docs/CHECKPOINT_HANDOFF.md)
- [공유 점검 결과](SHARING_REVIEW.md), [공유할 전체 파일 목록](SHARE_FILES.txt)

## 환경 설치

확인한 환경은 Windows 11, Python 3.14.8, NumPy 2.5.3, pandas 3.0.6, Matplotlib 3.11.2, WFDB 4.3.1, PyTorch 2.13.0+cu130이다. 학습은 CUDA, 최종 test 평가와 독립 추론 예제는 CPU에서 수행했다. 자세한 내용은 [ENVIRONMENT.md](ENVIRONMENT.md)를 참고한다.

팀 저장소의 `ecg_reconstruction/` 폴더에서 다음을 실행한다. 아래 명령은 안내이며 이번 공유 정리 과정에서는 환경 생성·설치를 실행하지 않았다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt`는 직접 사용하는 패키지의 버전을 고정하며 전이 의존성 전체의 lock은 아니다. PyTorch의 공개 버전 2.13.0을 지정하므로 CPU/CUDA 빌드는 설치 환경에 따라 달라질 수 있다. 새 컴퓨터에서의 설치와 bitwise 재현성은 아직 확인하지 않았다. 코드에서 지원하는 CPU 추론은 `--device cpu`로 지정한다.

기존 체크포인트는 Python 소스의 SHA-256을 검증한다. 기존 `.py` 파일을 자동 포맷하거나 줄바꿈을 변경하지 않는다. `.gitattributes`는 Python 파일의 Git 줄바꿈 변환을 막는다.

## 데이터 준비

원본 데이터와 가공 NPZ는 Git에 포함하지 않는다. 준비 스크립트는 PTB-XL v1.0.3의 metadata와 필요한 100 Hz WFDB 파일을 내려받거나 로컬 캐시를 재사용한다.

```powershell
.\.venv\Scripts\python.exe -B .\prepare_dataset_1000.py
.\.venv\Scripts\python.exe -B .\inspect_dataset_1000.py --seed 42
```

기본 NPZ는 `data/processed/ptbxl_100hz_1000records.npz`이며 X=(1000,1000,1), Y=(1000,1000,11), float32이다. 공식 fold 1~8/9/10에 따라 원본 분할은 train/validation/test=800/100/100이다. 수동 검토한 ECG 421·710을 train에서만 제외해 798/100/100을 사용한다. 검토 전압 기준 초과는 자동 삭제 기준이 아니며, 원본과 float32로 일치한 파형을 이번 실험 정책으로 제외했다.

기존 준비 결과를 바꾸는 `--overwrite`는 위 명령에 포함하지 않았다. 검사에서는 Y의 전압 범위가 수동 검토 기준을 넘었음을 보고할 수 있다. 근거는 실험 요약에 있다.

기존 체크포인트 평가에는 **바이트가 동일한 원본 NPZ**가 필요하다. 다시 생성한 NPZ는 파형이 같아도 파일 SHA-256이 다를 수 있다. 기존 데이터 해시는 다음과 같다.

```text
2828f1949a6c5205de243cf304afc47faabc52d55021c7b9120f19d6309662d6
```

```powershell
Get-FileHash -Algorithm SHA256 .\data\processed\ptbxl_100hz_1000records.npz
```

해시가 다르면 검증을 우회하지 않는다. 기존 실험을 평가하려면 데이터 보관 담당자에게 동일 NPZ를 별도 채널로 받아야 한다. 데이터 이용 조건은 팀에서 확인한 뒤 사용한다.

## 학습

학습 조건은 batch size 8, Adam lr=0.001, MSELoss, 30 epoch이며 validation MSE가 가장 낮은 체크포인트를 저장한다. 정규화·clipping·필터링·증강은 추가하지 않는다. 아래 명령은 필요할 때만 수동 실행하며 test 결과를 보고 추가 학습·모델 선택에 사용하지 않는다.

기존 seed 42 방식:

```powershell
.\.venv\Scripts\python.exe -B .\train_1000.py
```

이 실행이 생성한 Baseline 경로를 후보에 지정한다.

```powershell
$baseline42 = "checkpoints/baseline_1000/<생성된 run_id>/best_model.pth"
.\.venv\Scripts\python.exe -B .\train_dilated_1000.py --baseline-checkpoint $baseline42
```

다른 seed는 전용 스크립트를 사용한다.

```powershell
.\.venv\Scripts\python.exe -B .\train_baseline_multiseed.py --seed 43
$baseline43 = "checkpoints/baseline_multiseed/<위 실행에서 생성된 run_id>/best_model.pth"
.\.venv\Scripts\python.exe -B .\train_dilated_multiseed.py --seed 43 --baseline-checkpoint $baseline43
```

seed 44는 두 명령의 `--seed`를 모두 44로 바꾸고, 같은 seed의 새 Baseline 경로를 사용한다. 후보는 Baseline 학습 가중치를 복사하지 않는다. 학습 전 데이터·참조 검증만 하려면 해당 명령에 `--validate-only`를 추가한다. 각 실행은 새 run 폴더를 만들며 이전 실험 결과를 덮어쓰지 않는다.

## Validation 및 최종 test 평가

같은 seed 쌍을 validation 100개로 비교한다. 아래 변수에는 평가할 실제 경로를 지정한다.

```powershell
$baseline = "checkpoints/baseline_multiseed/<baseline run_id>/best_model.pth"
$candidate = "checkpoints/dilated_multiseed/<candidate run_id>/best_model.pth"
.\.venv\Scripts\python.exe -B .\compare_validation_multiseed.py --baseline-checkpoint $baseline --candidate-checkpoint $candidate --validate-only
.\.venv\Scripts\python.exe -B .\compare_validation_multiseed.py --baseline-checkpoint $baseline --candidate-checkpoint $candidate
```

기존 seed 42 쌍도 이 비교 스크립트가 지원한다. `compare_validation_1000.py`는 기존 seed 42 전용이다.

최종 test 평가는 `evaluate_test_multiseed.py`에 seed 42·43·44의 Baseline/Dilated 경로 여섯 개를 모두 명시한다. 실제 기존 경로가 포함된 명령은 [체크포인트 전달 안내](docs/CHECKPOINT_HANDOFF.md)에 있다. 먼저 `--validate-only`로 메타데이터를 확인할 수 있다. 이 옵션에서는 모델 forward와 test DataLoader 생성·순회를 하지 않는다.

기존 `evaluate_1000.py`는 기존 seed 42 Baseline 전용이며, 이번 여섯 모델의 float64 MSE/MAE/RMSE 평가에는 사용하지 않는다. 최종 평가 코드는 같은 test 100개를 원본 순서대로 평가하고, 예측·정답을 각각 float64로 변환한 뒤 pooled 지표를 계산한다. RMSE는 sqrt(MSE)이다.

## 단일 Lead I 추론

입력은 사용자가 준비한 `.npy` 파일이다. `examples/your_lead_i.npy`는 아래 명령의 자리 표시자이며 저장소에 포함된 파일이 아니다. 지원 shape는 (1000,) 또는 (1000,1), 단위는 mV, 주파수는 100 Hz이다.

```powershell
.\.venv\Scripts\python.exe -B .\predict_ecg.py `
  --checkpoint .\checkpoints\dilated_1000\run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636\best_model.pth `
  --input .\examples\your_lead_i.npy `
  --sampling-rate-hz 100 `
  --unit mV `
  --device cpu
```

모델은 메타데이터의 Baseline/Dilated 구조를 확인해 로드한다. 입력의 수치형·shape·유한값을 검사하고 float32로 변환한 뒤 다시 유한값을 확인한다. 자동 리샘플링·단위 환산·정규화·필터링·clipping을 하지 않으며 지원하지 않는 입력은 오류로 종료한다.

새 결과 폴더에 `signal=(1000,12)` NPZ, PNG, 메타데이터 JSON을 저장한다. 축은 (time, lead), 순서는 I, II, III, AVR, AVL, AVF, V1~V6이며 I만 입력이고 나머지 11개는 추정이다. 현재 저장된 출력의 [규격 요약](reports/representative/prediction_output_schema.json)을 확인할 수 있다. 원본 NPY/NPZ 예제는 공유 저장소에서 제외한다.

## 저장된 결과 요약

두 모델 모두 학습 파라미터 31,819개이며 수용영역은 Baseline 15 → Dilated 31 samples이다. 아래 감소율은 같은 seed의 `100 × (Baseline MSE − Dilated MSE) / Baseline MSE`이다.

| seed | validation MSE 감소율 | test MSE 감소율 |
|---:|---:|---:|
| 42 | 2.612512% | 6.240529% |
| 43 | 2.565625% | 3.211888% |
| 44 | 2.400024% | 4.485572% |

| 구간·모델 | seed 평균 MSE ± 표본 SD(ddof=1), mV² |
|---|---:|
| validation Baseline | 0.039374904 ± 0.000128297 |
| validation Dilated | 0.038380179 ± 0.000082463 |
| test Baseline | 0.037393183 ± 0.000526410 |
| test Dilated | 0.035651090 ± 0.000245562 |

세 seed는 같은 데이터 분할과 같은 test 100개를 반복 평가했다. 독립 test 300개가 아니다. test에서도 세 seed 모두 overall MSE·MAE·RMSE가 낮았지만 각 seed에서 MSE/RMSE가 악화된 Lead가 하나씩 있었다. 기존 seed 42 Baseline test 결과를 최종 비교 전에 이미 확인한 사실을 유지한다. test로 모델·seed·체크포인트를 다시 선택하지 않았다.

대표 결과는 [공유 폴더](reports/representative/README.md), 전체 수치와 한계는 [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)에 있다. 원본 결과 폴더는 로컬에 보존하고 공유 사본에서만 로컬 절대 경로를 정리했다.

## 한계와 팀 연결

현재 검증 범위는 PTB-XL이다. 실제 스마트워치 측정에 대한 검증과 분류 모델 연결은 미완료다. 동일 분할의 세 seed 결과로 통계적 유의성이나 임상적 유효성을 주장하지 않는다. validation으로 선택한 ECG 544·190·155는 결과를 보고 선택한 발표 사례다.

분류 담당자와 샘플링 주파수, 길이, 단위, Lead 순서, 배열 방향, dtype/batch 축, 전처리를 합의해야 한다. 조건을 임의로 가정해 변환하지 않는다.

## 공유 범위와 체크포인트 전달

Git 공유 대상은 Python 소스, 테스트, README·실험/인터페이스 문서, 의존성·공유 설정, `reports/representative/`의 선별 결과다. 데이터, 가상환경, 캐시, 모델 가중치, 생성 예제, 원본 results 전체와 중복 PNG는 `.gitignore`로 제외한다. 이미 추적된 파일에는 `.gitignore`가 소급 적용되지 않는다.

원본 공유 정리 이후 팀 저장소가 준비되어 이 모듈을 `ecg_reconstruction/`에 복사했다. 이후 HEAD 정상화와 GitHub 인증을 마친 뒤 원격 main의 기존 이력에 로컬 main을 연결했다. 원격 README·.gitignore와 공유 안내를 통합했으며, 충돌 파일과 공유 파일 전체는 연결 전에 저장소 밖에 백업했다. `git add`·commit·push는 수행하지 않았다. 이 폴더의 [SHARE_FILES.txt](SHARE_FILES.txt)는 원본 공유 목록 82개이고, 원격 파일까지 포함한 저장소 전체 목록은 [루트 SHARE_FILES.txt](../SHARE_FILES.txt)에 있다. 실제 Git 상태와 검사 범위는 [점검 문서](SHARING_REVIEW.md)에 기록했다.

체크포인트는 Git에 포함하지 않고 [별도 전달 안내](docs/CHECKPOINT_HANDOFF.md)에 따라 여섯 `best_model.pth`를 해시와 함께 전달한다. 기존 평가에는 동일 NPZ 및 대응하는 학습 이력도 필요하다.

초기 단일 샘플 시각화 사용법은 [기존 README 보관본](docs/README_FIRST_VISUALIZATION.md)에 보존했다.
