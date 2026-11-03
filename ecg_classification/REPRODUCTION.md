# 최종 전달본 기준 실행 방법

모든 상대 경로는 **이 문서가 있는 압축 해제 폴더**를 기준으로 합니다. `Set-Location`으로 그 폴더에 이동한 뒤 실행하세요. 시스템 Python 대신 기존 가상환경 Python의 절대 경로를 사용해도 됩니다.

## A. 독립 추론

목적은 이미 학습한 모델로 새 입력의 확률·라벨을 얻는 것입니다. ZIP의 `classifier_inference.py`, `models/`, 입력 NPZ만 필요합니다. 학습 데이터·정답 CSV·scikit-learn은 필요하지 않습니다.

필요 패키지는 `requirements-inference.txt`의 NumPy/PyTorch입니다. 새 환경 설치 자체는 이번 수정에서 검증하지 않았습니다.

```powershell
python .\package_paths.py --mode inference
python .\classifier_inference.py --model-dir .\models\multilabel\C\seed_42 --input .\examples\inputs\C_test_9.npz --output .\my_outputs\C_test_9.json --device cpu
```

기존 `examples/outputs/`는 검토용 실행 예시이므로 새 출력 경로를 사용합니다. 이진 모델은 `models/binary/C/seed_42`처럼 task만 바꿉니다. 입력 A/B/C는 모델과 일치해야 합니다. 출력에는 고정 validation 임계값과 미보정 sigmoid 점수, 예측 라벨이 포함됩니다.

## B. 입력 감사와 저장된 전체 결과의 재검증

재학습이나 새로운 test 추론 없이 저장된 18개 모델의 체크포인트 선택·임계값·정규화·기존 확률·성능 계산을 검증합니다.

먼저 원본 파일을 다음 위치에 별도로 복사합니다. 파일 내용은 `DATA_REQUIREMENTS.json`의 SHA-256과 같아야 합니다.

| 위치 | 역할 | ZIP 포함 |
|---|---|---|
| `data/received/ecg_labels_analysis_998.csv` | 정답 라벨과 ID·split·환자 정보 | 아니요 |
| `data/received/ecg_for_classification.npz` | Lead I, 실제·재구성 12유도, ID·split | 아니요 |
| `data/processed/ptbxl_100hz_1000records.npz` | 실제 파형·원래 split·sampling rate 대조 | 아니요 |
| `results/dilated_1000/run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636/experiment_config.json` | 기존 798/100/100 ID와 원본 NPZ hash | 예 |
| `checkpoints/*/*/best_model.pth` | 기존 재구성 모델 6개의 출처 대조 | 예 |

전체 분석에는 `requirements-classification.txt`의 기존 프로젝트 패키지와 scikit-learn이 필요합니다.

```powershell
python .\package_paths.py --mode full
python .\train_ecg_classification.py --phase audit --output .\reproduced_runs\audit_01
python .\verify_ecg_classification.py --report-dir .\verification\recheck_01
python .\summarize_ecg_classification.py --output .\derived_reports\summary_01
```

- 감사 명령은 데이터 결합·원본 대조와 기존 재구성 checkpoint 추론 비교만 하며 분류 모델을 학습하지 않습니다.
- 검증 코드의 `--experiment` 기본값은 `results/classification_abc/experiment_20261101/`입니다. 그 안의 `models/`에 모델별 `predictions.npz`, 이력, 고정 임계값이 모두 포함되어 있습니다.
- 검증 코드의 두 JSON 출력은 **새 `--report-dir`에만** 기록됩니다. 기존 `verification.json`이나 원본 결과는 덮어쓰지 않습니다.
- 요약 코드는 저장된 CSV를 읽고 **새 `--output`에만** 표·그림·REPORT.md를 만듭니다. 루트 최종 요약이나 기존 그래프를 수정하지 않습니다.
- 위 예시 폴더가 이미 있으면 다른 이름을 지정해야 합니다.

## C. 전체 18개 모델 재학습·평가

사용자가 새 실험을 실행할 때만 아래 명령을 사용합니다. 이번 ZIP 수정 과정에서는 이 재학습을 수행하지 않았습니다.

```powershell
python .\package_paths.py --mode full
python -u .\train_ecg_classification.py --output .\reproduced_runs\classification_01
python .\verify_ecg_classification.py --experiment .\reproduced_runs\classification_01 --report-dir .\verification\classification_01
python .\summarize_ecg_classification.py --experiment .\reproduced_runs\classification_01 --output .\derived_reports\classification_01
```

학습은 동일한 라벨 기준, seed 42/43/44, 40 epoch, 2 task×3입력으로 수행됩니다. train 정규화·클래스 가중치, validation checkpoint·임계값 선택, 모든 선택 고정 후 test 평가 순서를 유지합니다. 원래 split은 798/100/100이고 미분류 마스크 후 loss·지표 분모는 796/96/98입니다.

학습 코드의 기본 출력 경로는 포함된 원래 실험 경로입니다. 그 폴더가 이미 존재하므로 **`--output`을 생략하면 덮어쓰지 않고 중단**합니다. 반드시 위처럼 새 경로를 지정하세요. 학습 코드는 실행 당시 source hash와 기존 검증의 일관성을 위해 그대로 보존했습니다. 수정된 검증·요약 코드의 이전 버전은 `historical_source/`에 보관했습니다.

다른 Python/PyTorch/CUDA/CPU 환경에서는 같은 seed라도 bitwise 가중치·확률 재현을 보증하지 않습니다. 현재 가중치를 이용한 추론의 재현 확인과, 새 최적화 과정을 수행하는 전체 실험 재현은 구분해야 합니다. 전달 NPZ의 생성 명령·전체 역사적 로그는 미확인이라는 기존 한계도 유지됩니다.

## 경로와 보호 범위

`results/` 바로 아래의 CSV와 `models/`는 요약 링크·독립 추론 호환용입니다. 전체 학습·검증의 기준 실험은 `results/classification_abc/experiment_20261101/`입니다. 두 위치의 모델·지표는 같은 원본의 사본입니다.

기존 가중치, 임계값, config, 학습 이력, 확률 NPZ, 성능 CSV·PNG·JSON은 변경하지 않았습니다. 수정 전 ZIP도 보존했습니다. `MANIFEST.json`은 현재 전달 파일 전체를 열거하며, 포함된 과거 실험 manifest는 그 당시 산출물의 감사 기록입니다.
