# 역할 3 질환 분류 모델 개발: A/B/C 실험

**최종 전달본 실행 안내:** 현재 작업 디렉터리는 압축 해제 폴더입니다. 독립 추론, 데이터 배치, 저장 결과 검증, 재학습의 실제 명령은 [REPRODUCTION.md](../REPRODUCTION.md)를 우선 따르세요. 기존 실험은 `results/classification_abc/experiment_20261101/`에 그대로 포함되어 있습니다. 검증은 새 `--report-dir`, 요약은 새 `--output`, 학습은 새 `--output`을 지정하며 기존 파일은 변경하지 않습니다.

분석 코드: [`train_ecg_classification.py`](../train_ecg_classification.py). 기존 재구성 분석·전달 폴더와 원본 데이터는 수정하지 않는다. 기존 모듈 `analyze_disease_reconstruction.py`의 ID·split·실제 파형 대조를 먼저 실행한다.

역할표 정정: 라벨 정책 자료의 확인 대상은 1번 담당이며, 이 문서의 라벨 기준은 **이번 실험에 적용한 라벨 기준**이다. 사용자는 추가 정책 문서를 전달받지 못했음을 확인했다. [라벨 기준과 제외 ID](CLASSIFICATION_LABEL_CRITERIA.md), [4번 연동 규격](CLASSIFIER_INTERFACE.md)을 함께 참고한다.

## 정책 확인과 라벨

로컬 저장소 `ecg-disease-prediction/docs`, `src`, `tests`에는 `.gitkeep`만 있고, README는 분류 연결 미완료로 기록한다. `ECG_INTERFACE.md`의 마지막 절도 분류 담당자와의 합의 항목이 미확정이라고 명시한다. 따라서 확인 가능한 확정된 1번 담당 라벨 정책이 없어 사용자 지정 기본값을 적용했다. 원격 저장소나 팀원 개인 자료까지 확인했다고 주장하지 않는다.

- 주 실험: NORM/MI/STTC/CD/HYP의 5개 sigmoid 출력과 다중 라벨 BCE. CSV `diagnostic_classes`와 5개 이진 열을 교차 확인한다. `include_in_analysis`를 라벨로 사용하지 않는다.
- 정상·질환 혼합 기록: 원래 주어진 양성 라벨을 모두 유지한다. NORM과 CD가 함께 있으면 두 라벨을 모두 1로 둔다. 이는 제공된 라벨을 따르는 정책이며 의학적 정상과 질환 공존을 새로 판정하는 것이 아니다.
- 미분류: 라벨 5개가 모두 0인 기록은 정상/전부 음성으로 간주하지 않고 지도학습·지표에서 제외한다. 원래 ecg_id·patient_id·split은 유지하며 예측은 저장하고 정답은 NA로 표시한다.
- 보조 이진 실험: MI/STTC/CD/HYP 중 하나라도 있으면 ABNORMAL=1, NORM만 있으면 0. 혼합 기록은 양성, 미분류는 제외한다.

원래 분할은 798/100/100 그대로다. 미분류 train 2개, validation 4개, test 2개를 지도학습·평가에서 마스킹하므로 실제 loss/지표의 분모는 **796/96/98**이다. 이는 재분할이 아니다. 모든 입력과 task에서 같은 마스크를 사용한다. 환자 ID의 split 간 교집합도 검사해 JSON/CSV로 기록한다.

기존 파형 분석은 NORM+질환을 별도 혼합군으로 두었으나 이번 다중 라벨 분류는 원래 라벨을 유지한다. 따라서 test CD 양성은 기존 질환 단독 정책의 16개가 아니라 혼합 CD 3개를 포함한 19개이고 NORM 양성은 정상 단독 47개+혼합 3개=50개다. 두 분석의 표본 수 차이를 오류나 새로운 split으로 해석하지 않는다.

## 고정한 비교 조건

| 입력 | 채널 | 내용 |
|---|---:|---|
| A | 1 | 실제 Lead I |
| B | 12 | 실제 12유도 |
| C | 12 | 실제 Lead I + 전달된 재구성 11유도 |

CNN은 Conv1d(C→32,k7,s2,p3)→ReLU→Conv1d(32→64,k5,s2,p2)→ReLU→Conv1d(64→128,k5,s2,p2)→ReLU→시간축 mean/max 결합→Linear(256→64)→ReLU→Dropout(0.2)→Linear(64→출력 수)다. A/B/C는 첫 convolution 입력 채널 수만 다르다. task에 따라 출력은 5 또는 1이다. 첫 계층과 나머지 계층을 별도 RNG 범위에서 초기화해 같은 task·seed의 더 깊은 가중치는 A/B/C에서 같고 B/C는 전체 초기값도 같다. 파라미터 수는 개별 config에 기록한다.

고정 seed 42·43·44, 각 40 epoch, batch 32, AdamW lr=0.001, weight_decay=0.0001, scheduler·early stopping·데이터 증강 없음이다. validation/test 결과를 보고 조건을 바꾸는 탐색을 수행하지 않는다. 같은 seed의 minibatch 순서가 A/B/C에서 일치하는지는 epoch별 ID digest로 검증한다.

각 입력의 **지도학습 대상 train 796개**에서 채널별 전체 시간 평균과 모집단 SD를 구해 z-score한다. SD≤1e-8인 채널은 분모를 1로 처리하고 기록한다. test·validation 값은 평균/SD 계산에 쓰지 않는다. 필터링, 신호 정렬, per-record 중심화는 하지 않는다.

각 task의 train 라벨로 `pos_weight = train_negative / train_positive`를 구해 BCEWithLogitsLoss에 적용한다. train에서 한쪽 클래스가 없으면 가중치를 1로 대체한다. 현재 train에는 모든 라벨의 양성·음성이 존재한다. 이 가중 손실의 sigmoid 출력은 보정된 임상적 확률을 보증하지 않는다.

체크포인트는 40 epoch 중 가중 validation BCE가 처음으로 가장 낮은 epoch다. 이후 해당 checkpoint의 validation 예측만으로 출력별 F1 최대 임계값을 0.05~0.95, 0.01 간격에서 고른다. 동률이면 0.5에 가장 가까운 값, 다시 동률이면 큰 값을 고른다. validation에 한쪽 클래스가 없으면 0.5로 고정하고 사유를 기록한다. 정답 test 라벨은 임계값 선택에 쓰지 않는다.

18개 모델(2 task×3입력×3seed)의 학습·체크포인트·validation 임계값 선택을 **모두 마친 뒤** `selection_frozen_before_test.json`에 해시를 고정한다. 이후 test를 포함한 예측과 지표를 한 번 계산한다. 검증 시 동일 저장 확률로 지표를 다시 계산하는 것은 새 설정 탐색이나 새 모델 test 추론이 아니다.

CUDA deterministic 알고리즘, 고정 cudnn 설정, CUBLAS_WORKSPACE_CONFIG를 사용한다. 다른 라이브러리·장치에서 bitwise 재현성을 보증하지 않는다. 같은 test 98개와 같은 재구성 NPZ를 3개 분류 seed로 반복한 결과이며 독립 환자군 294개가 아니다.

## 지표

라벨별 support(양성·음성), AUROC, AUPRC, precision, recall, F1, sensitivity, specificity, 보조 accuracy와 TN/FP/FN/TP를 저장한다. **AUPRC는 sklearn average_precision_score의 비보간 Average Precision(AP)**이며 사다리꼴 PR 면적과 다르다. 혼동행렬의 행은 실제 음성/양성, 열은 예측 음성/양성으로 `[[TN,FP],[FN,TP]]`다.

다중 라벨은 라벨별 지표의 동일 가중 macro와 모든 ECG×라벨을 펼친 micro를 저장한다. macro는 계산 가능한 라벨만 평균하고 유효 클래스 수·제외 사유를 기록한다. macro accuracy는 라벨별 이진 정확도의 평균이고 exact match인 subset_accuracy와 다르며 후자는 별도 열이다. 이진 분류는 abnormal-any를 양성으로 정의한 민감도·특이도·F1·AUROC·AP를 사용한다.

양성/음성이 모두 있어야 AUROC를 계산한다. 양성이 없으면 AP/recall, 예측 양성이 없으면 precision, 실제 음성이 없으면 specificity를 NA로 두고 이유를 남긴다. F1은 `2TP/(2TP+FP+FN)`이며 분모가 0일 때만 NA다. 예측이 모두 음성이지만 실제 양성이 있으면 F1=0이다. 결과 CSV에서 빈 값은 NA이며 0으로 임의 채우지 않는다.

각 지표의 seed 산술평균과 표본 SD(ddof=1)를 저장한다. A/B/C 차이는 같은 seed끼리 C−A, C−B, B−A를 계산하고 평균·SD를 저장한다. AUROC/AP/F1 등은 클수록 좋으므로 양의 차이가 해당 지표의 개선이다. 작은 표본의 기술적 비교로 통계적 유의성은 주장하지 않는다.

## 재구성 출처와 한계

전달 NPZ는 생성 스크립트·명령·체크포인트 ID를 담고 있지 않다. 로컬의 저장된 체크포인트 6개로 전체 998개 입력을 다시 추론해 NPZ와 수치 비교했다. `reconstruction_checkpoint_consistency.csv`와 `reconstruction_provenance.json`을 참고한다. 수치 일치는 정확한 생성 명령/역사적 실행 로그를 증명하지 않는다.

일치한 체크포인트의 저장된 train ID와 데이터 해시, 학습 코드의 train/validation 사용을 근거로 출처를 설명한다. validation은 재구성 checkpoint 선택에 이미 관여했다. 전체 생성·학습 이력의 독립 검증이 불가능한 부분은 미확인으로 표시한다.

C의 train 입력은 같은 train ECG로 학습한 재구성기의 in-sample 출력이며 out-of-fold 출력이 아니다. test의 독립적인 새 환자군 일반화를 보증하지 않는다. C는 Lead I의 결정적 변환이므로 독립적으로 측정한 11유도의 정보를 추가한 것은 아니다. B와의 차이는 실제 측정 정보, C와 A의 차이는 표현 방식·최적화·재구성 사전학습·소폭 파라미터 수 차이 등이 함께 영향을 줄 수 있다.

이전 파형 분석에서 test 재구성 결과와 라벨은 이미 살펴본 이력이 있다. 이번 설정은 test 분류 점수 확인 전에 고정했지만 완전 미관측 외부 test라고 표현하지 않는다. 적은 표본과 다중 질환 중복, 단일 분할, 기존 모델 선택의 validation 관여, 임계값의 validation 과적합 가능성을 결과에 명시한다. 분류 결과도 임상적 진단 유효성의 증명이 아니다.

## 실행

`ecg-capstone` 폴더에서 기존 가상환경을 사용한다. 추가로 사용하는 설치된 패키지는 scikit-learn 1.9.1이다. 기존 requirements를 변경하지 않고 분류용 추가 버전은 `requirements-classification.txt`에 기록한다. 새 컴퓨터 환경 설치는 이번 작업에서 수행하지 않았다.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_ecg_classification.py -v
.\.venv\Scripts\python.exe -u -B .\train_ecg_classification.py --output reproduced_runs/new_run
```

기본 출력은 `results/classification_abc/experiment_20261101/`이고 기존 폴더가 있으면 덮어쓰지 않고 중단한다. 재실행에는 새 경로를 지정한다.

```powershell
.\.venv\Scripts\python.exe -u -B .\train_ecg_classification.py --output results/classification_abc/new_run
```

입력 검증·재구성 출처 대조만 하고 분류 학습은 하지 않는 명령:

```powershell
.\.venv\Scripts\python.exe -u -B .\train_ecg_classification.py --phase audit --output results/classification_abc/new_audit
```

## 산출물

| 경로 | 내용 |
|---|---|
| `experiment_config.json` | test 점수 확인 전에 고정한 정책·조건·코드 해시·환경 |
| `input_audit/`, `cohort.csv`, `label_support.csv` | 기존 ID·split·actual 검증, 라벨 및 미분류 마스크 |
| `patient_overlap.json/csv` | split 간 환자 중복 검사 |
| `reconstruction_provenance.json`, `reconstruction_checkpoint_consistency.csv` | 재구성 출처 근거와 미확인 사항 |
| `models/<task>/<A,B,C>/seed_<seed>/best_model.pth` | 18개 실제 학습 모델의 가중치·선택 epoch·설정 |
| 같은 폴더의 `config.json`, `normalization.npz`, `history.csv`, `train_order_digests.csv` | train-only 정규화·가중치·설정·학습 곡선·순서 |
| 같은 폴더의 `thresholds.json`, `selection.json`, `validation_probabilities.csv` | validation만으로 정한 임계값과 체크포인트 선택 |
| 같은 폴더의 `predictions.npz` | 전체 ECG의 타깃·확률·예측·split·라벨·eligible·seed |
| `selection_frozen_before_test.json`, `selected_models.csv` | test 전 18개 모델 선택 해시 고정과 요약 |
| `predictions_long.csv` | ECG별 정답·확률·예측 라벨·split·seed·task·입력 |
| `metrics_by_seed.csv` | task/입력/seed/split/클래스별 지표 및 NA 사유 |
| `abc_comparison_mean_sd.csv`, `test_paired_differences.csv` | A/B/C 평균·표준편차와 같은 seed의 차이 |
| `binary_confusion_matrices.csv`, `metric_na_reasons.csv` | 이진 혼동행렬과 미계산 지표 사유 |
| `test_*comparison.png`, `test_multilabel_by_class.png`, `validation_learning_curves.png` | 결과 비교와 학습 곡선 |
| `completion.json` | 실제 18개 학습과 test 평가 완료 기록 |

실험 결과의 실제 해석은 [최종 요약](../CLASSIFICATION_FINAL_SUMMARY.md)에 연결한다.
