# Lead I 기반 ECG 유도 추정 실험 요약

작성일: 2026-10-11. 저장된 CSV·JSON, 추론 출력 NPZ와 인터페이스 문서를 읽어 작성했다. 이 문서 작성 과정에서 학습, 모델 추론 또는 test 평가를 실행하지 않았다. 원본에서 `ecg-capstone/` 기준이던 상대 경로를 유지해 팀 저장소에서는 이 문서가 있는 `ecg_reconstruction/` 기준으로 읽는다.

팀 공유본의 근거 링크는 `reports/representative/`의 선별된 사본을 가리킨다. JSON과 일부 검사 CSV의 로컬 절대 경로만 공유용 사본에서 상대 경로로 바꿨으며, 원본 결과와 지표 수치는 보존했다. 원본 경로와 원본·사본 SHA-256은 [공유 결과 목록](reports/representative/MANIFEST.json)에 기록했다. 본문에 남은 `results/`, `examples/`, `checkpoints/` 경로는 기존 로컬 실행의 출처를 뜻한다.

공유본에는 원본 데이터·가공 NPZ·입력 NPY·체크포인트를 포함하지 않았다. 본문의 해당 경로는 바로 열 수 있는 공유 파일이 아니며, 실행하려면 [별도 준비 안내](docs/CHECKPOINT_HANDOFF.md)에 따라 준비해야 한다. 수치 근거 링크는 공유본에 포함된 CSV·JSON·PNG 또는 출력 규격 JSON으로 연결된다.

## 1. 목표와 현재 관찰

단일 Lead I의 10초 파형으로 나머지 11개 유도(II, III, AVR, AVL, AVF, V1~V6)를 추정한다. 모델 입력은 `[batch, 1, 1000]`, 출력은 `[batch, 11, 1000]`이다. 추론 프로그램은 관측한 입력 I을 보존하고 추정 11개를 합쳐 12유도 배열을 저장한다.

저장된 validation 결과에서 Dilated CNN은 seed 42·43·44 모두 Baseline보다 전체 MSE·MAE·RMSE가 낮았다. 세 seed는 **동일한 데이터 분할에서 학습 seed를 바꾼 반복 실험**이다. 세 번의 독립적인 데이터 분할 실험이나 독립 외부 검증이 아니다. 아래 결과는 관찰된 성능 차이를 기술하며, 통계적 유의성이나 임상적 유효성을 주장하지 않는다.

추가로 저장된 최종 test 결과에서도 세 seed 모두 Dilated의 전체 MSE·MAE·RMSE가 낮았다. 제8절에 같은 test 100개를 세 seed로 평가한 결과를 validation과 분리해 정리했다. 기존 seed 42 Baseline test 결과를 앞서 확인한 사실도 유지한다. 근거: [최종 test 비교 CSV][testComparison], [평가 설정 JSON][testConfig].

## 2. 데이터 검사와 학습 제외 정책

원본 데이터는 `data/processed/ptbxl_100hz_1000records.npz`의 PTB-XL 1,000개 기록이다. 100 Hz, 기록당 1,000 samples(10초), 전압 단위 mV이다. 검사 결과 X는 `(1000,1000,1)`, Y는 `(1000,1000,11)`이며 모두 float32이다.

저장된 검사 기록에서 필수 키, shape·dtype, X/Y 유한값, Lead 순서, ECG ID 고유성, 환자 ID 고유성, 분할 간 환자 중복 없음 및 공식 fold 매핑을 확인했다. fold 1~8은 train, 9는 validation, 10은 test이다. 원본 분할은 **800/100/100**이다. 단, Y 전압 범위 검사는 `FAIL`로 기록됐으며(min −11.144, max 8.321 mV), 이를 수동 검토 대상으로 삼았다. 전압 검사를 포함한 모든 항목이 통과한 데이터라고 표현하지 않는다. 근거: [데이터 검사 CSV][inspection].

| ecg_id | 원본 index (0부터) | patient_id | 원본 split | 이번 실험에서의 제외 근거 |
|---|---:|---:|---|---|
| 421 | 36 | 2490 | train | V4의 국소적인 큰 음의 돌출 및 기준선 변화 |
| 710 | 249 | 5542 | train | V2 시작 구간의 큰 음의 편차 및 기준선 변화 |

두 기록 모두 로컬 WFDB 원본을 float32로 변환해 NPZ와 비교했으며, 저장된 비교 CSV에서 각 기록의 12개 Lead 모두 최대 절대 차이가 **0.0 mV**이다. 따라서 이 기록을 NPZ 변환 오류로 판단한 것이 아니다. 원본과 일치하는 큰 전압 파형을 수동 검토한 뒤 **이번 실험의 학습에서만 제외하는 정책**을 적용했다. 단순히 `|전압| > 10 mV`라는 이유만으로 자동 제외한 것이 아니다.

두 기록을 train 마스크에서만 제외해 실제 사용 분할은 **798/100/100**이다. 원본 NPZ와 validation/test 기록은 유지했다. 근거: [WFDB 원본 비교 CSV][wfdb], [제외 내역 CSV][excluded], [Baseline 실험 설정][b42config].

여섯 모델 실행의 저장된 설정을 대조한 결과 데이터 SHA-256, 실제 분할별 ECG ID와 순서, 제외 정책이 같다. 데이터 SHA-256은 `2828f1949a6c5205de243cf304afc47faabc52d55021c7b9120f19d6309662d6`이다. 이번 문서에서는 원본 데이터 검사를 다시 실행하지 않고 기존 검사·설정 기록을 근거로 사용했다.

## 3. Baseline과 Dilated CNN 구조

Baseline은 `model.ECGModel`, 후보는 `model_dilated.ECGDilatedModel`이다. 채널, kernel, stride, bias와 ReLU를 유지하고 중간 Conv1d의 dilation과 padding만 변경했다.

| 계층 | 채널 | kernel | stride | Baseline dilation / padding | Dilated dilation / padding | 뒤 활성화 |
|---|---|---:|---:|---|---|---|
| Conv1 | 1 → 32 | 7 | 1 | 1 / 3 | 1 / 3 | ReLU |
| Conv2 | 32 → 64 | 5 | 1 | 1 / 2 | 2 / 4 | ReLU |
| Conv3 | 64 → 64 | 5 | 1 | 1 / 2 | 4 / 8 | ReLU |
| Conv4 | 64 → 11 | 1 | 1 | 1 / 0 | 1 / 0 | 없음 |

모든 Conv1d는 bias를 사용한다. pooling, BatchNorm, dropout, residual connection은 없다. 각 층에서 `2 × padding = dilation × (kernel − 1)`이므로 시간 길이 1,000을 유지한다.

두 모델의 학습 파라미터 수는 **31,819개**로 같다. stride가 모두 1일 때 수용영역은 `1 + Σ((kernel − 1) × dilation)`이다.

- Baseline: `1 + 6 + 4 + 4 + 0 = 15 samples`
- Dilated: `1 + 6 + 8 + 16 + 0 = 31 samples`

저장된 구조 메타데이터의 samples/100 Hz 정의로 150 → 310 ms이며, 첫 sample에서 마지막 sample까지의 시간 차이는 각각 140/300 ms이다. 수용영역 확대가 더 넓은 주변 파형 정보를 활용하는 데 도움이 될 수 있다는 가설이며, 아래 지표만으로 개선의 생리학적 원인을 확인한 것은 아니다.

근거: [Dilated 구조·설정 JSON][d42config], [seed 43 Baseline 완료 메타데이터][b43final]의 `model_structure`.

## 4. 공통 학습 조건과 재현성 범위

두 모델은 seed 외 학습 설정과 전처리가 동일하다. 같은 seed의 모델 쌍을 비교한다.

| 항목 | 설정 |
|---|---|
| 학습 / validation 기록 | 798 / 100, 모든 seed에서 같은 ID·순서 |
| seed | 42, 43, 44 |
| batch size | 8 |
| optimizer / learning rate | Adam / 0.001 |
| loss / epochs | MSELoss(mean) / 30 |
| epoch loss 집계 | `Σ(batch_mean_loss × batch_records) / total_records` |
| shuffle | train=True, validation=False |
| 기타 | num_workers=0, drop_last=False, scheduler 없음, early stopping 없음 |
| 전처리 | 정규화·clipping·필터링·증강 없음; 입력 축 NTC → NCT 변환 |
| 체크포인트 선택 | 각 모델에서 validation MSE가 엄격히 가장 낮은 epoch |

후보는 Baseline의 학습된 가중치를 복사하지 않고 새로 초기화한다. test는 학습과 모델 선택에 사용하지 않는 것으로 실험 설정에 기록돼 있다.

seed 43·44는 저장된 완료 메타데이터에서 같은 seed 쌍의 **실제 초기 가중치 해시와 30개 epoch의 학습 ECG ID 순서 기록이 모두 일치**함을 확인했다. seed 42의 기존 실행에는 실제 초기 가중치와 모든 epoch 순서의 이력이 저장되지 않아, 과거 실행의 완전한 일치를 소급 확인할 수 없다. seed 42의 설정에는 현재 환경에서 새 생성자의 초기화 일치만 확인한 제한이 명시돼 있다. 실행 장치·라이브러리 버전이 바뀔 때 bitwise 재현성은 보장하지 않는다.

seed 43·44의 `experiment_config.json`은 실행 시작 시점의 사본이므로 `training_complete=false`, 빈 순서 목록이 남아 있다. 완료 여부와 전체 순서는 `completed_experiment.json`을 근거로 삼았으며 각 학습 이력 CSV에 30개 epoch가 있는 것도 확인했다. 근거: [seed 43 Dilated 완료 JSON][d43final], [seed 44 Baseline 완료 JSON][b44final], [seed 44 Dilated 완료 JSON][d44final], 위 seed 43 Baseline 완료 JSON 및 seed 42 설정 JSON.

## 5. Seed 42·43·44의 validation 전체 지표

아래는 각 seed에서 validation MSE로 선택한 체크포인트의 **저장된 float64 pooled 지표**이다. 입력 Lead I은 지표에서 제외한다. validation 100개 × 11 target leads × 1,000 samples의 오차를 합쳐 MSE와 MAE를 구하고, pooled RMSE는 `sqrt(pooled MSE)`로 계산한다. 숫자는 원본 CSV의 반올림 전 값을 사용해 계산하고 표에서는 소수점 9자리로 표시했다.

| seed | 모델 | MSE (mV²) | MAE (mV) | pooled RMSE (mV) |
|---:|---|---:|---:|---:|
| 42 | Baseline | 0.039451031 | 0.096437779 | 0.198622837 |
| 42 | Dilated | 0.038420368 | 0.093957140 | 0.196011144 |
| 43 | Baseline | 0.039446903 | 0.096361996 | 0.198612443 |
| 43 | Dilated | 0.038434843 | 0.094373463 | 0.196048063 |
| 44 | Baseline | 0.039226779 | 0.096520309 | 0.198057513 |
| 44 | Dilated | 0.038285326 | 0.094659739 | 0.195666365 |

| seed | Baseline 선택 epoch | Dilated 선택 epoch | MSE 감소율 |
|---:|---:|---:|---:|
| 42 | 16 | 28 | 2.612512% |
| 43 | 22 | 23 | 2.565625% |
| 44 | 24 | 24 | 2.400024% |

MSE 감소율은 각 seed에서 `100 × (Baseline MSE − Dilated MSE) / Baseline MSE`이다. 양수는 감소를 뜻한다.

| 모델 | MSE seed 평균 ± 표본 SD (mV²) | MAE seed 평균 ± 표본 SD (mV) | pooled RMSE seed 평균 ± 표본 SD (mV) |
|---|---:|---:|---:|
| Baseline | 0.039374904 ± 0.000128297 | 0.096440028 ± 0.000079180 | 0.198430931 ± 0.000323431 |
| Dilated | 0.038380179 ± 0.000082463 | 0.094330114 ± 0.000353300 | 0.195908524 ± 0.000210527 |

평균은 seed별 값 3개의 산술평균이며, 표본 표준편차는 **ddof=1**, 즉 `sqrt(Σ(x − 평균)² / (3 − 1))`이다. seed별 MSE 감소율의 평균은 **2.526054%**, 표본 SD는 **0.111634 퍼센트포인트**이다. 별도로 seed 평균 MSE에 감소율 공식을 적용하면 **2.526292%**이며, 이는 감소율 3개의 평균과 다른 계산이다.

여기의 RMSE 평균은 seed별 pooled RMSE의 평균이다. 기록별 RMSE 평균과 구분해야 한다. 기록 길이가 같으므로 기록별 MSE/MAE 평균은 pooled MSE/MAE와 같지만, **기록별 RMSE 평균은 pooled RMSE와 다르다**.

세 seed 모두 저장된 최저 validation loss와 재계산 MSE의 일치 검사를 통과했다(`atol=1e-7`, `rtol=1e-5`). 이 문서에서는 추론을 다시 실행하지 않고 기존 검사 JSON을 확인했다.

근거:

- seed 42: [validation 지표 CSV][csv42], [비교 설정 JSON][cmp42], [loss 일치 검사 JSON][loss42]
- seed 43: [validation 지표 CSV][csv43], [비교 설정 JSON][cmp43], [loss 일치 검사 JSON][loss43]
- seed 44: [validation 지표 CSV][csv44], [비교 설정 JSON][cmp44], [loss 일치 검사 JSON][loss44]

이 표의 표준편차는 같은 분할에서 학습 seed에 따른 변동을 기술한다. 독립 데이터 표본의 불확실성 추정이나 통계적 유의성 검정으로 해석하지 않는다.

## 6. Seed 42 기록별 결과와 발표 사례

seed 42 validation 100개의 기록별 overall MSE를 비교하면 **개선 68개, 악화 32개, 동률 0개**이다. 개선/악화는 candidate−baseline 차이가 각각 음수/양수인 경우이며, 동률은 정확히 0이다. 저장된 기록별 CSV에서도 이 개수를 다시 집계해 확인했다. 근거: [기록 분석 요약 JSON][recordSummary], [기록별 지표 CSV][recordCsv].

발표용 세 사례는 **validation 100개의 결과를 확인한 뒤** 기록별 overall MSE 차이로 선택했다. 개선 최대는 차이 최소, 악화 최대는 차이 최대, 중앙 사례는 차이의 중앙값에 가장 가까운 기록이다. 동률이면 ecg_id 오름차순으로 정한다. 이는 validation에서 선택한 예시이며, 독립적인 test 근거나 전체 모집단을 대표하도록 사전 선정한 사례가 아니다.

| ecg_id | 원본 index (0부터) | patient_id | 선정 기준 | Δ overall MSE (mV²) |
|---:|---:|---:|---|---:|
| 544 | 871 | 13772 | overall MSE 차이 최소: 개선 최대 | -0.018146530655 |
| 190 | 828 | 10782 | overall MSE 차이 최대: 악화 최대 | 0.024221819788 |
| 155 | 820 | 12640 | MSE 차이의 중앙값에 가장 가까움 | -0.000697455950 |

차이의 중앙값은 **−0.0006795338466311779 mV²**이다. ECG 155는 이 중앙값에 가장 가까운 기록이며, 중앙값 자체와 차이가 정확히 같은 기록이라는 뜻은 아니다.

| ecg_id | 모델 | overall MSE (mV²) | overall MAE (mV) | overall RMSE (mV) |
|---:|---|---:|---:|---:|
| 544 | Baseline | 0.399143870 | 0.286645646 | 0.631778340 |
| 544 | Dilated | 0.380997340 | 0.275055309 | 0.617249820 |
| 190 | Baseline | 0.361829127 | 0.374097989 | 0.601522341 |
| 190 | Dilated | 0.386050947 | 0.386806052 | 0.621329982 |
| 155 | Baseline | 0.009489165 | 0.051898344 | 0.097412347 |
| 155 | Dilated | 0.008791709 | 0.047991724 | 0.093764116 |

사례 해석은 다음과 같다. Δ는 모두 Dilated−Baseline이며, 각 지표는 전체 10초를 대상으로 한다.

- **ECG 544:** overall MSE가 약 0.018147 mV² 감소했다. Lead별 MSE는 10개 개선, V5 1개 악화다. V2(Δ −0.066498), V4(−0.058001), V3(−0.037980 mV²)의 감소가 크다. V5는 MSE가 +0.002436 mV² 증가했지만 MAE는 −0.005240 mV 감소했다. AVL도 MSE는 감소하지만 MAE는 +0.000323 mV 증가했다. MSE 개선을 모든 Lead·모든 지표의 개선으로 해석하지 않는다.
- **ECG 190:** overall MSE가 약 0.024222 mV² 증가했다. II·III·AVR·AVF·V6의 5개 Lead MSE는 개선됐고, AVL·V1~V5의 6개는 악화됐다. 특히 V3(Δ +0.220108), V2(+0.187911 mV²)의 증가가 크며, II(−0.102865), AVF(−0.068257 mV²)의 감소에도 전체 MSE가 증가했다.
- **ECG 155:** overall MSE가 약 0.000697 mV² 감소했다. 9개 Lead MSE가 개선됐지만 V1(Δ +0.001881), V2(+0.006726 mV²)는 악화됐다. 차이의 중앙값 부근 사례에서도 Lead별 반응은 균일하지 않다.

위 해석은 **수정 CSV**의 수치와 열 정의를 사용했다. `overall_change`는 해당 기록 전체 MSE의 변화이고, `metric_change`는 현재 overall/per_lead 행의 MSE 변화 부호이다. `metric_change`는 MAE·RMSE의 공통 변화 판정 열이 아니다. 원본 CSV의 `change`는 전체 변화를 Lead별 행에도 반복하므로 Lead별 개선 판정 근거로 사용하지 않는다. 수정본은 기존 수치와 행 순서를 유지한 별도 파일이다.

근거: [수정된 selected_record_metrics CSV][corrected], [사례 선택·파형 비교 설정 JSON][waveConfig]. 기존 결과 폴더에는 각 사례의 10초 및 0~2초 PNG가 저장돼 있다. 이 문서는 파형을 새로 생성하거나, 수치만으로 특정 파형 형태·임상 의미를 판정하지 않았다.

## 7. 독립 추론 프로그램과 입출력 확인

`predict_ecg.py`의 실제 CPU 실행 결과가 저장돼 있다. 입력은 원본 순서의 첫 validation 기록 **index 800, ecg_id 8, patient_id 11275**에서 X만 추출한 Lead I이며, 정답 Y를 예제에 포함하지 않았다. 사용 모델은 seed 42 Dilated 체크포인트, 선택 epoch 28이다.

| 항목 | 확인된 내용 |
|---|---|
| 입력 | `examples/lead_i_20261011_000435_36e7c5eb5ccb4a7b942f56220c9de992/lead_i_example.npy`, (1000,1), float32 |
| 체크포인트 | `checkpoints/dilated_1000/run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636/best_model.pth` |
| 결과 폴더 | `results/ecg_predictions/run_20261011_000527_332f77c10d404360af25b91935f36c4a/` |
| 출력 signal | (1000,12), float32, axis 0=시간, axis 1=Lead, 유한값 |
| Lead 순서 | I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6 |
| 추정 여부 | `is_estimated_lead`는 I만 False, 나머지 11개 True |
| 주파수 / 단위 | 100 Hz / mV |

저장된 출력 NPZ를 `allow_pickle=False`로 읽어 shape·dtype·유한값을 확인했고, 출력 I이 예제의 float32 입력 I과 bitwise로 같음을 확인했다. 이를 위해 모델을 다시 실행하지 않았다. 다른 11개는 직접 측정한 정답이 아니라 모델의 추정값이다. 이 실행은 입출력 동작 확인이며 별도 성능 평가가 아니다.

현재 입력 규격은 .npy의 (1000,) 또는 (1000,1), 100 Hz, 10초, mV이다. 수치형·유한값을 검사한 뒤 float32로 변환하고 유한값을 재검사한다. 자동 리샘플링·단위 변환·정규화·필터링·clipping은 하지 않으며 지원하지 않는 주파수·단위·shape나 비유한값은 오류로 종료한다.

근거: [실제 추론 메타데이터 JSON][prediction], [출력 NPZ 규격 요약][predictionNpz], [입력 예제 출처 JSON][source], [ECG_INTERFACE.md][interface]. 파형 수치가 담긴 원본 NPZ는 공유용 저장소에서 제외하며 로컬 결과로 보존한다.

## 8. 최종 test 평가: Seed 42·43·44

최종 test 평가 실행 폴더는 `results/test_comparison_multiseed/run_20261011_010820_9c3607152b144c24b82e590f7c67f94a/`이다. 이 절은 이미 저장된 결과를 읽어 추가했으며 학습이나 모델 추론을 재실행하지 않았다. `completion.json`에 여섯 모델의 평가 완료가 기록돼 있다.

### 평가 대상과 집계 범위

validation MSE로 선택한 여섯 best-validation 체크포인트를 그대로 평가했다. 저장된 설정에서 실제 모델 클래스와 dilation, 체크포인트 경로·해시·선택 epoch, 학습 설정과 최저 validation 이력 검증을 확인했다. 모든 모델의 데이터 해시와 test ECG ID 목록이 같고, 완료 기록의 실제 평가 순서 해시도 여섯 모델에서 일치한다.

평가 대상은 원본 index 900~999의 **서로 다른 test ECG 100개**이다. 같은 100개를 seed 42·43·44의 각 모델로 반복 평가했다. **독립적인 test 300개가 아니다.** `all_seed_pooled_metrics.csv`의 `records=300`은 한 모델 구조에 대해 100개 기록을 세 seed로 예측한 300건의 평가를 뜻하며, `unique_test_records=100`이다.

저장된 실행 환경은 CPU, Python 3.14.8, PyTorch 2.13.0+cu130, NumPy 2.5.3이며 batch size는 8이다. `shuffle=False`, `eval()`, `inference_mode()`를 사용했다. 예측과 정답을 각각 float64로 변환한 뒤 오차를 계산하고 float64로 합산했다. 입력 Lead I은 제외하며 overall은 100 × 11 × 1,000개 오차의 pooled 지표이고 Lead별은 100 × 1,000개 오차의 pooled 지표다. RMSE는 `sqrt(MSE)`이다.

**기존 seed 42 Baseline test 결과는 이번 여섯 모델의 최종 평가 전에 이미 확인했다.** 이 사실은 평가 설정의 `prior_seed42_baseline_test_exposure.already_inspected=true`와 정책에 명시돼 있다. 따라서 전체 실험을 test를 한 번도 확인하지 않은 절차로 표현하지 않는다. 아래 수치는 이번 통일된 float64 집계 결과이며, 과거 seed 42 Baseline 평가 CSV를 혼합하지 않았다. test 결과로 모델·seed·체크포인트를 다시 선택하지 않았으며 test 파형 사례 선정이나 추가 학습을 수행하지 않았다.

근거: [최종 test 평가 설정 JSON][testConfig], [평가 완료·순서 확인 JSON][testCompletion], [앞서 확인한 seed 42 Baseline test CSV][priorTest42].

### Seed별 overall test 지표

| seed | 모델 | MSE (mV²) | MAE (mV) | pooled RMSE (mV) |
|---:|---|---:|---:|---:|
| 42 | Baseline | 0.038000941 | 0.090333426 | 0.194938301 |
| 42 | Dilated | 0.035629481 | 0.087300666 | 0.188757732 |
| 43 | Baseline | 0.037098298 | 0.090310109 | 0.192609186 |
| 43 | Dilated | 0.035906743 | 0.087241539 | 0.189490746 |
| 44 | Baseline | 0.037080311 | 0.090455272 | 0.192562486 |
| 44 | Dilated | 0.035417047 | 0.088313344 | 0.188194174 |

| seed | Baseline 선택 epoch | Dilated 선택 epoch | Δ test MSE (Dilated−Baseline, mV²) | test MSE 감소율 |
|---:|---:|---:|---:|---:|
| 42 | 16 | 28 | -0.002371459833 | 6.240529% |
| 43 | 22 | 23 | -0.001191555717 | 3.211888% |
| 44 | 24 | 24 | -0.001663263907 | 4.485572% |

MSE 감소율은 같은 seed 쌍에서 `100 × (Baseline MSE − Dilated MSE) / Baseline MSE`로 계산했다. 세 seed 모두 overall MSE·MAE·RMSE가 감소했다. 위 표의 epoch는 test 성능으로 고른 값이 아니라 앞서 validation으로 고정한 값이다. 근거: [seed별 test 지표 CSV][testMetrics], [같은 seed 쌍 test 비교 CSV][testComparison].

### 모델별 seed 평균과 표본 표준편차

| 모델 | test MSE seed 평균 ± 표본 SD (mV²) | test MAE seed 평균 ± 표본 SD (mV) | test pooled RMSE seed 평균 ± 표본 SD (mV) |
|---|---:|---:|---:|
| Baseline | 0.037393183 ± 0.000526410 | 0.090366269 ± 0.000077956 | 0.193369991 ± 0.001358397 |
| Dilated | 0.035651090 ± 0.000245562 | 0.087618516 ± 0.000602464 | 0.188814217 ± 0.000650129 |

평균은 seed별 지표 3개의 산술평균, 표본 SD는 **ddof=1**이다. 저장된 summary CSV를 원본 seed별 CSV의 수치로 다시 계산해 확인했다. 같은 test 기록에 대한 학습 seed 변동을 나타내며 독립 test 표본 3개나 독립 기록 300개의 변동을 나타내지 않는다.

seed별 test MSE 감소율의 평균은 **4.645996%**, 표본 SD는 **1.520681 퍼센트포인트**이다. seed 평균 MSE에 감소율 공식을 적용한 값은 **4.658852%**로, 감소율 3개의 평균과 계산 방식이 다르다. 근거: [seed 평균·표본 SD CSV][testSeedSummary], [seed 쌍 비교 CSV][testComparison].

seed별 pooled RMSE의 평균과 모든 seed의 오차를 합친 pooled RMSE는 다음처럼 구분된다.

| 모델 | seed별 pooled RMSE의 평균 (mV) | 모든 seed 오차를 합친 pooled RMSE (mV) |
|---|---:|---:|
| Baseline | 0.193369990782 | 0.193373171606 |
| Dilated | 0.188814217065 | 0.188814963243 |

앞 열은 `mean(sqrt(MSE_seed))`, 뒤 열은 `sqrt(mean(MSE_seed))`이다(각 seed의 평가 크기가 동일). 뒤 열에서도 고유 test 기록 수는 100개다. MSE·MAE는 동일 크기 seed의 평균과 모든 seed를 합친 pooled 값이 같지만, RMSE는 제곱근을 취하는 순서 때문에 다르다. 근거: [seed 평균 CSV][testSeedSummary], [모든 seed 오차의 pooled CSV][testAllPooled].

### Lead별 개선·악화

아래 개수는 각 seed의 **11개 target Lead에 대한 pooled 지표**의 비교이며, 개선·악화한 ECG 기록 수가 아니다. MSE와 RMSE는 제곱근의 단조성 때문에 변화 부호가 같다. MAE는 따로 비교한다. 동률은 차이가 정확히 0인 경우이다.

| seed | MSE/RMSE 개선 / 악화 / 동률 Lead 수 | MSE/RMSE 악화 Lead | MAE 개선 / 악화 / 동률 Lead 수 | MAE 악화 Lead |
|---:|---|---|---|---|
| 42 | 10 / 1 / 0 | V5 | 11 / 0 / 0 | 없음 |
| 43 | 10 / 1 / 0 | V2 | 10 / 1 / 0 | AVR |
| 44 | 10 / 1 / 0 | V1 | 9 / 2 / 0 | III, AVL |

세 seed 평균을 비교한 Lead별 차이는 다음과 같다. 각 Δ는 **Dilated seed 평균−Baseline seed 평균**이며 음수면 개선이다. RMSE의 Δ도 seed별 pooled RMSE의 평균끼리 비교한 값이다.

| Lead | Δ 평균 MSE (mV²) | Δ 평균 MAE (mV) | Δ 평균 RMSE (mV) |
|---|---:|---:|---:|
| II | -0.000967003 | -0.001646892 | -0.003824928 |
| III | -0.000557390 | -0.001090145 | -0.002218019 |
| AVR | -0.000250307 | -0.000596982 | -0.001901869 |
| AVL | -0.000288355 | -0.000938092 | -0.002219988 |
| AVF | -0.001015176 | -0.001783008 | -0.004026895 |
| V1 | -0.000606663 | -0.002400154 | -0.001583195 |
| V2 | -0.006010080 | -0.007290151 | -0.011080044 |
| V3 | -0.007369474 | -0.008748414 | -0.013046715 |
| V4 | -0.001546867 | -0.003641497 | -0.003044889 |
| V5 | -0.000150983 | -0.001003430 | -0.000403429 |
| V6 | -0.000400725 | -0.001086513 | -0.000835205 |

seed 평균에서는 11개 Lead 모두 MSE·MAE·RMSE가 낮아졌지만, 각 seed의 모든 Lead가 동시에 개선된 것은 아니다. 평균 MSE 감소의 절대 크기는 V3, V2에서 컸다. 이 수치만으로 파형의 특정 구간이나 임상적 특징이 개선됐다고 단정하지 않는다. 근거: [Lead별 seed 쌍 비교 CSV][testComparison], [Lead별 seed 평균·표본 SD CSV][testSeedSummary].

## 9. Validation과 test를 구분한 최종 결론

**Validation:** 같은 validation 100개에서 세 seed 모두 Dilated의 overall MSE·MAE·RMSE가 낮았다. 같은 seed의 MSE 감소율은 2.612512%, 2.565625%, 2.400024%였다. validation MSE는 체크포인트 선택 기준이었고 ECG 544·190·155도 validation 결과를 확인한 뒤 선택한 사례다. 이 사례를 test 사례로 해석하지 않는다.

**Test:** validation으로 고정한 여섯 체크포인트를 같은 test 100개에 적용했을 때도 세 seed 모두 Dilated의 overall MSE·MAE·RMSE가 낮았다. 같은 seed의 MSE 감소율은 6.240529%, 3.211888%, 4.485572%였다. 각 seed에서 MSE/RMSE가 악화된 Lead가 하나씩 있으므로 개선이 모든 seed·모든 Lead에서 균일하다고 말할 수 없다.

현재 결과는 동일 PTB-XL 데이터 분할에서 수용영역을 15→31 samples로 확대한 후보의 전체 오차 감소가 validation과 test 양쪽에서 관찰됐음을 보여준다. test 결과로 추가 모델·seed·체크포인트를 선택하지 않는다. 이전 Baseline test 결과를 이미 확인했던 이력을 유지하며, 통계적 유의성이나 임상적 유효성을 주장하지 않는다. validation과 test의 수치 크기 또는 감소율 차이만으로 두 데이터 집단의 난이도나 개선 원인을 확정하지 않는다.

근거: 제5절의 validation CSV·JSON과 제8절의 [최종 test 지표][testMetrics], [test 비교][testComparison], [평가 정책·설정][testConfig].

## 10. 한계와 미완료 사항

- 여섯 best-validation 체크포인트의 **최종 test 평가는 완료**됐다. 기존 seed 42 Baseline test 결과를 앞서 확인한 사실도 유지한다. 이번 결과로 개발을 이어가거나 모델·seed·체크포인트를 다시 선택하지 않는다.
- **실제 스마트워치 입력은 미검증**이다. 현재 데이터·validation/test 비교·독립 추론 예제의 검증 범위는 PTB-XL이다. 장치별 Lead·샘플링·단위·잡음 차이에 대한 검증이 필요하다.
- **분류 모델 연결은 미완료**이다. 분류 담당자의 필요한 주파수, 길이, 단위, Lead 구성·순서, 배열 방향·batch·dtype, 전처리를 확인해야 한다. 분류 입력 조건을 임의로 가정하거나 현재 출력을 자동 변환하지 않는다. 합의 항목은 ECG_INTERFACE.md에 정리돼 있다.
- 세 seed는 각각 같은 validation 100개와 같은 test 100개를 반복 사용했다. 서로 다른 데이터 분할이나 독립적인 test 300개로 표현하지 않는다. 데이터 분할 변동, 외부 기관·장치 및 더 큰 데이터셋으로의 일반화는 확인되지 않았다.
- validation MSE로 체크포인트를 선택했고 발표 사례도 validation 결과를 보고 선택했다. 이 validation 사례는 독립 test 사례가 아니다. test 비교를 포함한 현재 결과만으로 통계적 유의성이나 임상적 유효성을 판단할 수 없다.
- 학습에서 제외한 421·710은 원본과 일치하는 실제 파형이다. 제외는 이번 실험 정책이며, 모든 큰 전압 기록이 잘못된 데이터라는 뜻이 아니다.

## 11. 기존 계획 대비 현재 구현

### 계획과 결과의 출처 구분

팀의 최종 목표인 **ECG 기반 심장질환 예측은 유지**한다. 역할 재분배 이후 이 문서 작성자의 담당 범위를 **Lead I으로 나머지 11개 유도를 추정하는 모듈**로 구체화했다. 이 모듈의 개발·평가 완료를 팀의 질환 예측 시스템 전체가 완료됐다는 의미로 사용하지 않는다.

아래 **기존 계획**은 이번 문서 추가 요청에서 작성자가 제공한 기존 발표 내용에 근거한다. 현재 작업 폴더에서 발표자료 원본(PPT/PDF)을 확인하지 못했으므로 슬라이드 원문·페이지와 직접 대조해 확인한 계획이라고 주장하지 않는다. 반면 **현재 구현의 실험 결과**는 이 문서의 저장된 CSV·JSON과 입출력 규격을 근거로 한다. 계획은 완료 기능이나 실험 성과의 근거로 취급하지 않으며, 확인되지 않은 팀원의 진행 상태는 단정하지 않는다.

### 기존 계획 / 현재 구현 / 변경 이유

| 기존 계획 | 현재 구현 | 변경 이유 |
|---|---|---|
| 스마트워치의 단일 유도 ECG를 활용한 심장질환 예측·분석 시스템 | 팀 목표는 유지하되, 역할 재분배 이후 담당 모듈은 Lead I → 나머지 11개 유도 파형 추정으로 구체화 | 담당 모듈의 책임과 팀 전체 시스템의 목표·완료 여부를 구분 |
| 기존 발표에서 Icentia11k 단일 유도 데이터 사용을 계획 | 유도 추정 학습에는 PTB-XL의 같은 기록에서 얻은 Lead I 입력과 나머지 11개 정답 파형을 사용 | 지도학습으로 11개 파형을 추정하려면 입력에 대응하는 정답 파형이 필요하므로 PTB-XL을 선정 |
| 단일 유도 ECG 활용 계획 | 현재 입력 조건은 100 Hz·10초(1,000 samples)·mV로 고정 | 학습·평가·독립 추론의 입력 규격을 일치시키기 위함; 실제 스마트워치 입력이 이 조건과 같다고 가정하지 않음 |
| 필터·R-peak 검출·HRV 분석 계획 | 이번 재구성 baseline은 정규화·clipping·필터링·증강을 추가하지 않으며 R-peak·HRV 특징 분석을 학습 단계에 추가하지 않음 | 동일 데이터와 학습 조건에서 모델 구조 차이를 비교하기 위한 실험 정책; 전처리나 특징 분석 자체가 불필요하다는 결론이 아님 |
| ECG 기반 질환 예측을 위한 모델 개발 계획 | 기본 CNN과 Dilated CNN을 동일 조건에서 비교하고 파라미터 31,819개를 유지하면서 수용영역을 15→31 samples로 확대 | 채널·kernel·활성화·Conv 계층 수를 유지하고 중간 dilation·padding만 바꿔 더 넓은 주변 파형 정보의 활용 효과를 비교 |
| 기존 발표 모델의 성과와 이번 재구성 실험의 성과는 구분 필요 | 이번 Baseline 대비 Dilated의 seed 평균 MSE가 validation 약 2.53%, test 약 4.66% 감소 | 같은 PTB-XL 분할에서 seed 42·43·44를 반복한 두 재구성 모델의 오차 비교; 기존 발표 모델을 평가한 결과나 질환 분류 정확도가 아님 |
| 위험도 추적·AI 판단 근거·과거 기록 비교·자동 리포트 | 현재 재구성 모듈의 완료 기능에는 포함하지 않음. 현재 공유 범위는 유도 추정·입출력 저장 및 검증·실험 결과 | 기존 팀 서비스 계획과 현재 모듈의 구현 범위를 구분; 다른 팀원이 해당 기능을 구현했는지는 이 문서에서 확인하지 않음 |
| 실제 스마트워치 입력을 이용한 질환 예측 시스템 연결 | PTB-XL에서 재구성 모델과 독립 추론 입출력을 확인했으나 실제 스마트워치 적용 및 분류 모델 연결은 미완료 | 장치별 입력 차이의 검증과 분류 모델의 주파수·길이·단위·Lead 순서·배열 방향·전처리 조건 합의가 추가로 필요 |

### 이번 실험으로 확인한 범위

데이터 선택·제외 정책은 제2절, 모델 구조는 제3절, 동일 학습 조건과 validation MSE 기반 체크포인트 선택은 제4절을 따른다. 두 모델 모두 같은 train 798개와 validation 100개를 사용하고 같은 test 100개로 최종 평가했다. 파라미터 수와 수용영역의 근거는 [Dilated 구조·설정 JSON][d42config]과 [Baseline 구조 메타데이터][b43final]이다.

이 절의 감소율은 **모델별 seed 평균 MSE**에 `100 × (Baseline 평균 MSE − Dilated 평균 MSE) / Baseline 평균 MSE`를 적용한 값이다.

| 평가 구간 | Baseline seed 평균 MSE (mV²) | Dilated seed 평균 MSE (mV²) | seed 평균 MSE 기준 감소율 |
|---|---:|---:|---:|
| validation | 0.039374904 | 0.038380179 | 2.526292% (약 2.53%) |
| test | 0.037393183 | 0.035651090 | 4.658852% (약 4.66%) |

계산에는 저장된 반올림 전 수치를 사용했다. 근거: [seed 42 validation CSV][csv42], [seed 43 validation CSV][csv43], [seed 44 validation CSV][csv44], [test seed 평균 CSV][testSeedSummary]. 이는 seed별 감소율 3개의 평균(제5절 2.526054%, 제8절 4.645996%)과 계산 방식이 다르다. 세 seed는 동일 분할의 반복 실험이고 고유 test 기록은 100개이며, 독립적인 test 300개가 아니다.

위 수치는 **이번 Baseline과 Dilated CNN 사이의 유도 파형 추정 MSE 감소**다. 기존 발표 모델 대비 개선, 심장질환 분류 정확도·위험도 예측 성능, 임상적 유효성 또는 통계적 유의성을 뜻하지 않는다. 기존 seed 42 Baseline test 결과를 최종 비교 전에 이미 확인한 이력도 제8절과 동일하게 유지한다.

전처리 없음은 이번 baseline 비교의 범위와 재현성을 고정한 정책이다. 필터·R-peak·HRV 분석의 유용성이나 필요성을 검증한 실험이 아니며, 실제 스마트워치·분류 모델에 필요한 처리는 입력 조건과 별도 검증으로 정해야 한다. 현재 자동 리샘플링·필터링을 하지 않는 독립 추론 규격은 [ECG_INTERFACE.md][interface], 저장된 실제 실행 근거는 [추론 메타데이터 JSON][prediction]에 있다.

[inspection]: reports/representative/results/inspection_1000/run_20261010_223829_seed42_e4952ad1/inspection_checks.csv
[wfdb]: reports/representative/results/flagged_ecg_review/run_20261010_224757_103666c0/original_comparison.csv
[excluded]: reports/representative/results/baseline_1000/run_20261010_230450_b4bed3d513ff42ddba68f2d7a6647034/excluded_records.csv
[b42config]: reports/representative/results/baseline_1000/run_20261010_230450_b4bed3d513ff42ddba68f2d7a6647034/experiment_config.json
[d42config]: reports/representative/results/dilated_1000/run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636/experiment_config.json
[b43final]: reports/representative/results/baseline_multiseed/run_20261011_002652_a8cd2ab69a3d479db661a33430b4ecf9/completed_experiment.json
[d43final]: reports/representative/results/dilated_multiseed/run_20261011_002749_0d998b6396074cfe8f18ad6df4d0cb5e/completed_experiment.json
[b44final]: reports/representative/results/baseline_multiseed/run_20261011_003041_c5722aac03554995923061bf4806a705/completed_experiment.json
[d44final]: reports/representative/results/dilated_multiseed/run_20261011_003125_fcf2094d5c794ae7a87618edaa16aafc/completed_experiment.json
[csv42]: reports/representative/results/validation_comparison_1000/comparison_20261010_234617_9abb88c150bc42a59f5be2bb724d9ddd/validation_metrics_comparison.csv
[cmp42]: reports/representative/results/validation_comparison_1000/comparison_20261010_234617_9abb88c150bc42a59f5be2bb724d9ddd/comparison_config.json
[loss42]: reports/representative/results/validation_comparison_1000/comparison_20261010_234617_9abb88c150bc42a59f5be2bb724d9ddd/validation_loss_checks.json
[csv43]: reports/representative/results/validation_comparison_multiseed/comparison_20261011_002951_b0db27a37b654f4bb9863536a8475a25/validation_metrics_comparison.csv
[cmp43]: reports/representative/results/validation_comparison_multiseed/comparison_20261011_002951_b0db27a37b654f4bb9863536a8475a25/comparison_config.json
[loss43]: reports/representative/results/validation_comparison_multiseed/comparison_20261011_002951_b0db27a37b654f4bb9863536a8475a25/validation_loss_checks.json
[csv44]: reports/representative/results/validation_comparison_multiseed/comparison_20261011_003308_8be4347d223b4b76a0e4886f764a2a08/validation_metrics_comparison.csv
[cmp44]: reports/representative/results/validation_comparison_multiseed/comparison_20261011_003308_8be4347d223b4b76a0e4886f764a2a08/comparison_config.json
[loss44]: reports/representative/results/validation_comparison_multiseed/comparison_20261011_003308_8be4347d223b4b76a0e4886f764a2a08/validation_loss_checks.json
[recordSummary]: reports/representative/results/validation_record_analysis/run_20261010_235542_bec7e6db196a46c091a3fb3241ae1093/analysis_summary.json
[recordCsv]: reports/representative/results/validation_record_analysis/run_20261010_235542_bec7e6db196a46c091a3fb3241ae1093/record_metrics_comparison.csv
[corrected]: reports/representative/results/validation_waveform_comparison/run_20261011_004136_766b2ae8d3a44039a4279e34d9eaa807/selected_record_metrics_corrected_20261011_004938_427bafa7.csv
[waveConfig]: reports/representative/results/validation_waveform_comparison/run_20261011_004136_766b2ae8d3a44039a4279e34d9eaa807/comparison_config.json
[prediction]: reports/representative/results/ecg_predictions/run_20261011_000527_332f77c10d404360af25b91935f36c4a/prediction_metadata.json
[predictionNpz]: reports/representative/prediction_output_schema.json
[source]: reports/representative/examples/lead_i_20261011_000435_36e7c5eb5ccb4a7b942f56220c9de992/source.json
[interface]: ECG_INTERFACE.md
[testConfig]: reports/representative/results/test_comparison_multiseed/run_20261011_010820_9c3607152b144c24b82e590f7c67f94a/evaluation_config.json
[testCompletion]: reports/representative/results/test_comparison_multiseed/run_20261011_010820_9c3607152b144c24b82e590f7c67f94a/completion.json
[testMetrics]: reports/representative/results/test_comparison_multiseed/run_20261011_010820_9c3607152b144c24b82e590f7c67f94a/test_metrics.csv
[testComparison]: reports/representative/results/test_comparison_multiseed/run_20261011_010820_9c3607152b144c24b82e590f7c67f94a/test_metrics_comparison.csv
[testSeedSummary]: reports/representative/results/test_comparison_multiseed/run_20261011_010820_9c3607152b144c24b82e590f7c67f94a/seed_metric_summary.csv
[testAllPooled]: reports/representative/results/test_comparison_multiseed/run_20261011_010820_9c3607152b144c24b82e590f7c67f94a/all_seed_pooled_metrics.csv
[priorTest42]: reports/representative/results/baseline_1000/run_20261010_230450_b4bed3d513ff42ddba68f2d7a6647034/evaluation_20261010_230604_099dc08d7724480ba219b86264ab3918/test_metrics.csv

