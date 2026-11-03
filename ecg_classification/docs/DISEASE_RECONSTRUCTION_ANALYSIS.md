# 역할 3: 질환별 ECG 재구성 성능 분석

분석 스크립트는 [`analyze_disease_reconstruction.py`](../analyze_disease_reconstruction.py)다.
기존 학습·추론 코드를 import하거나 변경하지 않고 전달된 재구성 배열만 비교한다.
새 모델 학습, 질환 분류, split 재생성, 모델·체크포인트 선택은 수행하지 않는다.

## 실행

`ecg-capstone` 폴더에서 실행한다. 두 전달 파일의 기본 위치는 `data/received/`다.

```powershell
.\.venv\Scripts\python.exe -B .\analyze_disease_reconstruction.py --validate-only
.\.venv\Scripts\python.exe -B .\analyze_disease_reconstruction.py
```

다른 경로 또는 클래스 열을 명시할 수도 있다.

```powershell
.\.venv\Scripts\python.exe -B .\analyze_disease_reconstruction.py `
  --labels "data/received/ecg_labels_analysis_998.csv" `
  --reconstructions "data/received/ecg_for_classification.npz" `
  --classes-column diagnostic_classes
```

기존 프로젝트의 NumPy·pandas·Matplotlib을 사용하며 추가 패키지가 필요하지 않다.
각 실행은 `results/disease_reconstruction/run_<한국시간>_<고유값>/`을 만든다.
`--output <새 폴더>`도 가능하며 기존 결과 폴더는 덮어쓰지 않는다.
검증 실패 시 종료 코드는 2이고 `audit.json`에 사유를 기록한다.
검증 전용 실행은 지표·그래프를 계산하지 않고 감사 결과만 저장한다.

## 입력과 대조 기준

- CSV: `ecg_id`, `split`, 명시적인 진단 superclass가 필요하다. 실제 전달 파일은 `diagnostic_classes`에 `CD;MI`처럼 저장되어 있다. 같은 파일의 NORM/MI/STTC/CD/HYP 이진 열과도 일치해야 한다. `include_in_analysis`와 `label_status`만으로 클래스를 정하지 않는다.
- NPZ: `ecg_id`, `split`, `lead_names`, `actual_12lead`, `reconstructed_12lead`. 파형은 `[N,T,12]` 또는 `[N,12,T]`이며 유도 축을 자동 판별한다. 모호하면 `--waveform-layout NTC` 또는 `NCT`를 명시한다. `allow_pickle=False`를 사용한다.
- `patient_id`, `lead_i`, `sampling_rate_hz`, `voltage_unit`가 NPZ에 있으면 원본과 일치하는지 확인한다. CSV의 `patient_id`, `strat_fold`도 존재하면 원본과 대조한다. 단위는 원본 mV이며 실제 파형을 원본과 검증한 후 적용한다.
- 원본: `data/processed/ptbxl_100hz_1000records.npz`. 파일 SHA-256은 기존 Dilated seed 42 설정에 저장된 값과 같아야 한다.
- 분할 기준: `results/dilated_1000/run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636/experiment_config.json`의 `split_ecg_ids`. 원본 800/100/100 중 train ECG 421·710만 제외한 798/100/100의 정확한 ID 집합을 요구한다. 두 전달 파일의 998개가 이 집합과 일치해야 한다. 원본 split도 다시 대조한다.

중복을 자동 제거하거나 inner join으로 누락 기록을 버리지 않는다. 두 파일과 기존 분할 ID 집합이 다르거나 split이 어긋나면 지표 계산을 중단한다. CSV 순서와 NPZ 순서는 달라도 `ecg_id`로 one-to-one join한다. 모든 입력의 해시를 남기고 읽는 동안 파일 변경 여부도 확인한다.

실제 파형은 원본의 X(Lead I)와 Y(나머지 11개)를 결합하고 NPZ의 `lead_names` 순서로 재배열해 대조한다. 절대 허용 오차는 1e-6 mV, 상대 허용 오차는 0이다. 정규화·필터링·시간 이동·진폭 보정·리샘플링은 하지 않는다. 샘플링 속도가 전달 파일에 없으면 검증된 원본의 값을 사용하고 감사 파일에 명시한다.

## 그룹

| 그룹 | 조건 | 중복 포함 |
|---|---|---|
| normal_class_only | 진단 집합이 정확히 `{NORM}` | 없음 |
| MI, STTC, CD, HYP | NORM이 없고 해당 질환이 있음 | 여러 질환이 있으면 각 그룹에 포함 |
| normal_disease_mixed | NORM과 한 개 이상 질환이 동시에 있음 | 위 정상·질환 그룹에서는 제외 |
| unclassified | 진단 superclass 집합이 비어 있음 | 다른 그룹에서는 제외 |

`include_in_analysis`는 원래 값을 보존하고 교차표만 만든다. 참이라고 정상 또는 질환으로 간주하지 않는다. 미분류도 별도 그룹으로 집계한다. 알 수 없는 superclass 문자열은 미분류로 바꾸지 않고 오류를 낸다. 원시 SCP 코드를 진단 superclass로 임의 매핑하지 않는다.

이 정책에서 MI 등은 **정상 혼합을 제외한 질환 보유군**이다. `CD;NORM`은 CD군에 포함되지 않고 혼합군에만 들어간다. MI+STTC 등 질환끼리의 중복은 허용하므로 그룹 표본 수를 더해 전체 고유 ECG 수로 해석하면 안 된다. `group_overlap.csv`로 교집합 크기를 확인한다.

## 지표와 분모

전달된 전체 998개 기록에 ECG별·유도별 원시 지표를 계산하고 split을 보존한다. 그룹별 집계·비교 그래프·예시·해석은 validation과 test에 대해서만 따로 만든다. train 원시 지표는 학습 자료의 기술적 확인용이며 일반화 성능으로 해석하지 않는다. 유도 순서는 NPZ `lead_names`를 따르며 이름이 I인 유도만 제외한다. 유도 위치를 0으로 가정하지 않는다.

각 ECG e, 유도 l의 T개 샘플에서 float64로 계산한다.

```text
MAE(e,l)  = mean_t(abs(reconstructed - actual))
MSE(e,l)  = mean_t((reconstructed - actual)^2)
RMSE(e,l) = sqrt(MSE(e,l))
r(e,l)    = sum_t((actual-mean(actual))*(reconstructed-mean(reconstructed)))
            / sqrt(sum_t((actual-mean(actual))^2) * sum_t((reconstructed-mean(reconstructed))^2))
```

MAE/RMSE 단위는 mV, MSE는 mV²이며 r은 무차원이다. 그룹·유도별 mean, 표본 SD(ddof=1), median, 25/75 백분위는 ECG별 지표를 집계한 것이다. 모든 기록은 같은 1,000 samples이므로 유도별 평균 MAE는 샘플을 모두 모은 MAE와 같다. `pooled_rmse = sqrt(mean(MSE(e,l)))`는 평균 RMSE와 다르므로 별도 열로 남긴다.

`OVERALL`의 mean/SD/median/IQR은 **먼저 ECG마다 11유도를 집계하고 ECG 간에 집계**한다. ECG별 MAE는 11유도 MAE 평균, RMSE는 11유도 MSE 평균의 제곱근, r은 유효 유도 r의 산술평균이다. 유도×시간 파형을 이어 붙여 r을 계산하지 않으며 Fisher z 변환도 하지 않는다. SD는 ECG를 단위로 하며 11개 유도를 독립 표본으로 취급하지 않는다. OVERALL의 `pooled_rmse`는 전체 유효 ECG×유도 쌍의 MSE를 평균한 후 제곱근을 취한다.

비유한 값이 하나라도 있는 ECG·유도 쌍의 MAE/RMSE/r은 모두 빈 값으로 남긴다. 다른 유도는 계산한다. ECG 전체 MAE/RMSE는 11개 유도 오차가 모두 유효해야 계산한다. ECG 전체 r은 유효 유도가 있을 때만 평균하고 `n_valid_r_leads`를 기록한다. 유효 쌍 수와 전체 ECG 수를 구분해야 한다.

상수 actual/reconstructed/both는 r이 정의되지 않으므로 각각 `constant_actual`, `constant_reconstructed`, `constant_both`로 표시한다. 중심화한 L2 norm이 `1e-13 × abs(mean)` 미만인 준상수도 `near_constant`로 표시하고 r 평균에서 제외한다. 비유한 actual/reconstructed와 수치 실패도 별도 사유다. 실제 파형에 비유한 값이 있으면 원본 검증 단계에서 중단한다. 상수 파형의 MAE/RMSE는 유한한 경우 정상 계산한다. r의 빈 값을 0으로 채우지 않는다. n=0이면 통계는 NA, n=1이면 SD는 NA다.

## 산출물

| 파일 | 내용 |
|---|---|
| `audit.json` | 입력·코드 해시, 버전, 검증 결과, ID 누락 목록, 분할 수, 원본 파형 차이 |
| `duplicate_ids.csv`, `id_coverage.csv`, `split_mismatches.csv` | 중복·누락·split 감사; 문제가 없어도 헤더를 저장 |
| `joined_records.csv` | 998개 기록의 ID 기반 결합과 그룹, NPZ 원래 인덱스 |
| `include_flag_crosstab.csv` | include_in_analysis와 실제 클래스 그룹의 교차표 |
| `ecg_lead_metrics.csv` | 전체 998개×11유도 지표·상관계수 상태; split 유지 |
| `ecg_overall_metrics.csv` | ECG별 11유도 집계, r 유효 유도 수 |
| `correlation_issues.csv` | r을 계산하지 않은 ECG·유도와 사유 |
| `group_counts.csv`, `group_overlap.csv` | split별 그룹 수와 그룹 간 교집합 수 |
| `group_lead_metrics.csv` | split별 7그룹×(11유도+OVERALL)의 통계와 유효 분모 |
| `*_group_comparison.png` | ECG 단위 그룹별 지표 분포; 빈 그룹은 빈 축 위치 |
| `*_lead_heatmaps.png` | 그룹×유도별 평균 지표 |
| `waveform_examples.csv`, `waveforms/*.png` | 예시 ID·선정 기준·그룹 수와 실제/재구성 파형 겹침 |
| `REPORT.md` | 실제 수치, 그림, 결과 해석과 한계 |

파형 예시는 각 split·그룹에서 ecg_id가 가장 작은 기록으로 고정한다. test 오차, r, 파형 외관으로 선택하지 않는다. 여러 질환에 속한 동일 ECG가 반복될 수 있다. 입력 I도 그림에는 표시하지만 성능 지표에는 넣지 않는다.

## 해석 범위

MAE/RMSE는 진폭 오차, r은 선형 형태 유사도를 기술한다. 높은 r도 진폭·기준선 차이를 감출 수 있다. 10초 평균 오차로 국소 ST 변화나 QRS 형태, 진단 정보 보존을 보증할 수 없다.

n<10인 그룹은 주의 표시를 붙인다. 이는 편의상 정한 표시 기준이며 n≥10이라고 안정적이라는 뜻이 아니다. 질환군은 중복되고 질환 중증도·성별·연령·기타 질환 등 구성이 다르므로 그룹 차이를 해당 질환의 인과 효과로 해석하지 않는다. 유의성 검정이나 임상 평가, 환자 수준 보정은 수행하지 않는다.

validation은 기존 모델 선택에 사용되었으며 test와 따로 해석한다. 기존 test 노출 이력이 있으므로 새로운 완전 미관측 외부 검증으로 설명하지 않는다. 재구성 모델의 정확한 체크포인트와 생성 절차는 입력 파일에 증빙 메타데이터가 없으면 확정할 수 없다. ID·split·actual 일치만으로 모델 누출 부재를 보증하지 않는다. 이번 분석에서 모델을 실행하지 않는다.

파형 재구성 결과를 질환 분류 정확도, 민감도, 특이도, AUC 또는 임상적 진단 성능으로 해석하지 않는다.

## 검증 테스트

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_disease_reconstruction.py -v
```

합성 데이터로 ID 순서가 다른 결합, 중복·누락·split 오류 차단, 원본 해시·파형 불일치 차단, 다중 질환·정상 혼합·미분류, include 플래그 독립성, 유도 순서가 바뀐 Lead I 제외, 알려진 MAE/RMSE/r, 상수·준상수·비유한 파형, 평균 RMSE와 pooled RMSE 차이를 검사한다. 합성 데이터의 지표는 실제 연구 결과로 사용하지 않는다.
