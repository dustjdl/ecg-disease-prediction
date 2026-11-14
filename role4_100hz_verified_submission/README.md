# 4번 담당 — 100Hz CNN vs U-Net 공통 평가·검증

## 범위 및 결론
2026년 11월 팀 협의에 따라 **500Hz는 제외**하고 **100Hz CNN과 1D U-Net**의 공유 **validation 100건**을 검증하였다. 저장된 ECG별·유도별 오차 기록에서 독립적으로 집계한 3시드 평균 MAE는 CNN **0.096338 mV**, U-Net **0.098139 mV**이며, 저장 지표 기준 CNN이 낮다. 이는 테스트 성능 또는 통계적 유의성에 관한 결론이 아니다.

## 검증한 내용
- CNN/U-Net 공통 ECG ID·시드 42/43/44·출력 11유도 일치, 입력 Lead I 제외, 중복 평가 행 부재
- 저장된 기록별 MAE/MSE/RMSE 수학적 일관성 및 시드별 집계 재계산
- 공유 cohort 파일의 998개 ECG, train/validation/test = 798/100/100, patient_id 교차 분할 부재
- 11개 유도별 비교 및 v3 다중 라벨 기반 질환군별 오차 독립 재집계, 제공된 요약표와 대조

## 재현 방법
Python 3, `pandas`, `numpy` 필요. 이 폴더와 원본 GitHub ZIP을 같은 위치에 놓고 아래 명령 실행:

```bash
pip install pandas numpy
python run_all.py "ecg-disease-prediction-main (8)(1).zip"
```

원본 ZIP의 이름이 다르면 실제 파일명으로 변경한다. `run_all.py`가 3개 검증 스크립트를 순서대로 실행하고 `output/` CSV를 갱신한다. 원본 ZIP 자체는 이 제출 패키지에 포함하지 않는다. 저장된 CSV에서 재계산한 결과이며 모델 학습/추론은 수행하지 않는다.

## 산출물
- `run_all.py`: 공통 실행 진입점
- `verify_saved_metrics.py`: 시드별 기록 오차 재계산
- `verify_cohort.py`: 평가 ID·환자 분할 점검
- `verify_detailed.py`: 유도별·질환군별 독립 집계 및 원표 대조
- `independent_seed_metrics.csv`, `output/*.csv`: 검증 기록과 비교 결과

## 한계 및 후속 작업
1. **독립 test 100건의 성능은 미평가**: validation 결과를 test 성능이라고 표기하면 안 된다.
2. **원본 ECG와 U-Net 예측 배열·가중치가 없어 실제 추론 재현 불가**: 기록별 오차표 자체의 진위 또는 실제 파형과의 차이를 직접 확인하지 못했다.
3. 공유 cohort/라벨과 결과 CSV의 내부 일관성 검증이며, 원자료 수집·전처리 단계까지 독립 검증한 것은 아니다.
4. 500Hz는 팀 결정으로 평가 범위에서 제외한다.

자료 출처: 사용자가 제공한 `ecg-disease-prediction-main (8)(1).zip`의 `ECG_role2_100Hz_v3_baseline_20261111/` 및 `ECG_role3_UNet_100Hz_20261113/`.
