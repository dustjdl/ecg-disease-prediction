# 2번: 100Hz 기존 기본 CNN 기준 실험

## 범위
- 기존 model.py의 기본 CNN을 수정하지 않고 사용합니다. Dilated CNN이나 U-Net 실험이 아닙니다.
- 실제 Lead I → 실제 나머지 11유도를 학습합니다. 저장된 재구성 파형은 정답으로 사용하지 않습니다.
- v3 진단 라벨은 질환군별 검증 오차를 묶는 데만 쓰며 재구성 손실에 넣지 않습니다. v3 변경으로 재구성 성능이 좋아졌다는 실험이 아닙니다.
- 100Hz, 10초, ECG 998개(학습 798/검증 100/테스트 100), 기존 환자 분할 유지.
- 미분류 8개도 재구성 대상에 유지합니다. 검토 대상 38개를 자동 제외하지 않습니다.
- seed 42/43/44, 각 30 epoch, batch 8, Adam lr=0.001, MSE, 전처리 없음. 기존 기본 CNN 설정을 기준으로 실행했습니다. 공통 U-Net/500Hz 비교 프로토콜 확정 전의 탐색적 기준 실험입니다.
- 최저 검증 MSE로 체크포인트를 선택합니다. 테스트 추론 및 성능 평가는 하지 않습니다.
- 500Hz 원본 신호는 포함되어 있지 않으며 500Hz 비교 결과도 없습니다.

## 실행
Python 3.11 또는 3.12에서 필요한 패키지를 설치합니다.

```powershell
python -m pip install -r requirements.txt
python code/run_baseline.py --npz data/ecg_actual_100hz_998.npz --labels labels/ptbxl_100hz_v3_labels_998.csv --output results/my_new_run --validate-only
python code/run_baseline.py --npz data/ecg_actual_100hz_998.npz --labels labels/ptbxl_100hz_v3_labels_998.csv --output results/my_new_run
```

기존 `ecg_for_classification.npz`도 입력으로 지원합니다. 실제 파형과 메타데이터만 사용합니다. 결과 폴더가 이미 있으면 덮어쓰지 않고 종료하므로 새 경로를 지정하세요.

## 지표
관측 I을 제외한 11유도에서 계산합니다. MAE 단위 mV, MSE 단위 mV². pooled RMSE는 전체 평균 MSE의 제곱근이며, 기록별 RMSE 평균과 다릅니다. 요약표의 표준편차는 3 seed 사이의 변동으로, 환자 간 변동이나 신뢰구간이 아닙니다. 진단군은 중복될 수 있고 미분류는 별도 그룹으로 보고합니다.

## 재현 및 제한
환경 버전/입력 SHA-256/설정은 results/baseline/experiment_config.json에 기록합니다. 플랫폼·PyTorch 버전 차이로 과거 저장 결과와 완전히 같지 않을 수 있습니다. 기존 Dilated 성능과 직접 개선율을 계산하지 마세요. 모델 선택에 사용한 검증 데이터의 결과이므로 최종 일반화 성능으로 해석하지 않습니다. 학습 CPU 시간은 이 실행 환경에만 해당합니다.

## 후속 비교 시 주의
500Hz는 원본 신호를 사용하고 같은 ECG·분할로 연결해야 합니다. 같은 커널 크기의 CNN은 100Hz/500Hz에서 시간상 수용영역이 달라집니다. 이번처럼 구조를 고정한 비교와 시간 범위를 맞춘 추가 비교를 구분하세요. U-Net과의 학습 예산·선택 규칙 및 서로 다른 주파수의 평가 방식을 실행 전 합의해야 합니다.

## 원본 문서
labels/에 제공받은 라벨 및 정책 문서를 보존했습니다. 원본 정책 문서에 나열되었지만 이 ZIP에는 제공되지 않은 파일도 있으므로 실제 파일 목록을 기준으로 사용하세요. 라벨 정책은 연구용 제안이며 임상 검증·팀 최종 승인 완료를 의미하지 않습니다.
