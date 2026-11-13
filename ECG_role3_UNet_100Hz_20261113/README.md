# ECG_role3_UNet_100Hz_20261113

> GitHub 공유본입니다. 저장소의 기존 제외 규칙에 따라 원본 데이터(NPZ), 가중치(PTH), 예측 배열과 실행 로그는 포함하지 않습니다. 아래 실행 및 검증 명령은 `ECG_role3_UNet_100Hz_20261113.zip` 완전본에서 실행하세요. 완료된 학습 결과표·그래프·검증 기록은 이 폴더에 포함되어 있습니다.

3번 역할: Lead I → 나머지 11유도 1D U-Net, 100Hz 전용. 최종 실행 상태와 수치는 RESULTS_KO.md를 확인하세요.

## 비교 기준

GitHub: https://github.com/dustjdl/ecg-disease-prediction
최신 원격 브랜치를 확인하여 main의 92872ee (2026-11-11 18:06:38 KST), `ECG_role2_100Hz_v3_baseline_20261111/role2_100hz_v3`를 기준으로 삼았습니다. 다른 팀원의 코드나 원격 저장소는 수정하지 않았습니다.

- PTB-XL 실제 100Hz ECG 998개, 10초(1000 samples), mV.
- 기존 환자 분할 798/100/100, 환자 겹침 없음, ECG 421·710 제외 유지.
- 입력 I, target II/III/AVR/AVL/AVF/V1/V2/V3/V4/V5/V6. Lead I은 오차에서 제외.
- filtering, clipping, normalization, augmentation 없음.
- seed 42/43/44 × 30 epoch, batch 8, Adam lr=0.001, MSE, 최저 validation MSE 선택.
- validation 100개에서 비교. 테스트 추론/평가는 수행하지 않음.
- v3 진단 라벨은 그룹별 결과 분석에만 사용하며, 학습 target은 실제 ECG입니다.

## 설치 및 실행

Python 3.12 권장. ZIP을 푼 다음 이 README가 있는 폴더에서 실행합니다.

```bash
python -m pip install -r requirements.txt
python code/run_unet.py --npz data/ecg_actual_100hz_998.npz --labels labels/ptbxl_100hz_v3_labels_998.csv --output results/my_unet --validate-only
python code/run_unet.py --npz data/ecg_actual_100hz_998.npz --labels labels/ptbxl_100hz_v3_labels_998.csv --output results/my_unet
python code/compare_results.py --unet results/my_unet --cnn reference/cnn_baseline --output results/my_comparison
python code/test_integrity.py
```

기본 실행은 실제 수행 환경과 동일한 CPU 4 threads이며 GPU가 필수는 아닙니다. 학습 출력 폴더가 이미 있으면 덮어쓰지 않고 중단합니다. 같은 이름을 재사용하지 마세요. 다른 PyTorch/플랫폼 버전에서는 수치가 달라질 수 있습니다.

CNN 통제 재학습도 원본 학습 코드를 그대로 복사해 실행했습니다.

```bash
python reference/run_baseline_original.py --npz data/ecg_actual_100hz_998.npz --labels labels/ptbxl_100hz_v3_labels_998.csv --output results/my_cnn_control
```

원본 CNN 스크립트의 5000-point 무작위 텐서 shape 검사와 주석은 보존된 원본의 내용일 뿐 500Hz 데이터 처리나 학습이 아닙니다. U-Net 실험 코드는 1000-point 입력만 허용합니다.

## 폴더

- code/: U-Net 모델, 학습/비교/검증 코드
- data/: CNN 패키지의 실제 파형 전용 NPZ (예측 파형을 target으로 사용하지 않음)
- labels/: 기존 v3 라벨 CSV
- reference/: 논문 공개 코드·라이선스, 기존 CNN 코드·결과·가중치 보존
- results/unet/: seed별 가중치, 학습 이력, validation 예측 및 오차표
- results/cnn_control/: 동일 환경에서 원본 CNN을 다시 실행한 통제 결과
- results/comparison/: 모델 비교 CSV, 파형·학습곡선·유도별 오차 그래프
- docs/PAPER_AND_ADAPTATION.md: 논문 명시 사항, 공개 코드 확인 사항, 이번 변경 사항

## 지표 정의와 제한

MSE = 모든 기록·시간·예측 11유도의 squared error 평균 (mV²).
MAE = 같은 원소의 absolute error 평균 (mV).
pooled RMSE = sqrt(전체 MSE) (mV); 기록별 RMSE의 평균과 다릅니다.
유도별 지표는 기록과 시간 축을 평균합니다. seed 평균±표준편차는 3번 초기화/학습의 변동이며 환자 신뢰구간이 아닙니다.
상대 개선율=(CNN−U-Net)/CNN×100; 음수는 악화입니다. 모든 비교는 ECG ID와 유도 순서를 검증한 후 계산합니다.

원본 분류 NPZ와 실제 파형만 추린 NPZ는 압축 내용이 달라 파일 SHA가 다릅니다. 본 입력은 기존 verification.json의 packaged_npz_sha256와 일치합니다. results/input_verification.json 및 통제 재학습 비교를 참고하세요.

## 학습된 가중치로 재구성

```bash
python code/predict.py --npz data/ecg_actual_100hz_998.npz --checkpoint results/unet/seed_42/best_model.pth --output results/validation_inference.npz
```

이 명령은 검증 데이터에서 예측 11유도와 관측 Lead I을 붙인 12유도를 저장합니다. 기존 파일은 덮어쓰지 않습니다. 실제 측정 12유도와 동등한 검사 결과를 의미하지 않습니다.
