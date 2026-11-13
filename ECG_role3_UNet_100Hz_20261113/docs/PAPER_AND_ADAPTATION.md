# 논문 구조와 이번 구현의 구분

출처: Presacan et al. (2025), Evaluating the feasibility of 12-lead electrocardiogram reconstruction from limited leads using deep learning. Communications Medicine 5, 139.
https://doi.org/10.1038/s43856-025-00814-w
공개 코드: https://github.com/simulamet-host/ecg_1_to_8
확인한 공개 코드 commit: e58de3b6ef63c7bb9ea9e6216a872840079e284c
원본 networks.py 및 MIT 라이선스를 reference/에 보존했습니다.

## 논문에서 확인한 내용

논문은 GAN과 MSE 기반 순수 U-Net을 구분합니다. 본문 중심 결과는 GAN이며 U-Net 결과는 부록 Table S1에 있습니다. 이번 작업은 판별기나 GAN loss를 사용하지 않는 순수 U-Net 실험입니다. 논문은 정상 ECG 9,514명, 500Hz 10초 기록을 사용하지만, 본 작업은 팀의 100Hz 998개 혼합 진단군과 기존 환자 분할을 그대로 사용합니다. 논문 성능 수치와 직접 우열을 비교하지 않습니다.

본문은 6개 인코더 Conv1d, LeakyReLU, 6개 디코더 합성곱, ReLU, skip connection을 명시합니다. 그림 1에는 reflection padding, upsampling, Tanh가 표시됩니다. 본문은 U-Net MSE, lr 0.00005, batch 32, 1,000 epoch, validation 기반 모델 선택을 기술합니다. GAN Adam beta=(0.5,0.9)는 명시되어 있지만 본문만으로 U-Net beta까지 동일하다고 단정하지 않습니다. 공개 main.py의 기본 epoch 500 / README 예시 200은 논문의 1,000 epoch와 다릅니다. 이 값들을 혼합하여 논문 재현으로 주장하지 않습니다.

## 공개 코드에서 확인한 구조 상세

아래 숫자는 논문 본문에 모두 명시된 값이 아니라 공개 networks.py의 구현/기본값입니다.

| 항목 | 공개 코드 | 이번 구현 |
|---|---|---|
| 입력 | 1×5000 또는 2×5000 | 1×1000 (Lead I, 100Hz, 10초) |
| 출력 | 단일 입력에서 II,V1–V6 7유도 | II,III,AVR,AVL,AVF,V1–V6 11유도 직접 예측 |
| 채널 폭 | 32,64,128,256,512,512 | 동일 |
| 인코더 stride | 2,2,2,5,5,5 | 동일 |
| 커널 | 25 | 5: 25/500=5/100=0.05초를 기준으로 조정 |
| padding | kernel//2, reflect | 동일 규칙 (2) |
| 인코더 활성화 | F.leaky_relu 기본 slope 0.01 | 동일 |
| 업샘플링 | linear, align_corners=True | 동일 |
| 디코더 배율 | 5,5,5,2,2,2 | 동일 |
| skip | conv5→decoder2, conv4→decoder3, conv3→decoder4, conv2→decoder5, concat | 동일; conv1 skip은 원본에도 없음 |
| dropout | 0.1, encoder6/decoder1/decoder2 | 동일 |
| Conv 초기화 | kaiming_normal_, bias는 PyTorch 기본 | 동일 |
| 마지막 활성화 | tanh | linear (범위 제한 없음) |

입력 길이 변화: 1000 → 500 → 250 → 125 → 25 → 5 → 1.
디코더 길이: 1 → 5 → 25 → 125 → 250 → 500 → 1000.
모델 파라미터 수: 5,237,067.

## 이번 실험에서 정한 설정 및 이유

- 100Hz에서 kernel=25를 그대로 적용하면 깊은 인코더에서 길이 5에 reflection padding 12를 줄 수 없어 실행 오류가 납니다. kernel=5는 이를 해결하고 첫 층 커널의 시간 폭도 유지합니다. 모든 층의 유효 수용영역까지 원 논문과 완전히 같다는 뜻은 아닙니다. 논문 병목 길이 5와 달리 이번 병목은 길이 1입니다.
- 출력 11개는 사용자 요구 및 기존 CNN의 target과 일치시키기 위한 변경입니다. 논문처럼 7개를 예측한 뒤 사지유도를 공식으로 계산하지 않습니다.
- 원본 코드에는 /8 및 clipping 기반 정규화 경로가 있고 마지막 tanh는 [-1,1]로 제한합니다. 이번 CNN은 원 mV 값을 그대로 사용하므로 같은 전처리를 유지하며 마지막 활성화를 linear로 변경했습니다. 그렇지 않으면 ±1mV를 넘는 파형을 구조적으로 재구성할 수 없습니다.
- 학습 설정은 CNN에 맞춰 Adam lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0, batch=8, 30 epoch, seed=42/43/44, 전체 11유도 MSE 평균으로 고정했습니다. scheduler/early stopping/augmentation/추가 정규화는 없습니다.
- 최저 validation MSE(엄격한 감소) 모델을 선택합니다. 테스트 100개는 최종 팀 실험을 위해 평가하지 않습니다. 본 비교는 모델 선택에 사용된 validation 100개의 탐색적 비교입니다.
- 구조가 다르므로 파라미터 수와 FLOPs/학습시간은 같지 않습니다. 동일한 데이터 노출, epoch, batch, optimizer 및 선택 규칙에 따른 비교이며, 각 모델의 최적 하이퍼파라미터를 찾은 비교는 아닙니다. 특히 논문보다 큰 학습률이 U-Net의 최적 성능을 제한할 수 있습니다. 이번 결과를 본 뒤 추가 튜닝하지 않습니다.
- 논문은 재구성 파형의 개인별 특성 및 임상적 정보 손실을 지적합니다. 본 MSE/MAE/RMSE 결과는 진단 정확도나 임상적 사용 가능성을 입증하지 않습니다.
