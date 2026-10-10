# ECG 기반 심장질환 예측 시스템

## 프로젝트 소개
스마트워치에서 측정한 1-Lead ECG 데이터를 활용하여 심장질환 예측 및 분석 시스템을 개발하는 캡스톤디자인 프로젝트입니다.

## 주요 개발 목표
- 1-Lead ECG 데이터를 활용한 12-Lead ECG 재구성 연구
- ECG 데이터 전처리 및 분석
- 딥러닝 기반 심장질환 분류
- 예측 결과 시각화 및 분석 리포트 제공

## 개발 환경
- Python
- GitHub
- VS Code

## 프로젝트 진행 상태

- PTB-XL 기반 Lead I → 나머지 11개 유도 추정 모델 개발·평가 완료
- Baseline과 Dilated CNN을 동일 데이터 분할에서 seed 42·43·44로 비교
- seed 평균 MSE 기준 validation 약 2.53%, test 약 4.66% 감소
- 추론 프로그램과 입출력 규격, 대표 실험 결과 공유 완료
- 실제 스마트워치 입력 검증과 분류 모델 연결은 미완료

## 기존 계획 대비 현재 구현

팀의 최종 목표인 ECG 기반 심장질환 예측은 유지한다. 역할 재분배 이후 이 모듈 담당 범위를 Lead I으로 나머지 11개 유도를 추정하는 기능으로 구체화했다.

| 기존 계획 | 현재 구현 | 변경 이유 |
|---|---|---|
| Icentia11k 단일 유도 데이터 활용 | PTB-XL의 100 Hz·10초·mV 파형으로 유도 추정 | 입력 Lead I에 대응하는 11개 정답 파형이 필요 |
| 필터·R-peak·HRV 분석 | 이번 재구성 baseline은 전처리 없음 | 동일 조건에서 모델을 비교하기 위한 정책이며, 전처리 자체가 불필요하다는 뜻은 아님 |
| ECG 기반 질환 예측을 위한 모델 개발 계획 | 기본 CNN과 Dilated CNN 비교: 파라미터 31,819개 유지, 수용영역 15→31 samples | 유도 추정에서 수용영역 확대 효과를 비교 |
| 기존 발표 모델과 현재 실험의 성과는 별개 | 이번 Baseline 대비 seed 평균 MSE 감소: validation 약 2.53%, test 약 4.66% | 파형 추정 오차 비교이며 기존 발표 모델 대비 개선이나 질환 분류 정확도가 아님 |
| 위험도 추적·AI 판단 근거·과거 기록 비교·자동 리포트 | 현재 재구성 모듈의 완료 기능에 포함하지 않음 | 팀 계획과 이 모듈의 구현 범위를 구분; 팀원의 현재 진행 상태는 미확인 |

실제 스마트워치 적용과 분류 연결은 미완료다. 기존 계획은 작성자가 전달한 발표 내용 기준이며, 발표자료 원본 대조 여부와 저장된 실험 근거는 [상세 비교](ecg_reconstruction/EXPERIMENT_SUMMARY.md#11-기존-계획-대비-현재-구현)에 구분해 기록했다.

---

## ECG 재구성 모듈 공유 안내

현재 공유한 모듈은 [ecg_reconstruction/](ecg_reconstruction/README.md)의 Lead I 기반 ECG 유도 추정 코드다. PTB-XL의 100 Hz, 10초 Lead I으로 나머지 11개 lead를 추정한다. 출력의 Lead I은 입력이며 나머지 11개는 추정값이다. 분류 모델과의 연결은 아직 완료하지 않았다.

### 먼저 읽을 문서

- [재구성 모듈의 설치·데이터 준비·학습·평가·추론 안내](ecg_reconstruction/README.md)
- [저장된 validation·최종 test 결과와 한계](ecg_reconstruction/EXPERIMENT_SUMMARY.md)
- [입출력 규격과 분류 담당자에게 확인할 항목](ecg_reconstruction/ECG_INTERFACE.md)
- [대표 CSV·JSON·PNG](ecg_reconstruction/reports/representative/README.md)
- [데이터·체크포인트 별도 전달과 평가 준비](ecg_reconstruction/docs/CHECKPOINT_HANDOFF.md)
- [복사·검사 내역](ecg_reconstruction/SHARING_REVIEW.md)
- [공유 파일 전체 목록](SHARE_FILES.txt), [원본 대비 SHA-256 목록](SHARE_MANIFEST.json)

### 실행 위치와 환경

저장소 루트에서 아래처럼 이동한 뒤 실행한다. 환경 생성·설치는 안내이며 공유 복사 과정에서 실행하지 않았다.

```powershell
Set-Location .\ecg_reconstruction
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

직접 사용하는 NumPy, pandas, Matplotlib, WFDB, PyTorch 버전은 `requirements.txt`에 기록했다. 확인한 원래 환경과 새 환경 설치 검증의 한계는 [ENVIRONMENT.md](ecg_reconstruction/ENVIRONMENT.md)에 있다. 데이터·결과·체크포인트의 상대 경로는 아래 명령들에서 모두 `ecg_reconstruction/` 기준이다.

### 데이터·체크포인트는 별도 준비

원본 WFDB, metadata, 가공 NPZ, 입력 NPY, 모델 가중치는 이 공유본에 포함하지 않았다. `data/`, `checkpoints/`, `examples/`와 실행 시 생성되는 `results/`는 `ecg_reconstruction/` 아래에 준비하거나 생성한다.

새 데이터를 준비하는 명령은 다음과 같다. 네트워크 다운로드가 발생할 수 있으며 이번 복사 작업에서는 실행하지 않았다.

```powershell
.\.venv\Scripts\python.exe -B .\prepare_dataset_1000.py
.\.venv\Scripts\python.exe -B .\inspect_dataset_1000.py --seed 42
```

**기존 여섯 체크포인트를 평가할 때는 바이트가 동일한 기존 NPZ가 필요하다.** 재생성한 파일은 값이 같아도 SHA-256이 달라 검증에 실패할 수 있다. 원본 보관 담당자에게 NPZ와 필요한 체크포인트를 별도 채널로 받아 [전달 안내](ecg_reconstruction/docs/CHECKPOINT_HANDOFF.md)의 경로·해시를 확인한다. 대응하는 여섯 학습 이력 CSV는 공유 결과에 포함했으며, 평가 시 원래 `results/` 경로에 별도로 복원해야 한다. 기존 파일은 덮어쓰지 않는다. 코드의 데이터·구조·소스 해시 검증을 우회하지 않는다.

학습·validation·최종 test 명령은 [모듈 README](ecg_reconstruction/README.md)에 있다. 단일 CPU 추론은 다음처럼 실행한다. 입력 경로는 직접 준비해야 하는 자리 표시자이며, 체크포인트도 별도 전달받은 뒤 실행한다.

```powershell
.\.venv\Scripts\python.exe -B .\predict_ecg.py `
  --checkpoint .\checkpoints\dilated_1000\run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636\best_model.pth `
  --input .\examples\your_lead_i.npy `
  --sampling-rate-hz 100 `
  --unit mV `
  --device cpu
```

입력 shape는 `(1000,)` 또는 `(1000,1)`, 단위 mV, sampling rate 100 Hz다. 자동 리샘플링·정규화·필터링·clipping을 하지 않는다. 지원하지 않는 입력은 오류로 종료한다.

### 공유 범위와 결과 해석

실행에 필요한 로컬 import, 데이터 준비 코드, 테스트, 의존성·설명 문서와 선별 결과를 포함했다. Python 소스는 원본과 바이트 단위로 같으며, 모듈의 `.gitattributes`로 Git 줄바꿈 변환을 막는다. 대표 결과의 해시도 유지한다. 원본 `results/` 전체, 가상환경, 캐시, 가중치와 데이터는 제외했다. `.gitignore`는 이미 추적된 파일을 해제하지 않으므로 향후 staging 전에 추적 목록도 확인한다.

저장된 validation과 test에서 세 seed 모두 Dilated CNN의 overall MSE·MAE·RMSE가 낮았다. 세 seed는 동일 분할의 반복 실험이며 같은 test 100개를 평가했다. 독립 test 300개가 아니다. 기존 seed 42 Baseline test 결과를 최종 비교 전에 이미 확인한 사실도 유지한다. 통계적 유의성이나 임상적 유효성을 주장하지 않으며, 스마트워치 검증과 분류 모델 연결은 미완료다.

2026-10-11 원격 main의 기존 이력과 파일을 확인하고 로컬 main을 origin/main에 연결했다. 원격 프로젝트 소개와 목표는 이 문서 앞부분에 보존하고 재구성 모듈 안내를 통합했다. 충돌하던 README·.gitignore와 준비한 공유 파일 전체는 연결 전에 저장소 밖에 백업했다. `git add`, commit, push, 학습·모델 추론은 실행하지 않았다.
