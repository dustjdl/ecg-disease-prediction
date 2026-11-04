# ECG 3번: 데이터 복구 및 v2 분류 실험

최종 실행 상태·실제 수치·해석은 REPORT.md를 먼저 확인하세요.

## 폴더

- preserved/role3/: 기존 분류 코드·모델·성능 보존본. 덮어쓰지 않습니다.
- preserved/repository/: 제공된 GitHub 프로젝트 ZIP 원본.
- preserved/labels_v1/, labels_v2/: 제공된 라벨 원본.
- preserved/role2_reference/: 제공된 분석 결과 원본, 재구성 모델 개발을 하지 않습니다.
- recovery/: 공식 데이터로 복구한 NPZ와 메타데이터. raw WFDB 캐시는 배포 크기를 줄이기 위해 제외할 수 있습니다.
- scripts/: 신규 복구·검사·학습·비교 코드.
- working/: 원래 분류 Python 소스의 작업 사본. 원본 보존본과 내용 동일.
- runs/v2/: 실제 새 학습의 가중치·선택·이력·정답·확률·평가 결과.
- reports/: 실제 실행 로그, 기존 지표 재검산, 공통 정답 비교, 그래프 및 검증 보고서.

## 환경 설치

이 README 폴더에서 PowerShell 또는 터미널을 엽니다. Python 3.12를 사용했습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-recovered-cpu.txt
```

실제 사용한 버전은 reports/environment.json에 기록합니다. 설치 기록은 포함하지만 새 빈 가상환경 전체 설치는 따로 재시험하지 않았습니다.

## 포함된 복구 데이터로 새 분류 실험 실행

기존 runs/v2/를 덮어쓰지 않도록 새 출력 폴더를 지정합니다.

```powershell
python scripts/audit_attachments.py
python -u scripts/train_recovered.py --policy v2 --output runs/v2_again
```

v1 조건을 새 환경에서 재학습하려면 `--policy v1 --output runs/v1_again`을 사용합니다. 이번 작업에서는 같은 CPU 조건의 v1 대조군도 실제 실행했으며 runs/v1/에 보존했습니다. 원래 함수를 이용한 학습은 두 task×세 입력×세 seed, 총18개 모델과40 epoch입니다. 전체 모델 선택을 고정한 뒤 test 평가합니다.

## 복구 데이터를 처음부터 재생성

PTB-XL v1.0.3 공식 출처: https://physionet.org/content/ptb-xl/1.0.3/
공식 파일 endpoint는 이번 실행 중 physionet-open.s3.amazonaws.com으로 redirect되었습니다. 다운로드 코드는 이 공식 목적지를 사용합니다.

`recovery/`에 기존 출력이 있으면 먼저 별도 작업 사본을 만들고, 그 사본의 `recovery/processed/` 폴더와 `recovery/ecg_for_classification.npz`를 다른 위치에 보관한 뒤 실행하세요. 원래 전달 ZIP과 preserved/는 그대로 유지합니다. `build_recovered.py`는 기존 NPZ를 덮어쓰지 않습니다. 제공된 메타데이터는 SHA256 목록에서 확인할 수 있습니다.

```powershell
python -u scripts/download_locked.py
python -u scripts/build_recovered.py
```

새 기록 선정은 하지 않습니다. 제공받은 기존 1,000개 ID를 고정하고, 원래 함수의 파형 검사와 저장 구조를 재사용합니다. 원래 준비 코드는 캐시 우선 선택 때문에 빈 캐시에서 선정 결과가 달라질 수 있습니다. 421·710은1,000개 복구 NPZ에 유지하고 분류 입력에서만 제외합니다.

## 이번 완료 실험의 검증과 비교 재실행

```powershell
python scripts/verify_completed.py
python scripts/compare_and_verify.py
python scripts/verify_completed.py --policy v1
python scripts/compare_cpu_control.py
```

두 명령은 기본 runs/v2/의 실제 완료 결과를 대상으로 합니다. 다른 출력 폴더에 자동 대응하지 않습니다. reports/의 파생 표·그래프를 재작성하며 preserved/와 학습 결과를 수정하지 않습니다.

비교는 (1) 과거 v1 정답 성능과의 참고 비교, (2) 보존 v1 가중치를 복구 입력으로 다시 추론하여 동일 v2 정답에 평가한 비교를 구분합니다. 후자가 라벨 정의가 같은 비교입니다. 이전 GPU 학습과 현재 CPU 학습의 환경 차이는 별도로 고려해야 합니다.

주 비교표는 comparison_same_cpu_common_v2_*.csv이며, v1/v2 모두 동일 CPU에서 재학습하여 같은 v2 정답에 평가했습니다. 과거 보존 가중치와의 비교는 별도 참고 결과입니다.
