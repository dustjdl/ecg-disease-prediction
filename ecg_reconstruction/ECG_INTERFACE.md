# ECG 추정 프로그램 입출력 규격

팀 공유 안내: 아래의 examples/results/checkpoints 경로는 기존 로컬 실행의 이력이다. 원본 데이터·NPY/NPZ 파형·체크포인트는 저장소에서 제외한다. 공유용 [추론 메타데이터](reports/representative/results/ecg_predictions/run_20261011_000527_332f77c10d404360af25b91935f36c4a/prediction_metadata.json), [출력 NPZ 규격](reports/representative/prediction_output_schema.json), [예제 출처](reports/representative/examples/lead_i_20261011_000435_36e7c5eb5ccb4a7b942f56220c9de992/source.json)는 별도로 포함했다. 새 입력 준비와 체크포인트 전달은 [README](README.md)와 [전달 안내](docs/CHECKPOINT_HANDOFF.md)를 참고한다. 원본 실행 파일과 결과는 로컬에 유지했다.

확인일: 2026-10-11. 아래 현재 규격은 `predict_ecg.py`, 실제 CPU 실행 결과의 NPZ·JSON, 예제 `source.json`을 직접 확인한 내용이다. 문서 작성 중 모델 추론을 다시 실행하지 않았다. 분류 모델의 입력 조건은 아직 확인되지 않았으며, 마지막 절의 항목은 앞으로 합의해야 한다.

## 현재 프로그램의 입력

단일 기록의 Lead I을 NumPy `.npy` 파일로 전달한다. batch 축이나 다른 lead를 포함하지 않는다.

| 항목 | 지원 규격 |
|---|---|
| shape | `(1000,)` 또는 `(1000, 1)` |
| 원본 dtype | 실수형 부동소수점 또는 정수형 NumPy 배열 |
| 내부 dtype | 유한값 검사 후 `float32`로 변환하고 다시 유한값 검사 |
| sampling rate | **100 Hz**. `--sampling-rate-hz 100` 필수 |
| 기록 길이 | 1,000 samples, **10초** |
| 전압 단위 | **mV**. `--unit mV` 필수, 대소문자 구분 |
| 입력 lead | **I**. 호출자가 해당 파형이 Lead I임을 보장해야 함 |

`.npy` 자체에는 주파수·단위·lead 이름이 포함되지 않는다. CLI 값은 호출자의 선언이며, 프로그램이 파형만으로 실제 주파수·단위·측정 lead를 판별하지는 않는다. 현재 예제는 별도 `source.json`으로 출처를 확인할 수 있다.

프로그램은 `allow_pickle=False`로 읽는다. bool, complex, 문자열, object 배열, 지원하지 않는 shape·길이·주파수·단위, NaN/inf를 거부한다. 원본 값이 유한해도 `float32` 변환 중 overflow로 비유한값이 되면 오류로 종료한다. float64나 정수 입력은 float32 변환으로 정밀도가 달라질 수 있다.

자동 리샘플링, 단위 환산, 정규화, 표준화, 필터링, clipping, 증강을 하지 않는다. 수행하는 입력 처리는 float32 변환과 reshape뿐이다. 모델 입력 텐서는 `[1, 1, 1000]`이다.

## 현재 프로그램의 출력

매 실행 프로젝트의 `results/ecg_predictions/run_<날짜시각>_<UUID>/`에 새 폴더를 만들고 아래 파일을 저장한다. 기존 실행 결과를 덮어쓰는 옵션은 없다.

- `estimated_12lead.npz`: 수치 파형과 lead별 추정 여부.
- `estimated_12lead.png`: 초(s)·mV 축을 가진 12개 lead 그래프. I은 `Input Lead I`, 나머지는 `Estimated <lead>`로 표시.
- `prediction_metadata.json`: 입력·체크포인트 경로와 SHA-256, 모델 클래스·소스 해시·계층 구조, 입력 변환, 전처리 없음, 출력 규격, 실행 환경 등을 기록.

실제 생성된 NPZ를 `allow_pickle=False`로 읽어 확인한 키는 다음 **5개**다.

| 키 | shape | 실제 dtype | 의미 / 값 |
|---|---|---|---|
| `signal` | `(1000, 12)` | `float32` | 입력 Lead I과 예측 11개 lead를 합친 유한한 mV 파형 |
| `lead_names` | `(12,)` | `<U4` | 열 순서에 대응하는 Unicode lead 이름 |
| `sampling_rate_hz` | `()` | `int64` | 스칼라 `100` |
| `voltage_unit` | `()` | `<U2` | 스칼라 문자열 `mV` |
| `is_estimated_lead` | `(12,)` | `bool` | 각 열이 모델 추정값인지 표시 |

`()`는 0차원 NumPy 스칼라 배열이며 `.item()`으로 Python 값으로 읽을 수 있다. `<U4`, `<U2`는 이번 파일에서 확인한 Unicode dtype 표기다.

### Lead 순서와 배열 방향

```text
열 번호:   0   1    2    3    4    5   6   7   8   9  10  11
lead:      I  II  III  AVR  AVL  AVF  V1  V2  V3  V4  V5  V6
추정 여부: F   T    T    T    T    T   T   T   T   T   T   T
```

`signal`의 **axis 0은 시간**, **axis 1은 lead**다. `signal[t, c]`는 sample index `t`에서 lead `c`의 전압(mV)이며, `signal[:, 0]`은 Lead I 전체 파형이다. 저장 파일에는 batch 축이 없다. 프로그램은 분류 모델용 transpose나 batch 축 추가를 하지 않는다.

샘플 시간은 `t / 100`초다. index 0~999의 시각은 0.00~9.99초이며, 100 Hz의 1,000샘플 기록을 10초 길이로 취급한다.

Lead I은 **유효성 검사와 float32 변환을 마친 입력과 비트 단위로 동일하게 보존**된다. 나머지 11개 lead는 모델 추정값이다. 따라서 이 파일은 12개 모두 직접 측정한 파형이 아니다. float64 원본과 변환 후 Lead I의 원래 정밀도까지 같다는 의미는 아니다.

NPZ에 정답 Y, 성능 지표, ecg_id, patient_id는 포함되지 않는다. 현재 예제의 기록 ID는 예제 폴더의 `source.json`에서 확인한다. 프로그램은 정답 비교나 분류를 수행하지 않는다.

## 실제 확인한 예제와 CPU 실행

아래 경로는 원본 `ecg-capstone`에서 실행한 이력이며 팀 공유본에서는 `ecg_reconstruction/` 기준으로 같은 상대 경로를 사용한다. NPY·NPZ·체크포인트와 원본 결과 폴더는 복사하지 않았으므로 실행하려면 별도로 준비해야 한다. 공유된 출처 JSON·추론 JSON·PNG와 NPZ 규격은 문서 첫머리의 링크로 확인한다.

| 항목 | 실제 경로 |
|---|---|
| 프로그램 | `predict_ecg.py` |
| 입력 예제 | `examples/lead_i_20261011_000435_36e7c5eb5ccb4a7b942f56220c9de992/lead_i_example.npy` |
| 예제 출처 | `examples/lead_i_20261011_000435_36e7c5eb5ccb4a7b942f56220c9de992/source.json` |
| 체크포인트 | `checkpoints/dilated_1000/run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636/best_model.pth` |
| 확인한 결과 폴더 | `results/ecg_predictions/run_20261011_000527_332f77c10d404360af25b91935f36c4a/` |

예제는 `data/processed/ptbxl_100hz_1000records.npz`의 원본 순서상 첫 validation 기록이다. 원본 index는 **800(0부터 시작)**, ecg_id는 **8**, patient_id는 **11275**다. 입력은 X만 추출한 `(1000, 1)`, float32, Lead I이며 Y는 예제에 포함하지 않았다. test 기록을 사용하지 않았다.

실제 결과 JSON에는 모델 `model_dilated.ECGDilatedModel`, checkpoint epoch **28**, 실행 장치 **cpu**가 기록돼 있다. 이 모델의 dilation은 1/2/4/1이다. 프로그램은 메타데이터가 검증되는 `model.ECGModel` baseline도 지원하며, 구조가 없거나 불일치하는 체크포인트는 추측해서 로드하지 않는다.

확인한 SHA-256:

| 대상 | SHA-256 |
|---|---|
| 원본 데이터 (`source.json` 기록) | `2828f1949a6c5205de243cf304afc47faabc52d55021c7b9120f19d6309662d6` |
| 입력 예제 | `2e0330eedf5906f1ef8ea3819c813abbdeb67ec35d5e991dfd61b00ffa58c850` |
| 체크포인트 | `e6cfd066ab8c88f1e37d5635beff7df2c3606509d0e1455392dab6642d8c6a64` |

입력·체크포인트·프로그램의 실제 파일 해시가 결과 JSON 기록과 일치하고, 저장된 Lead I이 입력과 비트 단위로 일치함을 읽기 검사로 확인했다.

PowerShell에서 팀 저장소의 `ecg_reconstruction/` 폴더를 작업 디렉터리로 실행하는 명령은 다음과 같다. 아래 입력 예제와 체크포인트는 공유본에서 제외했으므로 원본 보관 담당자에게 별도로 받거나 지원 규격의 새 입력을 준비해야 한다. 재실행하면 새 결과 폴더가 만들어지며, 위에 기재한 기존 결과 폴더를 갱신하지 않는다.

```powershell
.\.venv\Scripts\python.exe predict_ecg.py `
  --checkpoint "checkpoints/dilated_1000/run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636/best_model.pth" `
  --input "examples/lead_i_20261011_000435_36e7c5eb5ccb4a7b942f56220c9de992/lead_i_example.npy" `
  --sampling-rate-hz 100 `
  --unit mV `
  --device cpu
```

`--checkpoint`, `--input`, `--sampling-rate-hz`, `--unit`은 필수다. `--device`는 `cpu`, `cuda`, `auto`를 지원하며 기본값은 `auto`다. CUDA를 명시했는데 사용할 수 없으면 오류로 종료한다. 모델 실행은 `eval()` 및 `inference_mode()`를 사용한다.

## NumPy로 실제 출력 읽기

아래 예제는 이미 저장된 출력만 읽으며 모델을 실행하지 않는다. 작업 디렉터리는 `ecg-capstone`이다.

```python
from pathlib import Path
import numpy as np

path = Path(
    "results/ecg_predictions/"
    "run_20261011_000527_332f77c10d404360af25b91935f36c4a/"
    "estimated_12lead.npz"
)
with np.load(path, allow_pickle=False) as data:
    signal = data["signal"].copy()               # (time, lead) = (1000, 12)
    leads = data["lead_names"].tolist()
    fs = int(data["sampling_rate_hz"].item())
    unit = str(data["voltage_unit"].item())
    estimated = data["is_estimated_lead"].copy()

assert signal.shape == (1000, 12) and signal.dtype == np.float32
assert np.isfinite(signal).all()
assert fs == 100 and unit == "mV"
assert estimated.tolist() == [False] + [True] * 11
time_s = np.arange(signal.shape[0]) / fs
lead_i = signal[:, leads.index("I")]            # 저장된 입력 Lead I
```

마지막 줄은 저장된 입력 Lead I을 가져온다. 추정 lead를 가져오는 예는 `signal[:, leads.index("V2")]`이다. 여기서는 분류 모델을 위한 배열 변환을 하지 않는다.

## 현재 검증 범위

현재 학습·validation 비교·실제 입력 예제의 확인 범위는 **PTB-XL**이다. 이번 단일 기록은 PTB-XL validation에서 가져왔으며 추론 프로그램의 입출력 확인에 사용했다. 단일 기록의 저장 성공만으로 다른 입력원에 대한 정확도를 입증하지는 않는다.

**실제 스마트워치 측정에 대한 적용은 미검증**이다. 측정 lead, 샘플링 주파수, 단위·교정, 기기 측 전처리 등이 현재 입력 조건과 같은지는 확인되지 않았다. 프로그램은 이를 자동 보정하지 않는다.

## 분류 담당자와 앞으로 합의할 사항 — 미확정

분류 모델의 입력 조건은 아직 제시되지 않았으며, 현재 출력을 그대로 사용할 수 있는지는 미확인이다. 다음 항목을 담당자에게 확인한 뒤 연결 규격을 결정한다.

| 확인 항목 | 현재 추론 프로그램 | 분류 담당자에게 확인할 내용 |
|---|---|---|
| 샘플링 주파수 | 100 Hz | 필요한 Hz, 다를 경우 변환 담당과 방법 |
| 길이 | 1,000 samples, 10초 | 필요한 샘플 수·초 길이, 구간 자르기·padding 조건 |
| 단위 | mV, 물리 전압 | mV/µV/정규화 값 중 어느 것인지, 스케일·교정 조건 |
| Lead 구성·순서 | I, II, III, AVR, AVL, AVF, V1~V6 | 필요한 lead와 순서, 이름 표기, 추정 11개 lead를 사용할 수 있는지 |
| 배열 방향·dtype | `(time, lead)=(1000,12)`, float32, batch 축 없음 | `(time,lead)` 또는 `(lead,time)`, batch 축과 dtype 요구사항 |
| 전처리 | float32 변환·reshape만 수행 | 필터, 기준선 처리, 정규화 등의 필요 여부, 학습 시 조건, 처리 담당 |

이 문서에서는 분류 모델에 맞춘 리샘플링, 정규화, 단위 변환, 순서 변경, transpose를 지정하거나 수행하지 않았다. 앞으로 필요한 변환은 담당자와 합의한 별도 연결 규격에 명시한다.
