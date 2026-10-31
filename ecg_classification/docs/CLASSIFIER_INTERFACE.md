# 4번 담당 연동용 질환 분류 추론 규격

제공자: **역할 3 질환 분류 모델 개발**. 라벨 정책 자료 확인 대상: **역할 1**. 수신자: **역할 4 추론 연동**.

라벨 처리는 [이번 실험에 적용한 라벨 기준](CLASSIFICATION_LABEL_CRITERIA.md)을 따른다. 완료된 A/B/C 실험의 18개 모델을 그대로 사용하며 test 점수로 연동 모델·seed를 새로 선택하지 않는다. 아래 C·seed 42 예시는 고정된 사용 예시이며 최우수 모델 선정 결과가 아니다.

## 코드와 필수 파일

`classifier_inference.py`는 NumPy와 PyTorch만 import하며 학습 코드·sklearn·원본 ECG 데이터·정답 라벨을 읽지 않는다. 모델별로 필요한 파일은 `best_model.pth`, `thresholds.json`, `selection.json`이다. 가중치·임계값의 SHA-256을 고정된 selection과 대조한다. train 정규화 값과 task/input/seed는 checkpoint에서 읽는다. 모델별 다른 임계값을 섞으면 오류가 난다.

## 입력 NPZ

| 키 | 형식 | 의미 |
|---|---|---|
| `signal` | 실수 `[1000,C]` 또는 `[N,1000,C]` | 100 Hz·10초, 시간 축 다음 유도 축. A의 C=1, B/C의 C=12 |
| `lead_names` | 문자열 `[C]` | A는 I만, B/C는 12개 유도 이름; 이름 기준으로 훈련 순서에 재배열 |
| `sampling_rate_hz` | scalar 숫자 | 정확히 100 |
| `voltage_unit` | scalar 문자열 | 정확히 `mV` |
| `input_kind` | scalar 문자열, 조건부 | `A`, `B`, `C`; 아래 `is_estimated_lead`가 있으면 생략 가능 |
| `is_estimated_lead` | bool `[C]`, 조건부 | A/B는 모두 false; C는 I만 false, 나머지 11개 true. kind와 둘 다 주면 일치해야 함 |
| `record_ids` | `[N]`, 선택 | ID는 출력 추적용이며 모델 입력·판정·정답으로 사용하지 않음 |

표준 12유도 순서는 `I,II,III,AVR,AVL,AVF,V1,V2,V3,V4,V5,V6`이며 aVR 등 대소문자는 정규화한다. A/B/C를 파형 모양만으로 추측하지 않는다. 선언/추정 유도 플래그는 생산자가 제공한 출처 메타데이터이며 실제 생성 과정을 파형에서 증명하는 기능은 아니다.

잘못된 샘플 수, 축, 단위, Hz, 중복/누락 유도 이름, 비유한 값, A/B/C 모델과 입력 종류 불일치는 오류로 종료한다. 자동 리샘플링·필터링·진폭 교정·시간 이동은 하지 않는다. 단일 I 벡터 `[1000]`는 호출자가 `[1000,1]`로 변환해야 한다.

기존 `predict_ecg.py` 출력인 `estimated_12lead.npz`는 `signal`, `lead_names`, `sampling_rate_hz`, `voltage_unit`, `is_estimated_lead`를 포함하므로 **C 모델에 직접 연결 가능**하다. 다만 통합 실제 실행 예시는 전달 NPZ에서 추출한 파형이며 스마트워치 또는 실제 서비스 입력 검증은 아니다.

## CLI

팀 전달 폴더를 작업 디렉터리로 실행한다. 출력 파일은 새 경로만 허용한다.

```powershell
python .\classifier_inference.py `
  --model-dir .\models\multilabel\C\seed_42 `
  --input .\examples\inputs\C_test_9.npz `
  --output .\my_outputs\C_test_9.json `
  --device cpu
```

이진 모델은 `models/binary/C/seed_42`, A/B 모델은 해당 입력 폴더와 같은 종류의 입력 NPZ를 사용한다. 기존 프로젝트 루트에서는 모델 경로 앞에 `results/classification_abc/experiment_20261101/`를 붙이면 된다. 입력 파일에 실제 파형·정답을 함께 담은 원래 대용량 NPZ를 그대로 CLI에 전달하지 말고 `signal` 규격으로 준비한다.

## Python API

```python
from classifier_inference import ECGClassifierPredictor, load_input

predictor = ECGClassifierPredictor("models/multilabel/C/seed_42", device="cpu")
result = predictor.predict(**load_input("examples/inputs/C_test_9.npz"))

# 서비스 내 배열을 직접 전달하는 경우
result = predictor.predict(
    waveform,  # [1000,12] 또는 [N,1000,12], mV
    lead_names=["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    sampling_rate_hz=100,
    voltage_unit="mV",
    input_kind="C",
    record_ids=["example-001"],
)
```

배치 길이와 record_ids 길이는 같아야 한다. API는 JSON 직렬화 가능한 dict를 반환하며 파일을 쓰지 않는다. 모델을 한 번 로드한 후 반복 호출할 수 있다. `eval()`/`inference_mode()`로 dropout을 비활성화한다.

## 출력 JSON

`schema_version`, `task`, `input_kind`, `seed`, `labels`, `n_records`, `best_epoch`, 가중치·임계값 hash, `score_type`, `label_policy_status`, `records`를 포함한다. CLI는 입력 파일 hash도 추가한다.

각 record에는 순서(`record_index`), 선택적 `record_id`, `scores`, `predicted_labels`가 있다. scores의 각 원소는 `label`, sigmoid `probability`, validation `threshold`, bool `predicted`다. 판정은 **probability >= threshold**다. 예측 확률은 보정된 임상 확률이 아니다.

다중 라벨은 NORM/MI/STTC/CD/HYP의 독립적인 다섯 점수다. `predicted_labels`는 복수 또는 빈 목록일 수 있고 NORM+질환도 허용한다. 별도 규칙으로 NORM 제거/단일 라벨화/합이 1인 softmax 변환을 하지 않는다.

이진은 scores에 `ABNORMAL` 하나만 있고 `binary_decision`이 `NORMAL` 또는 `ABNORMAL`이다. 음성일 때 `predicted_labels`는 빈 목록이며 **정상 여부는 binary_decision으로 읽는다**. 다중 라벨 모델에서 NORM이 예측되었다고 이진 모델 출력을 덮어쓰지 않는다.

실제 입력·출력은 전달 폴더의 `examples/inputs/`, `examples/outputs/`에 있다. A/B/C 각 입력으로 test ID 9·65를 CPU에서 실행하고, 18개 모델의 전체 998개 예측을 이전 GPU 평가 확률·라벨과 비교한다. 비교 오차와 판정 일치 여부는 `verification/inference_reproduction.csv`에 저장한다. 원래 GPU 평가의 cudnn TF32와 batch 크기 및 CPU 연산 경로 차이를 고려해 확률의 절대 허용 오차는 1e-3으로 두고 **실측 최대 오차와 판정 불일치 수를 별도로 보고한다**. 동일 확률의 bitwise 일치는 보증하지 않으며 임계값 가까이에서는 장치별 판정 차이가 생길 수 있다. 이번 검증의 불일치가 있으면 숨기지 않고 기록한다.

검증 범위는 제공된 PTB-XL 파형과 기존 재구성 출력 포맷이다. 다른 기기·단위·유도·전처리·샘플링 속도와 임상 유효성은 검증하지 않았다.
