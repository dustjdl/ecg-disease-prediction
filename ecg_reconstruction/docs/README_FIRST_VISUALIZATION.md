# PTB-XL 12-lead ECG 첫 시각화

`plot_ptbxl_sample.py`는 PTB-XL의 100 Hz 샘플 `00001_lr`을 사용합니다. 첫 실행에서 필요한 두 파일만 `data/ptb-xl/`에 내려받고, 12개 lead를 세로로 배치한 PNG를 `output/ptbxl_00001_12lead.png`에 만듭니다.

## 실행 전 Python 확인

이 컴퓨터에서는 `py` 명령이 Microsoft Store 별칭을 가리키지만 실제 Python을 실행하지 못했습니다. VS Code에서 다음 순서로 해결하세요.

1. `Ctrl+Shift+P` → **Python: Select Interpreter**에서 실제 Python 인터프리터 또는 프로젝트의 `.venv`를 선택합니다.
2. 인터프리터가 없다면 [python.org](https://www.python.org/downloads/windows/)에서 Python 3을 설치할 때 **Add python.exe to PATH**를 선택합니다.
3. VS Code 터미널을 새로 열고 아래를 실행합니다.

```powershell
python --version
python -m pip show numpy pandas matplotlib wfdb
```

## 실행

프로젝트 폴더에서 실행합니다.

```powershell
python plot_ptbxl_sample.py
```

데이터나 결과를 다른 위치에 둘 때는 다음처럼 지정할 수 있습니다.

```powershell
python plot_ptbxl_sample.py --data-dir data/ptb-xl --output output/ptbxl_00001_12lead.png
```

같은 PNG가 이미 있으면 프로그램은 멈추며 덮어쓰지 않습니다. 결과를 확인한 뒤에만 아래처럼 명시적으로 교체할 수 있습니다.

```powershell
python plot_ptbxl_sample.py --overwrite
```

## 코드의 흐름

1. `download_sample()`은 PhysioNet에서 ECG 신호 파일(`.dat`)과 설명 파일(`.hea`)을 가져옵니다. 이미 있는 파일은 다시 받지 않습니다.
2. `load_ecg()`는 `wfdb.rdrecord()`로 두 파일을 하나의 ECG 레코드로 읽고, 신호가 정확히 12개인지 검사합니다.
3. `save_12_lead_plot()`은 각 lead(I, II, III, aVR, aVL, aVF, V1~V6)를 별도 그래프로 그리고 하나의 PNG로 저장합니다.
4. `main()`은 명령줄 옵션을 읽고 위의 세 단계를 순서대로 실행합니다.

`record.p_signal`은 ECG의 실제 신호값이 담긴 2차원 배열입니다. 행은 시간의 순서이고, 열은 lead의 순서입니다. `record.fs`는 1초 동안 기록한 샘플 수(이 샘플은 100 Hz)입니다.
