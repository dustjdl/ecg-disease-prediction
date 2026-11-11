# PTB-XL 100Hz: v3 라벨 (ECG 998개)

이 패키지는 **100Hz 전용 라벨과 검토 자료**만 포함합니다. 500Hz 자료 및 실제 `.dat` / `.hea` 파형 파일은 포함하지 않습니다.

## CSV 용도

- `ptbxl_100hz_v3_labels_998.csv`: 원본 진단 코드/원본 5대 진단군 + v3 진단 라벨을 함께 보존. 전체 998개를 사용하여 재구성 학습용 ECG ID와 연결할 때 사용.
- `ptbxl_100hz_v3_supervised_990.csv`: 5개 진단군 지도학습 대상 990개만 포함. `NORM`, `MI`, `STTC`, `CD`, `HYP`가 v3 정답.
- `label_review_v3_38.csv`: 수동 검토가 필요한 38개 목록. 임상 판독 완료를 뜻하지 않음.
- `class_distribution_v3.csv`: train/validation/test별 진단 분포.
- `LABEL_POLICY_V3.md`: 라벨 생성/유지/검토 정책.

## 핵심 기준

- 원본 `scp_codes`, 환자 ID, ECG ID 및 train/validation/test 분할을 유지합니다.
- NORM과 다른 진단군(MI/STTC/CD/HYP)이 동시에 양성이면 **v3 NORM=0**으로 처리합니다(16개).
- 선택한 5개 진단군 중 해당하는 것이 없는 기록 8개는 **지도학습에서 제외**합니다. 정상으로 취급하지 않습니다.
- likelihood 0.0은 진단 부재가 아닌 확신도 미상으로 취급하며 검토 이력을 유지합니다.
- 38개는 **검토 대상**, 의료진의 임상 확인을 완료한 라벨이 아닙니다.
- `v3_optional_strict_STTC_mask`는 선택적 민감도 분석용; 기본 v3 라벨을 임의로 지우는 필드가 아닙니다.

## 파형 연결

원본 PTB-XL v1.0.3의 `records100` 디렉터리를 다운로드한 후 `filename_lr`의 값(예: `records100/00000/00001_lr`)을 **확장자 없이** WFDB `rdsamp`에 전달합니다.

```python
import wfdb
signal, info = wfdb.rdsamp(str(raw_root / row['filename_lr']))
assert signal.shape == (1000, 12)
```

## GitHub 업로드

기존 저장소 루트에 압축을 해제한 내용 중 `ecg_reconstruction/` 폴더를 병합하면 됩니다. GitHub 업로드 화면에는 ZIP 자체가 아니라 내부 파일을 올리세요.

기존 프로젝트의 `.gitignore`가 `*.csv`를 제외한다면 Git에서 다음 명령으로 CSV들을 강제로 추가할 수 있습니다:

```bash
git add -f ecg_reconstruction/reports/role1_label_policy_v3_20261109_100hz/tables/*.csv
git add ecg_reconstruction/reports/role1_label_policy_v3_20261109_100hz/README.md ecg_reconstruction/reports/role1_label_policy_v3_20261109_100hz/LABEL_POLICY_V3.md
```

파형이 포함되어 있지 않으므로 용량이 큰 원본 ECG 데이터는 GitHub에 올리지 않아도 됩니다.
