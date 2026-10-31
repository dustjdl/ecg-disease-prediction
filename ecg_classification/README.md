# 역할 3 질환 분류 모델 개발

이번 실험에 적용한 라벨 기준으로 A(실제 Lead I), B(실제 12유도), C(실제 I+재구성 11유도)를 비교했다. 다중 라벨·보조 이진 분류를 각각 seed 42·43·44, 40 epoch로 학습·평가한 **18개 모델의 실제 결과**다. 추가 정책 확인을 기다리지 않으며 처리 기준과 제외 기록은 유지했다.

- [이번 실험에 적용한 라벨 기준](LABEL_CRITERIA.md)
- [최종 결과 요약](CLASSIFICATION_FINAL_SUMMARY.md), [macro·micro·이진 비교](RESULT_SUMMARY.md)
- [집계 CSV와 그래프](reports/aggregate/)
- [추론 규격](CLASSIFIER_INTERFACE.md), [추론·검증·재학습 구분](REPRODUCTION.md)
- [학습 조건](EXPERIMENT_PROTOCOL.json), [필요 데이터 위치·해시](DATA_REQUIREMENTS.json)

## GitHub 공유 범위

이 모듈에는 코드, 문서, 합성 단위 테스트, 집계 지표 CSV 7개와 그래프 5장만 공유한다. 원본 ECG/NPZ, 모델 가중치, 레코드별 정답·확률·예측, ZIP은 포함하지 않는다. 집계 수치와 학습 조건은 기존 결과를 그대로 복사했으며 모델을 다시 학습하거나 test 기준으로 설정을 바꾸지 않았다.

모델 18개·입출력 예시·원래 실험 구조는 별도로 전달하는 `role3_classification_20261101_final.zip`에 들어 있다. 로컬 ZIP의 hash·파일 목록은 해당 배포본에서 확인한다. 저장소에 ZIP이나 가중치를 추가하지 않는다.

## 코드·집계 결과만으로 가능한 작업

NumPy·pandas·Matplotlib·PyTorch·scikit-learn이 설치된 Python으로 이 모듈 폴더에서 실행한다. 기존 환경 버전은 `requirements-classification.txt`에 기록했다. 새 환경 설치는 이번 작업에서 수행하지 않았다.

```powershell
python -B -m unittest discover -s tests -p test_ecg_classification.py -v
python -B -m unittest discover -s tests -p test_disease_reconstruction.py -v
python -B .\summarize_ecg_classification.py --experiment .\reports\aggregate --output .\derived_reports\aggregate_review
```

요약은 저장된 CSV를 읽어 새 폴더에만 쓰며 원본 결과를 변경하지 않는다. 코드만 있는 공유본에서 전체 모델 검증을 실행하면 필요한 모델·원래 예측·데이터가 없으므로 실행할 수 없다.

## 독립 추론

최종 ZIP의 `models/`, `examples/`를 이 모듈 아래에 준비하면 된다. NumPy·PyTorch만 필요하며 원본 학습 데이터는 필요하지 않다.

```powershell
python .\package_paths.py --mode inference
python .\classifier_inference.py --model-dir .\models\multilabel\C\seed_42 --input .\examples\inputs\C_test_9.npz --output .\my_outputs\result.json --device cpu
```

C·seed42는 고정 연결 예시이며 최적 모델 선정을 의미하지 않는다. 새 추론 출력의 `label_policy_status`는 `label criteria applied to this experiment`다. 확률·임계값·모델 계산은 변경하지 않았다.

## 전체 실험 검증·재학습

최종 ZIP에서 `results/classification_abc/`, `results/dilated_1000/`, `models/`, `checkpoints/`를 이 모듈의 같은 상대 경로에 준비한다. `DATA_REQUIREMENTS.json`에 적힌 원본 파일 3개를 `data/received/`, `data/processed/`에 별도로 배치한다. 파일 hash를 확인한 다음 [REPRODUCTION.md](REPRODUCTION.md)의 전체 검증·재학습 명령을 따른다. 이 런타임 폴더들은 Git에서 제외된다.

학습 코드는 실행 당시 hash를 유지한다. 현재 검증·요약 코드는 새 보고서 디렉터리만 쓰도록 고쳤으며 원래 코드는 `historical_source/`에 보존했다. 기록된 과거 코드·설정의 표현과 현재 라벨 기준 안내를 구분한다.

## 결과 해석

C는 A보다 macro AUROC/AP/F1 평균이 +0.010369/+0.008371/+0.023326 높았지만, micro AUROC/AP는 −0.023981/−0.042979였다. C가 A를 일관되게 개선했다고 볼 수 없고 실제 12유도 B와의 차이도 남았다. 단일 고정 test 98개·질환 중복·HYP 양성 8개·이전 test 노출·in-sample 재구성·미확인 생성 이력의 한계를 유지한다. 임상 유효성·통계적 유의성을 주장하지 않는다.
