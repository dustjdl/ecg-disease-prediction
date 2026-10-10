# 공유용 대표 결과

이미 저장된 결과에서 45개 근거 파일을 선별한 사본이다. 합계 약 11.11 MiB이며 원본 결과는 로컬에 유지했다. 이 폴더를 만들면서 학습·추론을 실행하지 않았다.

## 포함한 근거

- seed 42·43·44의 validation 비교 CSV, 설정과 loss 일치 검사 JSON.
- 여섯 실행의 설정/완료 메타데이터와 학습 이력 CSV.
- 최종 test의 seed별·쌍별·평균/표본 SD·전체 seed pooled CSV, 설정·완료 JSON.
- 데이터 검사, WFDB 원본 일치, train 제외 근거.
- seed 42 기록별 지표, 수정된 사례 CSV, validation 선정 사례의 전체/확대 PNG.
- 단일 Lead I CPU 추론의 메타데이터, 예제 출처, 출력 규격 JSON과 PNG.

[MANIFEST.json](MANIFEST.json)에 원본 상대 경로·SHA-256, 공유본 경로·SHA-256과 변환 내용을 기록했다. JSON과 일부 검사 CSV의 로컬 절대 경로만 공유 사본에서 정리했다. 지표 수치와 다른 메타데이터는 유지했다. 원본 NPY/NPZ 파형과 checkpoint는 포함하지 않았다. `prediction_output_schema.json`은 실제 출력 NPZ의 규격만 담고 파형 수치는 담지 않는다.

실험 요약의 근거 링크는 이 폴더의 실제 파일을 가리킨다. 수치와 해석은 [EXPERIMENT_SUMMARY.md](../../EXPERIMENT_SUMMARY.md), 입출력은 [ECG_INTERFACE.md](../../ECG_INTERFACE.md)를 참고한다.

## 발표용 validation 사례

이 사례들은 seed 42 validation 100개의 결과를 본 뒤 overall MSE 차이로 선택했다. 독립 test 사례가 아니다. 수정 CSV의 `overall_change`는 기록 전체, `metric_change`는 해당 행의 MSE 부호를 뜻하며 MAE의 변화 판정과 구분한다.

| 기준 | ECG ID | 전체 10초 | 0~2초 |
|---|---:|---|---|
| 개선 최대 | 544 | [전체 PNG](results/validation_waveform_comparison/run_20261011_004136_766b2ae8d3a44039a4279e34d9eaa807/most_improved_ecg_544_full_10s.png) | [확대 PNG](results/validation_waveform_comparison/run_20261011_004136_766b2ae8d3a44039a4279e34d9eaa807/most_improved_ecg_544_zoom_0_2s.png) |
| 악화 최대 | 190 | [전체 PNG](results/validation_waveform_comparison/run_20261011_004136_766b2ae8d3a44039a4279e34d9eaa807/most_worsened_ecg_190_full_10s.png) | [확대 PNG](results/validation_waveform_comparison/run_20261011_004136_766b2ae8d3a44039a4279e34d9eaa807/most_worsened_ecg_190_zoom_0_2s.png) |
| 차이 중앙값 근접 | 155 | [전체 PNG](results/validation_waveform_comparison/run_20261011_004136_766b2ae8d3a44039a4279e34d9eaa807/median_nearest_ecg_155_full_10s.png) | [확대 PNG](results/validation_waveform_comparison/run_20261011_004136_766b2ae8d3a44039a4279e34d9eaa807/median_nearest_ecg_155_zoom_0_2s.png) |

[수정 사례 지표](results/validation_waveform_comparison/run_20261011_004136_766b2ae8d3a44039a4279e34d9eaa807/selected_record_metrics_corrected_20261011_004938_427bafa7.csv)를 사용한다. 의미가 모호했던 원본 `selected_record_metrics.csv`는 공유 대상에서 제외했다.

## 최종 test와 단일 추론

최종 test는 **동일한 test 100개**를 세 seed로 반복 평가한 것이다. `all_seed_pooled_metrics.csv`의 records=300은 반복 예측 건수이며 고유 기록은 100개다. seed 평균 RMSE와 모든 seed를 합친 pooled RMSE는 별도 지표다. 기존 seed 42 Baseline test 결과를 이미 확인했던 사실도 평가 설정에 남아 있다.

단일 추론 예제는 PTB-XL validation ECG 8의 Lead I 입력으로 실행한 결과다. [추정 12유도 PNG](results/ecg_predictions/run_20261011_000527_332f77c10d404360af25b91935f36c4a/estimated_12lead.png)에서 I은 입력, 나머지 11개는 추정이다. 실제 스마트워치 적용 검증이나 분류 모델 연결을 의미하지 않는다.

통계적 유의성이나 임상적 유효성을 주장하지 않는다. 원본 데이터의 출처는 PTB-XL v1.0.3이며, 추가 데이터·파형을 공유할 때는 팀에서 이용 조건을 확인한다.
