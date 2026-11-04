# ECG 질환 분류 — v2 라벨 비교 실험

3번 질환 분류 담당의 데이터 복구 및 v1/v2 비교 실험 결과입니다. 기존 저장소 파일을 덮어쓰지 않는 별도 폴더입니다.

## 실제 실행 결과

- 공개 PTB-XL v1.0.3에서 기존 1,000개 ECG ID의 파형을 복구했습니다.
- 품질 제외 ECG 421·710 이후 998개, split 798/100/100을 유지했습니다.
- 미분류 8개를 제외한 지도학습 대상은 796/96/98입니다.
- v1/v2 각각 18개, 총 36개 분류 모델을 같은 CPU 환경에서 40 epoch씩 실제 학습·평가했습니다.
- 같은 환자 목록·구조·하이퍼파라미터·정규화·배치 순서와 동일 v2 test 정답으로 비교했습니다.
- 재구성 입력 C의 Macro F1은 0.5825 → 0.5714입니다. CD F1은 상승했지만 전반적인 성능 개선은 확인되지 않았습니다.
- 기존 18개 모델의 복구 입력 재추론에서 최종 예측 라벨은 과거 저장 결과와 모두 일치했습니다. 복구 NPZ의 파일 해시는 과거와 다르므로 바이트 동일 복원은 아닙니다.

## 먼저 볼 파일

- [전체 분석 보고서](REPORT.md)
- [동일 CPU·동일 v2 정답 비교 그래프](reports/same_cpu_common_v2_per_class.png)
- [seed별 Precision·Recall·F1 비교](reports/comparison_same_cpu_common_v2_by_seed.csv)
- [평균·표준편차 비교](reports/comparison_same_cpu_common_v2_mean_sd.csv)
- [조건 일치 검증](reports/controlled_comparison_checks.json)
- [실행 환경](reports/environment.json)

## 이 GitHub 공유본과 전체 실행본

이 폴더는 코드·정책·표·그래프·검증 기록을 검토하는 공유본입니다. 큰 파형 NPZ, 모델 가중치, 모델별 학습 이력·예측 배열 및 전체 로그는 별도 `ECG_role3_recovered_v2_final.zip`(약 135MB)에 보관했습니다.

**공유본만으로 학습·기존 모델 재추론·전체 재검증 명령이 실행되는 것은 아닙니다.** 전체 ZIP을 압축 해제한 `role3_recovery/`에서 [전체 실행 방법](FULL_PACKAGE_EXECUTION.md)을 따라 실행하세요. GitHub의 scripts/는 그 전체 실행본과 동일한 코드입니다. REPORT.md의 runs/models 등 일부 참조도 전체 ZIP의 경로입니다.

전체 실행본 파일명은 `ECG_role3_recovered_v2_final.zip`입니다. 팀원에게는 별도 파일 공유 경로로 전달하세요.

공식 데이터 출처: https://physionet.org/content/ptb-xl/1.0.3/

1번 라벨 생성, 재구성 모델 개발, 4번 추론 기능 연동은 수행하지 않았습니다. 기존 재구성 가중치의 고정 forward 계산은 분류 입력 복구에만 사용했습니다.
