# 1번 담당: PTB-XL 100Hz 라벨 정책 v3.0 (제안)

작성일: 2026-11-09  
정책 ID: `role1_v3.0_proposed_2026-11-09`  
상태: **연구용 파생 라벨 제안. 임상 판독 변경이나 팀 최종 승인으로 간주하지 않음.**

## 목적과 원본 보존

- 대상으로 삼은 자료는 기존 PTB-XL 100Hz 샘플 1,000개 가운데 과거 품질검사에서 421과 710을 제외한 998개이며, 이번 작업에서 새로 제외한 기록은 없음.
- 원본 `scp_codes`, likelihood, 원본 5개 진단군 `NORM/MI/STTC/CD/HYP`, `ecg_id`, `patient_id`, `strat_fold`, `split`, 파형 경로를 변경하지 않음.
- 기본 5진단군 분류 라벨은 v2와 **같음**. 신규 개선은 코드별 근거 기록, 0.0 진단 likelihood의 의미 분리, 놓친 NORM+AFIB 코드 2건 표시, 선택적 STTC 마스크, 분리된 검토·승인 기록 관리.
- 2번 담당의 재구성 분석 998개와 3번 담당의 지도학습 대상 990개를 혼동하지 않음.

## 우선순위별 라벨 규칙

| 상황 | 기본 v3 진단 라벨 | 사용 및 검토 |
|---|---|---|
| NORM만 해당 | NORM=1, 4개 질환=0 | 선택된 진단군 기준의 정상 분류 가능. 모든 임상 이상이 없다는 뜻은 아님 |
| MI/STTC/CD/HYP 중 하나라도 존재 | 원본 질환 양성은 모두 유지, 파생 NORM=0 | 혼합 원본 NORM 16개는 그대로 보존. 이진 라벨 ABNORMAL=1 |
| 여러 질환 진단군 동시 존재 | 해당 질환군 모두 1 유지 | 단일 대표 질환으로 축약하지 않음 |
| 다섯 진단군 모두 없음 | `target_v3_*` 다섯 개 공란 | 8개(PACE 5, AFLT 3) 지도학습/평가 제외. 정상 또는 모두 음성으로 오인 금지 |
| diagnostic SCP likelihood가 0 | 코드 자체와 해당 기본 라벨 유지, 불확실성 기록 | 14개 확인. likelihood 0은 PTB-XL에서 미상이라는 뜻이며, 진단이 없다는 뜻 아님 |
| STTC 양성의 근거가 전부 likelihood=0 | 기본 STTC=1 유지 | 총 8개. **선택적 민감도 실험에 한해서** `v3_optional_strict_STTC_mask=0` 적용 가능. 이 값을 STTC=0으로 사용 금지 |
| NORM과 비진단군 AFIB 기록 공존 | 기본 NORM은 자동 수정하지 않음 | ECG 330, 759의 원본 보고서 검토 요청. AFIB의 likelihood 0만으로 실제 임상 판독 확정 불가 |

## 중요: 진단 likelihood의 의미

PTB-XL 공식 문서: https://physionet.org/content/ptb-xl/1.0.3/

- `scp_codes`는 `진단문구: likelihood` 사전으로 저장되며 likelihood 0은 정보가 알려지지 않았을 때 사용됨.
- `scp_statements.csv`에서 `diagnostic=1`인 항목만 위 다섯 진단 superclass로 매핑함. 리듬/형태 코드(AFIB, PAC, PVC, PACE, AFLT 등)를 임의로 다섯 진단군에 포함시키지 않음.
- `0`이 곧 음성도 아니고 임상적으로 낮은 신뢰도를 확정하는 것도 아니므로, 숫자 하나만으로 코드 삭제나 재라벨링을 하지 않음.
- 코드 매핑은 공유 ZIP에 저장된 `code_details.csv`와 원본 진단 라벨을 한 기록씩 재검증하였고 불일치는 없었음.

## 구체적 결과

| 항목 | 수치 |
|---|---:|
| 분석 ECG | 998 |
| 기존 분할 | Train 798 / Validation 100 / Test 100 |
| 기본 5진단군 지도학습 | Train 796 / Validation 96 / Test 98, 합계 990 |
| 원본 혼합 NORM+질환 | 16 |
| 원본 5진단군 미분류 | 8 (PACE 5, AFLT 3) |
| 진단 likelihood 미상 | 14 |
| 그중 STTC의 **모든** 근거가 미상 | 8 |
| 추가 발견: NORM+AFIB 기록 | 2 (ECG 330, 759) |
| 고유 검토 대상 | 38 (항목별 중복 존재) |
| 기존 NORM 양성 | 545 |
| v3 파생 NORM 양성 | 529 |
| MI, STTC, CD, HYP | 각 183, 217, 195, 98 (v2와 동일) |

## 출력 파일의 안전한 사용

- `tables/ecg_labels_v3_master_998.csv`: 원본 및 v2 열을 보존하고 v3 라벨·증거·검토 플래그를 추가한 998개 마스터. **기본 학습용 CSV가 아님.**
- `tables/ecg_labels_v3_training_990.csv`: 5개 클래스의 v3 값을 `NORM/MI/STTC/CD/HYP` 열로 명확히 출력한 **990개 학습용 후보**. 원본 라벨은 `source_*` 열에 별도로 보존.
- `tables/label_review_v3_38.csv`: 신규 검토 대상 38개, 변경 제안과 근거. 이 파일은 자동 생성 자료이며 사람이 직접 편집하지 말 것.
- `tables/review_code_evidence_v3.csv`: 검토 대상의 원본 코드 설명, likelihood, 진단군을 행별로 저장.
- `tables/review_decisions_v3.csv`: 검토자·근거·승인 여부만 사람이 입력. **생성 코드를 다시 돌려도 기존 이 파일은 덮어쓰지 않음.**
- `tables/class_distribution_v3.csv`, `tables/split_status_v3.csv`: 분할별 진단 분포, 주요 검토 사유 통계.
- `tables/label_policy_rules_v3.csv`, `tables/audit_v3.json`: 라벨 규칙표와 해시·원본 대응·환자 중복 무결성 기록.

## 추가 검증 및 향후 연구

- ECG 330, 759에서 원본은 `NORM:50`과 `AFIB:0`이 같이 존재함. 이 둘은 **5개 진단군의 원래 정의 아래서는 NORM 단독**이었지만, 다른 유형의 리듬 주석을 담고 있어 원본 판독문의 문맥 확인 필요. v3에서는 자동으로 NORM을 0으로 바꾸지 않음.
- likelihood 미상 진단 14개 중 8개는 STTC를 뒷받침하는 코드가 `NST_:0`뿐이므로, 추가 판독 없이 그 라벨의 참·거짓을 판단할 수 없음. 마스크는 **선택적 민감도 분석용**임. 학습에 적용하려면 분류 손실에 per-label mask를 지원하도록 별도 구현하고 모델을 다시 학습해야 함.
- 학습·검증·테스트 비율과 환자 교집합 0은 계속 유지. 특히 Test 선택/제외 규칙을 성능 결과를 본 뒤 변경하면 선택 편향이 생기므로 실험 전에 고정할 것.
- 실제 의료 판독문 원본 `report`, `validated_by_human`, `second_opinion`은 이 공유본에 없으며, 임상 검증 완료로 표시할 수 없음.
- 3번 담당의 과거 v1/ v2 체크포인트는 새 정책으로 자동 재학습되지 않음. v3 학습과 과거 모델 결과를 섞어 보고하지 않을 것.

## 검수 현황

`audit_v3.json`에 다음이 기록됨: 998개 ID, 기존 split 유지, 원본 코드/likelihood와 코드 상세표 일치, 진단군 매핑 일치, v2와 v3 기본 목표 일치, 환자 단위 split 중복 0, 미분류 8개의 v3 목표 공란, 원본 파일 쓰기 0건, 검토 결정 파일 덮어쓰기 방지.
