from pathlib import Path
import json,hashlib,zipfile
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]
def fmt(v):return 'NA' if pd.isna(v) else f'{v:.4f}'
def main():
 assert json.loads((R/'runs/v2/completion.json').read_text())['status']=='complete'
 recovery=json.loads((R/'reports/recovery_status.json').read_text())
 consistency=pd.read_csv(R/'reports/restored_vs_historical_predictions.csv')
 f=pd.read_csv(R/'reports/comparison_same_cpu_common_v2_by_seed.csv')
 d=f[(f.task=='multilabel')&f.label.isin(['NORM','MI','STTC','CD','HYP'])]
 means=d.groupby(['input','label'])[[k+s for k in ['precision','recall','f1'] for s in ['_baseline','_v2','_delta']]].mean()
 table=['| 입력 | 진단군 | 기존 P | v2 P | ΔP | 기존 R | v2 R | ΔR | 기존 F1 | v2 F1 | ΔF1 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for inp in ['A','B','C']:
  for label in ['NORM','MI','STTC','CD','HYP']:
   row=means.loc[(inp,label)];table.append('| '+inp+' | '+label+' | '+' | '.join(fmt(row[k+s]) for k in ['precision','recall','f1'] for s in ['_baseline','_v2','_delta'])+' |')
 improvements=[]
 for inp in ['A','B','C']:
  x=means.loc[inp];up=[k for k in x.index if x.loc[k,'f1_delta']>1e-10];down=[k for k in x.index if x.loc[k,'f1_delta']< -1e-10];same=[k for k in x.index if abs(x.loc[k,'f1_delta'])<=1e-10]
  improvements.append(f'- {inp}: F1 상승 {", ".join(up) or "없음"}; 하락 {", ".join(down) or "없음"}; 동일 {", ".join(same) or "없음"}.')
 h=pd.read_csv(R/'preserved/role3/results/metrics_by_seed.csv');h=h[(h.task=='multilabel')&(h.split=='test')&h.label.isin(['NORM','MI','STTC','CD','HYP'])]
 h.groupby(['input','label'])[['precision','recall','f1']].agg(['mean','std']).to_csv(R/'reports/historical_v1_test_mean_sd.csv')
 text='''# ECG 3번: 데이터 복구 및 v2 질환 분류 결과

작성일: 2026-11-05 (한국시간)

**공식 파형 복구, 기존 가중치 재추론, v1/v2 각각18개,총36개 분류 모델 학습 및 평가를 실제 실행했다.** 기존 코드·모델·성능표는 preserved/에 그대로 보존했다. 1번 라벨 생성이나 재구성 모델 개발, 4번 추론 연동은 수행하지 않았다. 기존 재구성 가중치를 고정하여 분류 입력을 복구하는 forward 계산만 수행했다.

## 실행 사실과 실험 구분

| 구분 | 수행 여부 | 근거 |
|---|---|---|
| 과거 v1 학습 결과 보존 | 완료 | preserved/role3/ |
| 과거 저장 예측 지표 재계산 | 162행 일치 | reports/historical_metric_checks.csv |
| 공식 1,000기록 파형 복구 | 완료 | reports/recovery_status.json |
| 기존 18개 분류 가중치로 복구 입력 재추론 | 완료 | reports/v1_reinference_metrics.csv |
| v1 동일 CPU 대조군 재학습 | 18모델×40epoch 완료 | runs/v1/, train_v1_cpu_control.log |
| v2 학습 | 18모델×40epoch 완료 | runs/v2/models/, completion.json |
| test 평가 | 전체 선택 고정 후 실행 | selection_frozen_before_test.json |
| 새 결과 검증 | 별도 검증 기록 참조 | reports/new_run_verification.json |

## 제공 자료와 실제 기존 조건

첨부 5개 ZIP의 내부를 직접 열었다. GitHub 코드 ZIP에는 데이터 준비 코드가 있고, role3 ZIP에는 분류 18개 모델과 재구성 6개 checkpoint 및 과거 결과가 있었다. role2 ZIP은 기존 분석표·그림만 제공하므로 입력 파형으로 사용하지 않았다. 라벨 ZIP에는 1,000/998개 목록과 split이 있었다.

분류 구조: Conv1d(C,32,k7,s2,p3)→ReLU→Conv1d(32,64,k5,s2,p2)→ReLU→Conv1d(64,128,k5,s2,p2)→ReLU→시간 mean/max concat→Linear(256,64)→ReLU→Dropout(.2)→Linear(64,출력수).

A=실제 Lead I, B=실제12유도, C=실제 Lead I+추정11유도. 입력은 모델 내부에서 N×C×1000 float32. 파일은 N×1000×12 구조이다.
다중 라벨5개와 보조 이진 정상/비정상, seeds42/43/44, batch32,40epoch, AdamW(lr .001, weight_decay .0001), train 양성/음성 비율 기반 BCEWithLogitsLoss를 유지했다. 정규화는 지도학습 train의 채널별 평균·표준편차만 사용한다. validation 최저 가중 BCE의 첫 epoch를 선택하고, 각 출력별 validation F1 최대 임계값(.05~.95,.01간격;동률은 .5에 가까운 값,그다음 높은 값)을 선택한다.18개 모델 checkpoint/임계값 고정 후 test를 평가한다.

## 원본 파형 복구와 동일성의 한계

공식 출처: https://physionet.org/content/ptb-xl/1.0.3/ (100Hz records100/).
기존 prepare_dataset_1000.py는 fold당100개,환자 중복 제외,캐시 우선/ECG ID 순으로 고른다. 캐시가 달라지면 기존과 다른 기록이 선정될 수 있다. 이번에는 새로 고르지 않고 기존 ecg_labels_all_1000.csv의1,000개 ID와 순서를 고정했다. 공식 metadata의 patient_id/fold/filename_lr와 대조하고 원래 split_record 및 write_outputs_atomically 함수를 재사용했다.

X=(1000,1000,1), Y=(1000,1000,11), float32,100Hz,mV. 유도 I/II/III/AVR/AVL/AVF/V1~V6. 원본1,000개는800/100/100 유지. 421·710은 원본 NPZ에 남겨 두고 분류용에서만 제외하여798/100/100으로 유지했다. ID·환자·fold·split 대조 및 누수 검사를 통과했다.

추정11유도는 과거 provenance가 지목한 Dilated seed42 epoch28 checkpoint를 고정하여 CPU에서 생성했다. checkpoint SHA256=e6cfd066ab8c88f1e37d5635beff7df2c3606509d0e1455392dab6642d8c6a64. 새 재구성 모델을 학습하지 않았다.

'''
 for v in recovery['files']:text+=f"- {v['role']}: 복구 SHA256 `{v['sha256']}`, 과거 파일과 바이트 일치 **{v['byte_identical']}**.\n"
 text+=f'''
기존 예제 A/B/C6개와의 오차는 recovery_status.json에 기록했다. 기존 18개 분류 모델 재추론의 최대 확률 차이는 **{consistency.max_abs_probability_difference.max():.9g}**, 예측 이진 셀 차이는 총 **{int(consistency.predicted_cell_differences.sum())}**개다. 이는 수치적 일치 검증이며 원래 재구성 NPZ 생성 명령·전체 실행 이력을 복원했다는 뜻은 아니다. 과거 파일 바이트가 없어 해시 불일치의 원인을 완전히 특정할 수 없다. NPZ 직렬화·압축과 배열 전체를 원본에 직접 대조하지 못했다. 재구성 예제에는 CPU 수치 차이가 실제 관찰되었다. 따라서 바이트 동일 복원이라고 주장하지 않는다. 같은 복구 입력으로 v1/v2 대조군을 모두 학습하여 이번 라벨 비교의 입력 조건은 통제했다.

원래 감사 코드의 hash 검사 자체를 제거하지 않았다. 복구 NPZ를 가리키는 별도 recovered_split_config.json을 생성해 새 hash를 명시하고, 과거 hash·원본 config는 보존했다. 기존 입력 구조·공식 ID·원본 파형 대조 검사는 그대로 수행했다.

## v2 라벨과 누수 검사

- v1→v2 NORM545→529. 혼합16개(train13/validation0/test3)의 NORM만1→0. 질환4개 정답 변화0.
- 미분류8개는 target NA 및 학습·평가 제외. 지도학습796/96/98 유지.
- v2 test 양성: NORM47,MI19,STTC26,CD19,HYP8. 환자 교집합0,split 변경0.
- v2는 제공된 연구 정책 제안이다. 이번 실험 적용이 팀 승인이나 임상 진단 변경을 뜻하지 않는다.

## 동일 v2 정답에서 비교한 test 결과

동일 CPU 환경에서 v1 라벨로 재학습한 대조군과 v2 라벨로 재학습한 모델을 **동일98개 test/동일v2 정답**에 평가했다. 이 표의 기존 열은 과거 저장값이 아닌 v1 CPU 대조군이다. seed42/43/44 평균이다. 아래 차이는 원시 비율 단위이며, 예를 들어 .01은1%p다. 표준편차·seed별 값·TP/FP/TN/FN은CSV에 포함한다.

'''+ '\n'.join(table)+ '\n\n'+ '\n'.join(improvements)+'''

이 증감은 동일 정답에서 관찰한 변화다. 이 표는 v1/v2를 같은 CPU 환경에서 새로 학습하여 장치 차이를 통제한 결과다. 학습 대상·입력·학습 조건을 유지하고 라벨 정책만 바꾸었다. 통계적 유의성이나 외부 환자 집단 성능을 입증하는 것은 아니다. 과거 보존 GPU 가중치와의 비교는 별도 CSV로 구분했다.

과거 v1 정답으로 계산된 성능은 historical_v1_test_mean_sd.csv, 과거값과의 참고 비교는 comparison_historical_v1_targets_*.csv에 별도로 제공한다. NORM 정답 정의가 다르므로 그 비교의 NORM 증감을 직접 개선으로 해석하면 안 된다. 네 질환 정답과 평가 대상은 동일하다.

![동일 v2 test 정답 비교](reports/same_cpu_common_v2_per_class.png)

## 개선 여부와 문제점

1. 모든 질환이 일괄 개선된다고 주장할 수 없다. 입력별 상승·하락은 위 표에서 구분한다. seed 평균과 같은 test 환자 반복의 표본 SD만 제공하며 독립 외부 검증이나 통계적 유의성을 뜻하지 않는다.
2. HYP test8개는 한 건으로 Recall .125가 달라진다. MI train136/796과validation28/96은 분포 차이가 있다. test를 보고 split이나 하이퍼파라미터를 바꾸지 않았다.
3. C는 Lead I의 결정적 변환이며 독립적으로 측정한11개의 추가 유도 정보가 아니다. C train은 같은798개로 학습했던 재구성 모델의 in-sample 출력이다. OOF 재구성을 새로 만들지 않았다.
4. 재구성 모델은validation 선택에 참여했고 test 결과는 과거부터 검토된 집합이다. 완전히 미공개 외부 test로 표현하지 않는다.
5. NORM 학습 변화는 공유 CNN 표현과 가중치·checkpoint·임계값 선택에 영향을 줄 수 있다. 네 질환의 정답이 안 바뀌어도 그 예측이 바뀔 수 있다.
6. 보조 이진 정답은v1/v2 동일하다. 동일 CPU 대조군과 v2 간 확률·예측 차이는 binary_same_cpu_repeat_check.csv에 검증했다. 이진 정답이 그대로이므로 이진 과제의 차이를 라벨 개선으로 설명하면 안 된다.
7. 초기 다운로드에서429 오류가 발생했다. 완성 파일을 보존하고 요청을초당2회 이하로 제한하여 누락 파일만 재시도했다. 실패 로그와 최종 다운로드 상태를 함께 남겼다.
8. PyTorch2.13.0,NumPy2.5.3,pandas3.0.6,sklearn1.9.1,matplotlib3.11.2,wfdb4.3.1을 실제 설치했다. 과거+cu130과 이번+cpu 장치 빌드는 다르다. Python·운영체제도 environment.json에 기록했다.

## 변경 파일 및 실행 방법

preserved/의 제공 원본은 변경0개이며 reports/preservation_check.json에서ZIP원본 바이트와 대조한다. 신규 scripts/에는 download_locked.py,build_recovered.py,train_recovered.py,audit_attachments.py,compare_and_verify.py,verify_completed.py,compare_cpu_control.py,make_report.py가 있다. 작업용 trainer는 원본 Python파일의 그대로인 사본이며 수정은 별도 adapter에서만 한다.

새로 생성한 주요 경로는 recovery/복구데이터,runs/v1/대조군결과,runs/v2/학습결과,reports/검산·비교·그래프,README.md,requirements-recovered-cpu.txt,REPORT.md,MANIFEST_SHA256.json이다. 상세 변경 목록은 CHANGED_FILES.csv에 제공한다. 실행 순서와새 출력 폴더 지정법은README.md에 있다.
'''
 secondary=pd.read_csv(R/'reports/comparison_common_v2_targets_by_seed.csv')
 secondary=secondary[(secondary.task=='multilabel')&(secondary.input=='C')&secondary.label.isin(['NORM','MI','STTC','CD','HYP'])].groupby('label')[[k+s for k in ['precision','recall','f1'] for s in ['_baseline','_v2','_delta']]].mean()
 text+='\n## 참고: 과거 보존 v1 가중치와 새 v2 비교 (C, 동일 v2 정답)\n\n이 표는 기존 GPU 학습 가중치와 새 CPU 학습 가중치의 비교이며 장치·학습환경 효과가 섞인다.\n\n| 진단군 | 기존 P | v2 P | 기존 R | v2 R | 기존 F1 | v2 F1 | ΔF1 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for label in ['NORM','MI','STTC','CD','HYP']:
  row=secondary.loc[label]
  text+='| '+label+' | '+' | '.join(fmt(row[k]) for k in ['precision_baseline','precision_v2','recall_baseline','recall_v2','f1_baseline','f1_v2','f1_delta'])+' |\n'
 macro=f[(f.task=='multilabel')&(f.input=='C')&(f.label=='macro')][['f1_baseline','f1_v2','f1_delta']].mean()
 micro=f[(f.task=='multilabel')&(f.input=='C')&(f.label=='micro')][['f1_baseline','f1_v2','f1_delta']].mean()
 text+=f'\n## 핵심 해석: 재구성 입력 C\n\n동일 CPU 대조군 기준 Macro F1은 {macro.f1_baseline:.4f} → {macro.f1_v2:.4f} (Δ {macro.f1_delta:+.4f}), Micro F1은 {micro.f1_baseline:.4f} → {micro.f1_v2:.4f} (Δ {micro.f1_delta:+.4f})이다. CD F1은 상승했지만 NORM·MI·HYP F1은 하락했고 STTC F1은 사실상 동일하다. CD는 Precision 상승과 Recall 하락, STTC는 Recall 상승과 Precision 하락이 함께 나타났다. 따라서 v2를 전반적인 성능 개선으로 결론 내릴 수 없다. 라벨 정의의 정합성과 예측 성능은 구분하여 판단해야 한다.\n'
 text+='\n동일 CPU 대조군 검증은 reports/v1_cpu_control_verification.json, 주 비교는 comparison_same_cpu_common_v2_*.csv이다. 과거 보존 가중치의 공통 v2 정답 비교는 comparison_common_v2_targets_*.csv이다.\n'
 (R/'REPORT.md').write_text(text,encoding='utf-8')
 print(means.loc['C'].to_string())
if __name__=='__main__':main()
