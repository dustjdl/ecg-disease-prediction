"""Create final classification report from fixed saved results only."""
from pathlib import Path
import argparse
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT = Path(__file__).resolve().parent
DEFAULT = PROJECT / 'results/classification_abc/experiment_20261101'
COLORS = ['#2563eb','#64748b','#d97706']


def table(frame):
    lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join('NA' if pd.isna(v) else f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def mean_sd_table(summary,task,label,metrics):
    s=summary[(summary.task==task)&(summary.label==label)&(summary.split=='test')].set_index('input').loc[['A','B','C']]
    rows=[]
    for inp,r in s.iterrows():
        rows.append({'입력':inp,**{m:f"{r[m+'_mean']:.4f} ± {r[m+'_sd']:.4f}" for m in metrics}})
    return table(pd.DataFrame(rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,default=DEFAULT)
    args=p.parse_args()
    out=args.experiment
    completion=json.loads((out/'completion.json').read_text(encoding='utf-8'))
    verification=json.loads((out/'verification.json').read_text(encoding='utf-8'))
    assert completion['status']=='complete' and verification['status']=='passed'
    s=pd.read_csv(out/'abc_comparison_mean_sd.csv')
    metrics=pd.read_csv(out/'metrics_by_seed.csv')
    support=pd.read_csv(out/'label_support.csv')
    differences=pd.read_csv(out/'test_paired_differences.csv')
    selections=pd.read_csv(out/'selected_models.csv')
    labels=['NORM','MI','STTC','CD','HYP']
    data=s[(s.task=='multilabel')&(s.split=='test')&s.label.isin(labels)]
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,metric in zip(axes,['auroc','auprc','f1']):
        for j,inp in enumerate(['A','B','C']):
            f=data[data.input==inp].set_index('label').loc[labels]
            ax.bar(np.arange(5)+(j-1)*.24,f[metric+'_mean'],width=.24,yerr=f[metric+'_sd'],capsize=3,label=inp,color=COLORS[j])
        ax.set_xticks(range(5),[f'{label}\npositive n={n}' for label,n in zip(labels,[50,19,26,19,8])],fontsize=9)
        ax.set_ylim(0,1.1);ax.set_title(metric.upper());ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Test multilabel: 98 ECGs; mean +/- sample SD over 3 seeds (same test cohort)')
    fig.savefig(out/'test_multilabel_by_class_annotated.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(10,9),layout='constrained')
    for i,seed in enumerate([42,43,44]):
        for j,inp in enumerate(['A','B','C']):
            r=metrics[(metrics.task=='binary')&(metrics.split=='test')&(metrics.input==inp)&(metrics.seed==seed)].iloc[0]
            matrix=np.array([[r.tn,r.fp],[r.fn,r.tp]],dtype=int)
            ax=axes[i,j];ax.imshow(matrix,cmap='Blues',vmin=0,vmax=51)
            for y in range(2):
                for x in range(2): ax.text(x,y,str(matrix[y,x]),ha='center',va='center',fontsize=17,color='white' if matrix[y,x]>25 else 'black')
            ax.set_xticks([0,1],['Normal','Abnormal']);ax.set_yticks([0,1],['Normal','Abnormal'])
            ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(f'{inp}, seed {seed}, threshold {r.threshold:.2f}')
    fig.suptitle('Test binary confusion matrices: normal 47 / abnormal 51, unclassified excluded')
    fig.savefig(out/'test_binary_confusion_matrices.png',dpi=170);plt.close(fig)
    perclass=[]
    for label in labels:
        for inp in ['A','B','C']:
            r=data[(data.label==label)&(data.input==inp)].iloc[0]
            perclass.append({'라벨':label,'양성 n':int(r.positive),'입력':inp,**{m:f"{r[m+'_mean']:.4f} ± {r[m+'_sd']:.4f}" for m in ['auroc','auprc','precision','recall','f1']}})
    paired=differences[(differences.comparison.isin(['C-A','C-B']))&differences.metric.isin(['auroc','auprc','f1'])]
    paired=paired[['task','label','comparison','metric','paired_mean_difference','paired_sd_difference']]
    doc=f"""# ECG 질환 분류 A/B/C 실제 실험 결과

실험일: 2026-11-01 · GPU: NVIDIA GeForce RTX 5070 · 2 task × 3입력 × 3seed = **18개 실제 학습·평가 완료**.

## 핵심 결론

C(실제 Lead I+재구성 11유도)는 A(실제 Lead I)보다 다중 라벨 **macro AUROC +0.010369, macro AUPRC +0.008371, macro F1 +0.023326**의 평균 차이를 보였다. 그러나 **micro AUROC −0.023981, micro AUPRC −0.042979**였고, micro AUPRC는 3개 seed 모두 낮았다. 따라서 재구성이 분류 성능을 일관되게 개선했다고 결론내리지 않는다.

B(실제 12유도)에 비해 C는 macro AUROC −0.040772, macro AUPRC −0.046015, macro F1 −0.049871이었다. 이 세 macro 지표는 모든 seed에서 C가 B보다 낮았다. 이 실험의 재구성 입력이 실제 12유도를 대체하는 수준이라고 주장할 수 없다.

보조 정상·비정상 이진 분류에서는 C−A의 평균 AUROC +0.009039, AUPRC +0.010140, F1 +0.030706이었다. 다만 seed 42의 AUROC/AP는 C가 A보다 낮았고, C−B의 평균 AUROC −0.021555, AP −0.003483, F1 −0.017737로 B에 비해 낮았다. 이는 같은 소규모 test에 대한 기술 통계이며 유의성이나 일반화를 보증하지 않는다.

## 정책 확인·분할·평가 분모

로컬 저장소의 확정된 2번 담당 라벨 정책은 찾지 못했다. `ecg-disease-prediction/docs`, `src`, `tests`는 .gitkeep만 있고, 기존 `ECG_INTERFACE.md`는 분류 합의가 미확정이라고 기록한다. 사용자 지정 기본값인 NORM/MI/STTC/CD/HYP 다중 라벨을 적용했다. 저장소 밖의 팀원 자료나 원격 정책을 확인했다고 주장하지 않는다.

원본 CSV와 NPZ는 ecg_id로 1:1 결합했고 중복·누락·추가·split 불일치가 없었다. **train/validation/test=798/100/100을 변경하지 않았다. patient_id 교집합은 train–validation, train–test, validation–test 모두 0명**이었다. 실제 파형은 원본 NPZ와 최대 차이 0.0 mV로 일치했다.

5개 라벨이 모두 0인 미분류 train 2개, validation 4개, test 2개는 loss/지표에서만 제외하고 split을 보존했다. 따라서 지도학습·평가 분모는 **796/96/98**이다. 미분류도 확률·예측을 저장했고 정답은 NA다.

다중 라벨은 NORM+질환 혼합의 원래 라벨을 모두 유지한다. 이진 분류는 질환이 하나라도 있으면 양성(혼합 포함), NORM만 있으면 음성이다. test 이진 분모는 정상 47/비정상 51이다. include_in_analysis를 라벨로 사용하지 않았다.

{table(support[support.task=='multilabel'][['split','label','n_original','n_eligible','positive','negative']])}

기존 파형 분석은 혼합군을 별도로 분리했기 때문에 test CD=16개, 정상 단독=47개였다. 이번 다중 라벨은 혼합 CD 3개를 포함해 CD 양성=19개, NORM 양성=50개다. 분석 목적에 따른 마스크 차이이며 split을 바꾼 것이 아니다.

## 학습·선택 방법

A=1채널 실제 Lead I, B=12채널 실제 파형, C=실제 I와 재구성 11유도다. C의 I는 명시적으로 actual I로 채웠고 입력 NPZ에서도 두 I가 같았다. 첫 Conv1d 입력 채널 수 외의 CNN 구조는 동일하다. 같은 task/seed에서 더 깊은 계층의 초기값과 epoch별 minibatch 순서는 A/B/C에 동일하게 맞췄다.

Conv 채널 32/64/128, kernel 7/5/5, stride 2/2/2, ReLU, global mean+max, Linear 256→64→출력, dropout 0.2를 사용했다. 다중 라벨 파라미터는 A=68,421, B/C=70,885이며 이진은 A=68,161, B/C=70,625다. A와 12채널 모델의 파라미터 차이를 완전히 없애지는 못했다.

seed 42/43/44, 각 40 epoch, batch 32, AdamW lr 0.001, weight decay 0.0001로 고정했다. 정규화 평균·SD와 BCE pos_weight는 **해당 입력의 train 796개에서만** 계산했다. 같은 task에서 A/B/C의 train 라벨·가중치·순서가 일치한다. sigmoid 출력의 확률 보정은 수행하지 않았다.

최저 weighted validation BCE의 첫 epoch를 checkpoint로 선택했다. 그 checkpoint의 validation 96개에서 출력별 F1 최대 임계값을 0.05~0.95, 0.01 간격으로 정했다. 동률은 0.5와 가까운 값, 다시 동률이면 큰 값이다. **18개 모델과 임계값을 모두 고정한 후 test를 평가했다.** `selection_frozen_before_test.json`과 checkpoint/threshold SHA-256으로 확인할 수 있다. test 분류 점수를 본 뒤 재학습·조건 변경을 하지 않았다.

## Test 다중 라벨: macro

아래 및 이후 표는 **seed 평균 ± 표본 SD(ddof=1)**이며 같은 test 98개에 대한 반복이다. AUROC/AP는 확률 기반, precision/recall/F1은 validation 임계값 기준이다. AUPRC는 비보간 Average Precision(AP)이다.

{mean_sd_table(s,'multilabel','macro',['auroc','auprc','precision','recall','f1'])}

![다중 라벨 macro 비교](test_multilabel_comparison.png)

## Test 다중 라벨: micro

{mean_sd_table(s,'multilabel','micro',['auroc','auprc','precision','recall','f1'])}

macro는 클래스별 동등 가중, micro는 전체 ECG×클래스 쌍을 모은 집계다. 가중 손실로 학습한 클래스별 확률 스케일 차이가 micro 순위에도 영향을 줄 수 있지만, 이번 실험에서는 확률 보정이나 원인 검증을 하지 않았다. macro 개선만 골라 전체 개선으로 표현하지 않는다.

## Test 클래스별 성능

{table(pd.DataFrame(perclass))}

![클래스별 비교](test_multilabel_by_class_annotated.png)

C의 CD F1 평균은 0.427576으로 A의 0.460159보다 낮았다. HYP F1은 C=0.650314, A=0.508485로 높았지만 HYP 양성이 **8개**뿐이므로 불안정하다. 클래스별·seed별 결과가 달라 일괄적인 질환별 개선으로 해석하지 않는다. AUROC만 보면 HYP가 높아 보이더라도 precision/F1·양성 수를 함께 확인해야 한다.

## Test 보조 이진 분류

{mean_sd_table(s,'binary','ABNORMAL',['auroc','auprc','precision','sensitivity','specificity','f1'])}

![이진 비교](test_binary_comparison.png)

![이진 혼동행렬](test_binary_confusion_matrices.png)

혼동행렬은 실제 행(정상/비정상), 예측 열(정상/비정상)이다. seed별 `TN,FP,FN,TP`와 임계값 원본은 [CSV](binary_confusion_matrices.csv)에 저장했다. C의 평균 민감도는 A보다 높으나 B에 비해 평균 특이도가 낮았다. 임계값과 오류 유형을 함께 고려해야 한다.

## 같은 seed의 입력 차이

{table(paired)}

차이의 SD도 같은 seed의 차이 3개로 계산했다. 독립 환자군의 차이나 신뢰구간이 아니다. 모든 seed별 차이는 [test_paired_differences.csv](test_paired_differences.csv)에 있다.

## 학습 이력과 실제 완료 항목

{table(selections[['task','input','seed','best_epoch','best_validation_loss','training_seconds']])}

![validation 학습 곡선](validation_learning_curves.png)

40 epoch 후반의 train/validation 손실 차이는 과적합 가능성을 보여 준다. 실제 평가는 최종 epoch가 아니라 위 validation 선택 epoch를 사용했다. 학습 루프 18개의 기록된 시간 합계는 {selections.training_seconds.sum():.1f}초이며, 출처 추론·검증·파일 생성 시간은 포함하지 않는다. 고속 실행 여부와 무관하게 각 모델의 history.csv에 40개 epoch와 checkpoint가 존재한다.

총 18개 checkpoint, 53,892개 ECG×라벨 예측 행, 216개 seed별 지표 행을 저장했다. 예측은 train/validation/test와 미분류를 모두 포함한다. 실제 데이터에서 정의 불가한 평가 지표는 없었으며, `metric_na_reasons.csv`는 헤더만 있다. 코드에서는 상수 예측으로 인한 precision 불가·양성/음성 부재 등 이유를 따로 저장하도록 구현했다.

단위 테스트 7개를 통과했고, 저장 확률로 162개 task/입력/seed/split/클래스 지표 집합을 sklearn으로 독립 재계산해 일치함을 확인했다. macro/micro 재계산, 체크포인트 최저 validation loss, train-only 정규화·가중치, validation-only 임계값, A/B/C minibatch 순서도 모두 검증했다. [검증 JSON](verification.json)을 참고한다.

## 재구성 NPZ 출처

로컬 6개 checkpoint의 전체 998개 추론과 비교한 결과, **Dilated seed 42, epoch 28** (`checkpoints/dilated_1000/run_20261010_234519_9f46beb7d33c40ed8124dc08ffabb636/best_model.pth`)만 전달 NPZ와 수치적으로 일치했다. 최대 차이 9.536743×10⁻⁷ mV, 평균 절대 차이 1.589357×10⁻⁸ mV다. 부동소수점 연산 오차 수준의 일치이며 정확한 NPZ 생성 명령을 확인한 것은 아니다.

checkpoint SHA-256은 `e6cfd066ab8c88f1e37d5635beff7df2c3606509d0e1455392dab6642d8c6a64`다. 저장 metadata의 원본 NPZ 해시는 현재 원본과 일치하고 train 798개 ID가 현재 validation/test와 겹치지 않는다. 관련 model/train 소스 해시도 저장 값과 일치했다. 학습 코드는 train만 optimizer에 사용하고 validation은 checkpoint 선택에 사용한다. 저장된 protocol은 test를 optimizer에 사용하지 않았다고 기록한다.

**미확인:** NPZ 생성 스크립트·명령·실행 로그, 역사적 전체 실행 과정은 저장소에서 확인하지 못했다. 따라서 제공 파일의 전체 생성 이력과 누출 부재를 독립적으로 완전 증명하지 않는다. 근거는 [출처 JSON](reconstruction_provenance.json), [6개 checkpoint 대조 CSV](reconstruction_checkpoint_consistency.csv), [소스 해시 대조](reconstruction_source_hash_verification.json)에 분리해 기록했다.

## 기존 파형 분석과 연결한 해석

[기존 재구성 보고서](../../disease_reconstruction/analysis_20261101/REPORT.md)는 Lead I을 제외한 11유도의 MAE/RMSE/r을 비교했다. 정상 단독의 test 평균 MAE는 0.071970 mV, CD군은 0.107799 mV였지만, 파형 오차 차이가 분류 성능 변화의 원인이라고 검증한 것은 아니다.

정상 test ID 9의 V2에는 형태·진폭 차이가 있고 CD test ID 65의 V5는 높은 r에도 진폭 차이가 남았다. 이들은 그룹 전체를 대표하지 않는 예시다. 기존 [발표 자료](../../../team_handoff/disease_reconstruction_20261101/presentation/PRESENTATION_SUMMARY.md)의 확대 그림은 설명을 위한 자료이며 분류 개선의 증거로 사용하지 않는다. 이번에는 파형 유사도를 분류 성능으로 대신하지 않고 별도 분류 모델과 정답 라벨로 평가했다.

관찰 결과 C는 macro 일부 지표와 이진 평균 지표에서 A보다 나았지만 micro AP와 CD F1은 개선되지 않았고, 실제 12유도 B와의 차이도 남았다. 재구성 파형을 사용하면 모든 질환 분류가 좋아진다는 결론은 뒷받침되지 않는다.

## 한계

- C는 Lead I에서 결정적으로 만들어진 표현이며 독립 측정 11유도의 새 정보를 추가하지 않는다. A보다 입력 채널·첫 계층 파라미터 수가 많고 재구성 사전학습이 있어 차이를 순수한 채널 수 효과나 재구성만의 인과 효과로 분리하지 못했다.
- C의 train 파형은 같은 train으로 학습한 재구성기의 in-sample 출력이다. out-of-fold 재구성을 사용하지 못했고 train/test 표현 분포 차이 가능성이 있다.
- 재구성 checkpoint도 같은 validation으로 선택되었다. 분류 임계값·checkpoint 역시 validation에서 선택되어 해당 96개에 과적합할 수 있다.
- train/validation/test 환자 중복은 없지만 단일 소규모 고정 분할이다. 동일 test를 3seed로 반복한 것이며 외부 독립 검증·신뢰구간·유의성 검정·환자군 보정은 수행하지 않았다.
- test HYP 양성 8개, 질환 중복, 라벨 정확도·불확실성, 혼합 라벨 정책의 영향이 있다. 원시 SCP 진단 likelihood를 새로 임상 검증하거나 판독 라벨을 정정하지 않았다.
- 기존 파형 분석에서 test 결과/라벨을 이미 살펴본 이력이 있다. 이번 분류 설정은 점수 확인 전에 고정했지만 완전 미관측 test라고 주장하지 않는다.
- sigmoid 확률은 클래스 가중 손실로 학습했고 calibration을 하지 않았다. 이 결과는 임상 진단 성능의 검증이나 의료 의사결정 근거가 아니다.

요청한 학습·평가·3seed 비교·확률/라벨 저장은 모두 실행했다. 확인하지 못한 항목은 확정된 팀원 정책과 재구성 NPZ의 완전한 생성 이력이며 위에 명시했다. 새 분할·GitHub commit/push는 하지 않았다.

## 파일과 실행

[학습·평가 코드 및 실행 안내](../../../docs/ECG_CLASSIFICATION_EXPERIMENT.md), [모델 선택 요약](selected_models.csv), [전체 예측](predictions_long.csv), [전체 지표](metrics_by_seed.csv), [평균·SD](abc_comparison_mean_sd.csv), [라벨 support](label_support.csv).

실험 폴더 `source/`에 실행 코드·검증·요약 코드·실행 안내를 복사해 보관했다. 원래 입력 파일과 기존 파형 분석·전달 폴더는 유지했다.
"""
    (out/'REPORT.md').write_text(doc,encoding='utf-8')
    root=f"""# ECG 질환 분류 최종 요약

2026-11-01: A=실제 Lead I, B=실제 12유도, C=실제 I+재구성 11유도로 다중 라벨·보조 이진 분류를 각각 seed 42/43/44, 40 epoch씩 학습·평가했다. **총 18개 모델을 실제 실행했다.**

- [전체 결과·표·그래프·한계](results/classification_abc/experiment_20261101/REPORT.md)
- [실행 방법·라벨 정책·집계 방식](docs/ECG_CLASSIFICATION_EXPERIMENT.md)
- [분석·학습 코드](train_ecg_classification.py), [저장 결과 검증 코드](verify_ecg_classification.py)
- [A/B/C 평균·표준편차 CSV](results/classification_abc/experiment_20261101/abc_comparison_mean_sd.csv)
- [ECG별 확률·정답·예측 CSV](results/classification_abc/experiment_20261101/predictions_long.csv)

## Test 다중 라벨 macro 성능

{mean_sd_table(s,'multilabel','macro',['auroc','auprc','f1'])}

평균 ± 표본 SD, 동일 test 98개·3seed. 원래 798/100/100 분할은 유지했고, 미분류를 지도학습/지표에서 제외해 분모는 796/96/98이다. 환자 ID의 분할 간 중복은 없다.

C−A의 macro AUROC/AP/F1은 각각 +0.010369/+0.008371/+0.023326이지만, micro AUROC/AP는 −0.023981/−0.042979로 낮았다. C−B의 macro AUROC/AP/F1은 −0.040772/−0.046015/−0.049871이다. **C가 A를 일관되게 개선했다고 할 수 없고, B와의 성능 차이도 남았다.**

## 보조 이진 분류

{mean_sd_table(s,'binary','ABNORMAL',['auroc','auprc','sensitivity','specificity','f1'])}

정상 47/비정상 51이며 혼합 라벨은 비정상에 포함한다. C는 A보다 평균 AUROC/AP/F1이 +0.009039/+0.010140/+0.030706 높았지만 모든 seed에서 모든 지표가 개선되지는 않았다.

## 출처·파형 분석·한계

재구성 NPZ는 기존 Dilated seed 42·epoch 28 checkpoint 출력과 수치적으로 일치했다(최대 9.54e-7 mV). 해당 저장 metadata의 학습 ID는 train 798개이며 validation/test와 겹치지 않고 관련 학습 소스 해시도 일치했다. 다만 NPZ 생성 스크립트·실행 로그는 찾지 못해 완전한 생성 이력은 미확인이다.

[기존 질환별 파형 분석](DISEASE_RECONSTRUCTION_SUMMARY.md)의 MAE/RMSE/r과 이번 분류 성능은 별도 결과다. 파형 오차가 작거나 r이 높다고 분류 성능 개선을 보증하지 않는다. C의 CD F1은 A보다 낮았고, HYP의 높은 결과는 test 양성 8개라는 한계가 있다.

동일 소규모 분할 반복, 다중 질환 중복, 재구성 train의 in-sample 출력, upstream validation 선택, 미확인 생성 이력, 이전 test 노출, 미보정 확률이라는 한계가 있다. 임상 유효성·통계적 유의성을 주장하지 않는다.

모델·임계값은 모두 test 평가 전에 고정했다. 테스트 7개와 저장 확률 기반 지표·정규화·임계값·학습 순서 검증을 통과했다. 원본은 보존했고 GitHub commit/push는 하지 않았다.
"""
    (PROJECT/'CLASSIFICATION_FINAL_SUMMARY.md').write_text(root,encoding='utf-8')
    snapshot=out/'source'
    snapshot.mkdir(exist_ok=True)
    for name in ['train_ecg_classification.py','verify_ecg_classification.py','summarize_ecg_classification.py','analyze_disease_reconstruction.py','model.py','model_dilated.py','requirements.txt','requirements-classification.txt']:
        shutil.copy2(PROJECT/name,snapshot/name)
    shutil.copy2(PROJECT/'docs/ECG_CLASSIFICATION_EXPERIMENT.md',snapshot/'ECG_CLASSIFICATION_EXPERIMENT.md')
    print('Report generated; no training or test reselection performed.')


if __name__=='__main__':
    main()
