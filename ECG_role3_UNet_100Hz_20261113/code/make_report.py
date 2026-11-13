"""Generate Korean result report from completed, verified runs."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
def main():
    p=ROOT/'results/comparison';f=pd.read_csv(p/'metrics_recomputed.csv');summary=f[f.lead=='ALL_11'].groupby('model')[['mse_mV2','mae_mV','pooled_rmse_mV']].agg(['mean','std']);comp=pd.read_csv(p/'comparison_by_lead.csv');allrow=comp.iloc[0];check=json.loads((p/'verification.json').read_text());assert check['identical_protocol']
    txt='# 3번 100Hz U-Net 실험 결과\n\n'
    txt+='실제 데이터로 CPU 학습과 검증 평가를 완료했습니다. seed 42/43/44에서 각각 30 epoch를 수행했습니다. 동일 조건에서 U-Net의 평균 MSE, MAE, pooled RMSE 모두 기존 CNN보다 높았습니다. U-Net 구현은 완료했으나 성능 개선에 성공했다고 보고하지 않습니다.\n\n'
    txt+='## 전체 결과: 검증 100개, 예측 11유도\n\n'
    rows=[]
    for name in ['CNN','UNet']:rows.append([name]+[f'{summary.loc[name,(k,"mean")]:.6f} ± {summary.loc[name,(k,"std")]:.6f}' for k in ['mse_mV2','mae_mV','pooled_rmse_mV']])
    txt+=table(['모델','MSE (mV²)','MAE (mV)','pooled RMSE (mV)'],rows)
    txt+=f'\nCNN 대비 오차 증가: MSE {-allrow.mse_mV2_improvement_percent:.2f}%, MAE {-allrow.mae_mV_improvement_percent:.2f}%, RMSE {-allrow.pooled_rmse_mV_improvement_percent:.2f}%. ±는 seed 3개의 표본 표준편차입니다. RMSE는 seed별 pooled RMSE를 구한 뒤 평균했습니다.\n\n'
    txt+='![전체 지표](results/comparison/figures/overall_metrics.png)\n\n## Seed별 결과\n\n'
    rows=[]
    for name,path in [('CNN',ROOT/'reference/cnn_baseline'),('UNet',ROOT/'results/unet')]:
        for seed in [42,43,44]:
            s=json.loads((path/f'seed_{seed}/run_summary.json').read_text());r=f[(f.model==name)&(f.seed==seed)&(f.lead=='ALL_11')].iloc[0];rows.append([name,seed,s['best_epoch'],f'{r.mse_mV2:.6f}',f'{r.mae_mV:.6f}',f'{r.pooled_rmse_mV:.6f}'])
    txt+=table(['모델','seed','선택 epoch','MSE','MAE','pooled RMSE'],rows)
    txt+='\n## 유도별 결과\n\n양수 개선율은 U-Net 오차 감소, 음수는 증가입니다. 3 seed 평균입니다.\n\n'
    txt+=table(['유도','CNN MSE','U-Net MSE','MSE 개선율','CNN MAE','U-Net MAE'],[[r.lead,f'{r.cnn_mse_mV2:.6f}',f'{r.unet_mse_mV2:.6f}',f'{r.mse_mV2_improvement_percent:+.2f}%',f'{r.cnn_mae_mV:.6f}',f'{r.unet_mae_mV:.6f}'] for _,r in comp.iloc[1:].iterrows()])
    txt+=f'\nV2·V3에서는 평균 MSE가 소폭 감소했지만 나머지 9개 유도에서는 증가했습니다. 기록별로 seed 평균 MSE를 비교하면 {check["records_improved_mse"]}/100개에서 감소했습니다. 이는 통계적 유의성 검정 결과가 아닙니다. 유도별 RMSE와 전체 정밀 수치는 results/comparison/comparison_by_lead.csv에 있습니다.\n\n'
    txt+='![유도별 오차](results/comparison/figures/lead_errors.png)\n\n## 실제 파형 비교\n\n성능을 보고 사례를 선택하지 않았습니다. 검증 데이터의 가장 작은 ECG ID 8·10·17을 사전에 정한 규칙으로 선택하고 seed 42의 각 모델 최저 검증 MSE 가중치를 사용했습니다. 관측 I은 입력으로만 표시하며 점수에 포함하지 않습니다. 검은색=실제, 주황색=CNN, 파란색=U-Net입니다.\n\n![ECG 8 파형](results/comparison/figures/waveform_ecg_8.png)\n\n다른 두 사례 및 학습곡선, 기록별 MSE 산점도는 results/comparison/figures/에 있습니다.\n\n'
    txt+='## 공정성 및 재현 확인\n\n- GitHub 최신 main 92872ee의 2번 결과를 기준으로 사용했습니다.\n- 학습 798 / 검증 100 / 테스트 100, 동일 ECG와 환자 분할, 동일 mV 원신호 및 전처리 없음 조건입니다.\n- 같은 seed, 30 epoch, batch 8, Adam 0.001, MSE, 최저 validation MSE 선택을 사용했습니다.\n- 기존 CNN을 동일 환경에서 원본 코드 그대로 재학습했습니다. 3개 seed 모두 기존 예측과 학습 이력의 최대 절대 차이는 0이었습니다.\n- 데이터와 라벨 및 CNN 코드 해시를 검증했습니다. 공개된 실제 파형 전용 NPZ 해시를 사용했습니다.\n- 모든 비교 지표를 저장된 실제 파형/예측에서 float64로 독립 재계산하여 저장 표와 대조했습니다.\n- 모델 출력 shape, 유한한 gradient, 잘못된 입력 길이 차단 및 알려진 값의 지표 계산 검사를 통과했습니다.\n\n'
    txt+='## 해석과 한계\n\n이번 결과는 동일 학습 예산에서의 구조 비교입니다. CNN 31,819개와 U-Net 5,237,067개로 파라미터 수는 다릅니다. 학습시간도 같게 맞춘 실험이 아닙니다. 일부 CPU 작업은 동시에 수행했으므로 기록된 실행시간은 모델 속도 벤치마크로 사용하지 마세요.\n\n논문 공개 코드의 6단계 U-Net을 바탕으로 100Hz kernel=5, 출력 11개, linear 출력으로 변경했습니다. 논문의 정상군·학습률·학습량을 그대로 재현한 실험이 아닙니다. 특히 CNN과 맞춘 lr=0.001과 30 epoch가 U-Net에 최적인지는 확인하지 않았습니다. 성능 악화의 원인을 단정할 수 없으며, 결과를 보고 임의로 설정을 바꾼 추가 실험은 하지 않았습니다. 세부 구분은 docs/PAPER_AND_ADAPTATION.md를 참고하세요.\n\n모델 선택에 사용한 검증 데이터 결과이므로 최종 일반화 성능이 아닙니다. 테스트 100개는 추론하지 않았습니다. 500Hz 데이터 처리·학습·평가·주파수 비교 및 다른 팀원 코드 수정은 하지 않았습니다. 향후 최종 테스트를 할 경우 두 모델을 같은 테스트 데이터에서 한 번에 평가해야 합니다.\n\n## 전달용 요약\n\n100Hz 기준으로 논문 공개 코드 기반 U-Net 구현 및 3개 seed 학습·검증을 완료했습니다. 기존 CNN과 데이터, 환자 분할, 전처리, 학습 조건을 맞췄으며 전체 MSE는 CNN 0.039281, U-Net 0.040291로 U-Net이 약 2.57% 높았습니다. V2·V3는 소폭 개선되었지만 전체 성능 개선은 확인하지 못했습니다. 코드, 가중치, 예측, 결과표와 비교 파형을 함께 정리했습니다.\n'
    (ROOT/'RESULTS_KO.md').write_text(txt,encoding='utf-8')
    (ROOT/'results/run_status.json').write_text(json.dumps({'status':'completed','actual_data_training':True,'seeds':[42,43,44],'epochs_each':30,'device':'cpu','evaluation_split':'validation','test_evaluated':False,'sampling_rate_hz':100,'overall_unet_improved':False},indent=2))
if __name__=='__main__':main()
