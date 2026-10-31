"""Read saved classification metrics and write a NEW derived report directory."""
from pathlib import Path
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parent
DEFAULT=ROOT/'results/classification_abc/experiment_20261101'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path,default=ROOT/'derived_reports/classification_summary',help='New directory only')
    args=parser.parse_args()
    if args.output.exists():parser.error('Output directory exists; choose a new --output')
    summary=pd.read_csv(args.experiment/'abc_comparison_mean_sd.csv')
    differences=pd.read_csv(args.experiment/'test_paired_differences.csv')
    args.output.mkdir(parents=True,exist_ok=False)
    relative=Path(os.path.relpath(args.experiment,args.output)).as_posix()
    lines=['# 역할 3 질환 분류 모델: 저장 결과 재요약','',
           '**라벨 정책은 이번 실험에 적용한 라벨 기준이다.** 새 학습·추론·test 선택 없이 저장된 CSV를 읽어 정리했다.',
           '',f'원본: [{relative}/abc_comparison_mean_sd.csv]({relative}/abc_comparison_mean_sd.csv)',
           '', 'A=실제 Lead I, B=실제 12유도, C=실제 I+재구성 11유도. 같은 test 98개, seed 42/43/44의 평균 ± 표본 SD다.']
    fig,axes=plt.subplots(3,3,figsize=(13,11),layout='constrained')
    for row,(task,label,title) in enumerate([('multilabel','macro','다중 라벨 macro'),('multilabel','micro','다중 라벨 micro'),('binary','ABNORMAL','보조 이진')]):
        f=summary[(summary.task==task)&(summary.label==label)&(summary.split=='test')].set_index('input').loc[['A','B','C']]
        lines.extend(['',f'## {title}','','| 입력 | AUROC | AUPRC(AP) | precision | recall | F1 |','|---|---:|---:|---:|---:|---:|'])
        for inp,r in f.iterrows():
            lines.append('| '+inp+' | '+' | '.join(f"{r[k+'_mean']:.4f} ± {r[k+'_sd']:.4f}" for k in ['auroc','auprc','precision','recall','f1'])+' |')
        for col,key in enumerate(['auroc','auprc','f1']):
            ax=axes[row,col]
            ax.bar(['A','B','C'],f[key+'_mean'],yerr=f[key+'_sd'],capsize=4,color=['#2563eb','#64748b','#d97706'])
            ax.set_ylim(0,1.08);ax.set_title(f'{task} {label}: {key.upper()}');ax.grid(axis='y',alpha=.2)
    fig.savefig(args.output/'comparison.png',dpi=170);plt.close(fig)
    lines.extend(['','![저장 결과](comparison.png)','','## C와 A·B의 비교',''])
    for task,label in [('multilabel','macro'),('multilabel','micro'),('binary','ABNORMAL')]:
        for comparison in ['C-A','C-B']:
            f=differences[(differences.task==task)&(differences.label==label)&(differences.comparison==comparison)].set_index('metric')
            lines.append(f"{task} {label} {comparison}: "+', '.join(f"{k} {f.loc[k,'paired_mean_difference']:+.6f}" for k in ['auroc','auprc','f1'])+'.')
            lines.append('')
    lines.extend(['C의 macro 일부 지표는 A보다 높지만 micro AP는 낮았다. C가 A를 일관되게 개선했다고 볼 수 없으며 B와의 차이도 남았다.',
                  '', '기존 798/100/100 분할은 유지했고 미분류를 loss·지표에서만 제외해 분모는 796/96/98이다. HYP 양성 8개, 질환 중복, 동일 분할 반복, in-sample 재구성 및 기존 test 노출의 한계가 있다. 임상적 유효성·통계적 유의성을 주장하지 않는다.',
                  '', '이 보고서는 새 파생 파일이며 기존 모델·설정·예측·지표를 수정하지 않았다.'])
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Created a new derived report; saved experiment was not modified.')


if __name__=='__main__':main()
