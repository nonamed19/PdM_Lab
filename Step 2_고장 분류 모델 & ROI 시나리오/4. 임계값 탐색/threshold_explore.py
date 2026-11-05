"""저장된 Test 예측점수로 임계값 탐색(Test 라벨을 보는 탐색, 독립 평가 아님). 사용: python -I threshold_explore.py <예측CSV> <출력폴더>"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
d=pd.read_csv(sys.argv[1]); out=sys.argv[2]; y=d.y_true.values
M={'RandomForest':d.rf_proba.values,'XGBoost':d.xgb_proba.values}; C={'RandomForest':'#1f77b4','XGBoost':'#d62728'}
def at(p,t):
    pred=p>=t; tp=int((pred&(y==1)).sum()); fp=int((pred&(y==0)).sum()); fn=int((~pred&(y==1)).sum()); tn=int((~pred&(y==0)).sum())
    pr=tp/(tp+fp) if tp+fp else np.nan; rc=tp/(tp+fn); f1=2*pr*rc/(pr+rc) if tp+fp and tp else 0.0
    return dict(threshold=float(t),precision=pr,recall=rc,f1=f1,TP=tp,FN=fn,FP=fp,TN=tn)
res={}
for nm,p in M.items():
    cand=np.unique(p)                                   # 관측된 예측점수 전체를 후보 임계값으로
    tab=pd.DataFrame([at(p,t) for t in cand])
    fb=tab.loc[tab.f1.idxmax()]                          # F1 최대
    ok=tab[tab.recall>=0.85]; rb=ok.loc[ok.threshold.idxmax()]   # Recall>=0.85 중 가장 큰 임계값
    res[nm]=dict(F1최대=fb.to_dict(),Recall085=rb.to_dict(),기본0_5=at(p,.5),PRAUC=float(average_precision_score(y,p)),
                 격자=[at(p,t) for t in np.round(np.arange(.1,.91,.05),2)])
    # Precision이 임계값 상승에도 감소하는 구간
    grid=pd.DataFrame(res[nm]['격자']); dec=[(float(grid.threshold[i-1]),float(grid.threshold[i]),float(grid.precision[i-1]),float(grid.precision[i])) for i in range(1,len(grid)) if grid.precision[i]<grid.precision[i-1]-1e-12]
    res[nm]['precision_감소구간']=dec
    print(nm,'F1max',{k:round(v,4) if isinstance(v,float) else v for k,v in fb.items()}); print(nm,'R>=.85',{k:round(v,4) if isinstance(v,float) else v for k,v in rb.items()}); print(nm,'기본',{k:round(v,4) if isinstance(v,float) else v for k,v in res[nm]['기본0_5'].items()}); print('Precision 감소 구간',[(a,b,round(c,3),round(e,3)) for a,b,c,e in dec])
WARN='※ Test 라벨을 보며 기준을 고르는 탐색 (독립 평가 아님)'
# 1) PR 곡선
fig,ax=plt.subplots(figsize=(7,6)); base=y.mean()
for nm,p in M.items():
    pr,rc,th=precision_recall_curve(y,p); ax.plot(rc,pr,color=C[nm],lw=2,label=f"{nm} (PR-AUC={res[nm]['PRAUC']:.3f})")
    f=res[nm]['F1최대']; ax.plot(f['recall'],f['precision'],'o',color=C[nm],ms=9,mec='k',label=f"{nm} F1 최대 (t={f['threshold']:.3f}, F1={f['f1']:.3f})")
    r=res[nm]['Recall085']; ax.plot(r['recall'],r['precision'],'s',color=C[nm],ms=9,mec='k',label=f"{nm} Recall≥0.85 (t={r['threshold']:.3f})")
ax.axhline(base,ls='--',color='gray',label=f'무작위 기준선 ({base:.3f})'); ax.axvline(.85,ls=':',color='k',lw=1)
ax.set_xlabel('재현율 (Recall)'); ax.set_ylabel('정밀도 (Precision)'); ax.set_ylim(0,1.02); ax.set_title('Precision-Recall 곡선\n'+WARN,fontsize=11); ax.legend(fontsize=8,loc='lower left'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(out+'/pr_curve.png',dpi=130); plt.close()
# 2) 임계값별 꺾은선
fig,axs=plt.subplots(1,2,figsize=(13,5),sharey=True)
for ax,nm in zip(axs,M):
    g=pd.DataFrame(res[nm]['격자']); ax.plot(g.threshold,g.precision,'-o',color='#2ca02c',label='Precision'); ax.plot(g.threshold,g.recall,'-s',color='#d62728',label='Recall'); ax.plot(g.threshold,g.f1,'-^',color='#1f77b4',label='F1')
    ax.axvline(.5,ls=':',color='gray'); ax.axhline(.85,ls='--',color='#d62728',alpha=.4)
    ax.set_title(nm); ax.set_xlabel('임계값'); ax.set_xticks(np.arange(.1,.91,.1)); ax.grid(alpha=.3); ax.legend(loc='lower left')
axs[0].set_ylabel('값'); fig.suptitle('임계값(0.1~0.9)에 따른 Precision·Recall·F1  |  '+WARN,fontsize=12)
plt.tight_layout(); plt.savefig(out+'/threshold_lines.png',dpi=130); plt.close()
# 4) 혼동행렬 (Recall>=0.85 기준)
fig,axs=plt.subplots(1,2,figsize=(11,5))
for ax,nm in zip(axs,M):
    r=res[nm]['Recall085']; cm=np.array([[r['TN'],r['FP']],[r['FN'],r['TP']]]); ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2): ax.text(j,i,f'{cm[i,j]:,}\n({[["정확","오경보 FP"],["놓친 고장 FN","적발 TP"]][i][j]})',ha='center',va='center',fontsize=12,color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set_xticks([0,1]); ax.set_xticklabels(['예측: 정상','예측: 고장']); ax.set_yticks([0,1]); ax.set_yticklabels(['실제: 정상','실제: 고장'])
    ax.set_title(f"{nm}  (임계값 {r['threshold']:.3f})\nRecall {r['recall']*100:.1f}% · Precision {r['precision']*100:.1f}% · F1 {r['f1']:.3f}",fontsize=10.5)
fig.suptitle('Recall ≥ 0.85를 만족하는 가장 큰 임계값의 혼동행렬\n'+WARN,fontsize=12); plt.tight_layout(); plt.savefig(out+'/confusion_recall085.png',dpi=130); plt.close()
rows=[]
for nm in M:
    for key,lab in [('기본0_5','기본 0.5'),('F1최대','F1 최대'),('Recall085','Recall≥0.85 최대 임계값')]: rows.append(dict(model=nm,기준=lab,**res[nm][key]))
pd.DataFrame(rows).to_csv(out+'/selected_thresholds_metrics.csv',index=False,encoding='utf-8-sig')
json.dump(dict(주의='Test 라벨을 보며 기준을 고른 탐색이며 독립 평가가 아님',결과=res),open(out+'/threshold_results.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
# 3b) 관측 점수 기준 Precision 곡선 (임계값을 올려도 Precision이 항상 오르지는 않음을 확인)
fig,axs=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
for ax,(nm,p) in zip(axs,M.items()):
    ts=np.unique(p); pr=np.array([((p>=t)&(y==1)).sum()/(p>=t).sum() for t in ts])
    ax.step(ts,pr,where='post',color='#2ca02c'); dec=np.where(np.diff(pr)<-1e-12)[0]
    ax.plot(ts[dec+1],pr[dec+1],'v',color='#d62728',ms=6,label=f'임계값을 올렸는데 Precision이 내려간 지점 ({len(dec)}곳)')
    ax.set_title(nm); ax.set_xlabel('임계값 (관측된 예측점수 전체)'); ax.grid(alpha=.3); ax.legend(loc='lower right',fontsize=9)
axs[0].set_ylabel('Precision'); fig.suptitle('임계값에 따른 Precision (관측 점수 단위)  |  '+WARN,fontsize=12)
plt.tight_layout(); plt.savefig(out+'/precision_nonmonotonic.png',dpi=130); plt.close()
