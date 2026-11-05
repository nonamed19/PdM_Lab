"""도입 비용 시나리오(가정 기반). 사용: python -I cost_scenario.py <selected_thresholds_metrics.csv> <출력폴더>"""
import sys, json, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
m=pd.read_csv(sys.argv[1],encoding='utf-8-sig'); out=sys.argv[2]
r=m[(m.model=='RandomForest')&(m['기준']=='Recall≥0.85 최대 임계값')].iloc[0]
TP,FP,FN,TN,thr=int(r.TP),int(r.FP),int(r.FN),int(r.TN),float(r.threshold)       # 측정값 (Test, Step 2의 4번 폴더)
YEAR_N,YEAR_FAIL=2000,68                                                         # 가정: 연간 건수
COST_FAIL,COST_INSP,COST_PM,COST_SYS=5000,100,500,20000                          # 가정: 만 원
def scen(cf,prev=1.0,tp=TP,fp=FP,fn=FN):
    gross=tp*prev*(cf-COST_PM)-fp*COST_INSP; net=gross-COST_SYS
    return dict(gross=gross,net=net,roi=net/COST_SYS*100)
base=scen(COST_FAIL)
# 도입 전/후 총비용 비교 (참고)
no_sys=YEAR_FAIL*COST_FAIL
with_sys=FN*COST_FAIL+TP*COST_PM+FP*COST_INSP+COST_SYS
be=(COST_SYS+FP*COST_INSP)/TP+COST_PM                                            # 손익분기 고장비용
sens=[dict(고장비용=cf,**scen(cf)) for cf in range(3000,8001,1000)]
prev=[dict(예방률=p,**scen(COST_FAIL,p)) for p in (1.0,.7,.5,.3)]
ref05=m[(m.model=='RandomForest')&(m['기준']=='기본 0.5')].iloc[0]
alt=scen(COST_FAIL,1.0,int(ref05.TP),int(ref05.FP))
print(dict(TP=TP,FP=FP,FN=FN,thr=thr)); print(base); print('도입 전',no_sys,'도입 후',with_sys,'차이',no_sys-with_sys,'BE',be)
for s in sens: print(s)
for s in prev: print(s)
print('기본0.5',alt)
json.dump(dict(측정=dict(TP=TP,FP=FP,FN=FN,TN=TN,threshold=thr),가정=dict(연간건수=YEAR_N,연간고장=YEAR_FAIL,예방률=1.0,고장비용_만원=COST_FAIL,점검비_만원=COST_INSP,정비비_만원=COST_PM,운영비_만원=COST_SYS),
          기준시나리오=base,도입전후=dict(도입전=no_sys,도입후=with_sys),손익분기_고장비용=be,민감도_고장비용=sens,민감도_예방률=prev,기본0_5임계값비교=alt),open(out+'/cost_scenario_results.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
pd.DataFrame(sens).to_csv(out+'/sensitivity_failure_cost.csv',index=False,encoding='utf-8-sig')
fig,ax=plt.subplots(figsize=(9,5.2)); x=[s['고장비용']/10000 for s in sens]; y=[s['net']/10000 for s in sens]
bars=ax.bar([f"{v:.1f}" for v in x],y,color='#4c78a8',width=.65)
for b,s in zip(bars,sens): ax.text(b.get_x()+b.get_width()/2,b.get_height()+.4,f"{s['net']/10000:.1f}억\nROI {s['roi']:,.1f}%",ha='center',fontsize=9)
bars[2].set_color('#1f4e8c')
ax.set_xlabel('고장 1건당 비용 [억 원]  (진한 막대 = 기준 시나리오 0.5억)'); ax.set_ylabel('연간 순절감액 [억 원]'); ax.set_ylim(0,max(y)*1.2)
ax.set_title('고장 비용 민감도: 연간 순절감액 (RF 임계값 0.44, 가정 기반 시나리오)\n※ 실제 절감 실적이 아님 · 초기 구축비 미포함',fontsize=11); ax.grid(axis='y',alpha=.3)
plt.tight_layout(); plt.savefig(out+'/sensitivity_failure_cost.png',dpi=130)
