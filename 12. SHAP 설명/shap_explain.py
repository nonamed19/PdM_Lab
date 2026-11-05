"""RF SHAP 설명. 사용: python -I shap_explain.py <전처리폴더> <RF폴더> <피처CSV> <출력폴더>"""
import sys, json, joblib, numpy as np, pandas as pd, shap, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
prep,rfd,feat,out=sys.argv[1:5]; THR=0.44
Xte=pd.read_csv(prep+'/X_test.csv'); yte=pd.read_csv(prep+'/y_test.csv').iloc[:,0].values
raw=pd.read_csv(prep+'/test_original_raw.csv'); P=joblib.load(prep+'/preprocessor.joblib')
rf=joblib.load(rfd+'/rf_model.joblib'); saved=pd.read_csv(rfd+'/test_predictions.csv')
proba=rf.predict_proba(Xte)[:,1]
assert np.allclose(proba,saved.proba_failure.values) and (yte==saved.y_true.values).all(); print('저장 예측과 일치 OK')
# 행 정렬 확인: 스케일 역변환 == 원본 Test 값
assert np.allclose(P['scaler'].inverse_transform(Xte[P['continuous_cols']]),raw[P['continuous_cols']].values); print('원본 Test 행 정렬 OK')
ft=pd.read_csv(feat,encoding='utf-8-sig')[['UDI','TWF','HDF','PWF','OSF','RNF']]; raw=raw.merge(ft,on='UDI',how='left')
pred=(proba>=THR).astype(int); kind=np.where((yte==1)&(pred==1),'TP',np.where((yte==1)&(pred==0),'FN',np.where((yte==0)&(pred==1),'FP','TN')))
print(pd.Series(kind).value_counts().to_dict())
KOR={'air_temp':('공기 온도','K'),'process_temp':('공정 온도','K'),'rpm':('회전 속도','rpm'),'torque':('토크','Nm'),'tool_wear':('공구 마모','min'),'temp_diff':('temp_diff(파생)','K'),'power':('power(파생)','W'),'tool_torque_interaction':('마모×토크(파생)',''),'Type_H':('Type_H',''),'Type_L':('Type_L',''),'Type_M':('Type_M','')}
def val(i,f):
    if f.startswith('Type_'): return '예' if f=='Type_'+raw.loc[i,'Type'] else '아니오'
    v=raw.loc[i,f]; return f"{v:,.1f}{KOR[f][1] and ' '+KOR[f][1]}" if f in('air_temp','process_temp','torque','temp_diff') else f"{v:,.0f}{KOR[f][1] and ' '+KOR[f][1]}"
ex=shap.TreeExplainer(rf); base=float(np.ravel(ex.expected_value)[-1])
def sv(X):
    s=ex.shap_values(X); s=np.array(s)
    return s[...,1] if s.ndim==3 else s
# 1) 표본 200행 (Test 전체에서 무작위, 층화 아님)
rng=np.random.RandomState(42); samp=np.sort(rng.choice(len(Xte),200,replace=False))
S=sv(Xte.iloc[samp]); comp=pd.Series(kind[samp]).value_counts().to_dict(); nfail=int(yte[samp].sum())
print('표본 구성',comp,'실제 고장',nfail)
imp=pd.Series(np.abs(S).mean(0),index=Xte.columns).sort_values(ascending=False).head(10)[::-1]
fig,ax=plt.subplots(figsize=(8.5,5.8)); ax.barh([KOR[k][0] for k in imp.index],imp.values,color='#4c78a8')
for i,v in enumerate(imp.values): ax.text(v+.001,i,f'{v:.3f}',va='center')
ax.set_xlabel('평균 |SHAP| (고장 확률에 미치는 평균 기여, 확률 단위)')
ax.set_title(f'SHAP 기여도 상위 10개 - Test 무작위 표본 200행\n표본 구성: 실제 고장 {nfail}건·정상 {200-nfail}건 (맞힘 {comp.get("TP",0)}·놓침 {comp.get("FN",0)}·오경보 {comp.get("FP",0)}) ※ 전체 Test의 중요도가 아님',fontsize=10.5)
fig.text(.5,.005,'SHAP은 모델 판단의 설명이며 실제 고장 원인을 증명하지 않습니다',ha='center',fontsize=9,color='#b00020')
plt.tight_layout(rect=(0,.03,1,1)); plt.savefig(out+'/shap_top10_sample200.png',dpi=130); plt.close()
# 2) 사례 선택: 유형별로 예측점수가 해당 유형 중앙값에 가장 가까운 사례(고르는 규칙을 고정)
cases={}
for k in ['TP','FN','FP']:
    idx=np.where(kind==k)[0]; med=np.median(proba[idx]); i=int(idx[np.argmin(np.abs(proba[idx]-med))]); cases[k]=i
    print(k,'후보',len(idx),'중앙값',round(med,3),'선택 행',i,'UDI',raw.loc[i,'UDI'],'점수',round(proba[i],3))
TITLE={'TP':'맞힌 고장 (TP)','FN':'놓친 고장 (FN)','FP':'오경보 (FP)'}
def waterfall(i,k,path,top=8):
    s=sv(Xte.iloc[[i]])[0]; order=np.argsort(-np.abs(s)); main=order[:top]; rest=order[top:]
    labels=[f"{KOR[Xte.columns[j]][0]} = {val(i,Xte.columns[j])}" for j in main]; vals=[s[j] for j in main]
    if len(rest): labels.append(f'나머지 {len(rest)}개 변수'); vals.append(s[rest].sum())
    labels=labels[::-1]; vals=vals[::-1]; cum=base+np.cumsum(vals[::-1])[::-1]*0  # placeholder
    start=base; segs=[]; run=base
    for v in vals[::-1]: segs.append((run,v)); run+=v
    segs=segs[::-1]
    fig,ax=plt.subplots(figsize=(9.5,5.6))
    for y,((st,v),lab) in enumerate(zip(segs,labels)):
        ax.barh(y,v,left=st,color='#d62728' if v>0 else '#1f77b4',height=.6); ax.text(max(st,st+v)+.006,y,f'{v:+.3f}',va='center',fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels,fontsize=9.5)
    ax.axvline(base,ls=':',color='gray'); ax.axvline(proba[i],color='k',lw=1.2); ax.axvline(THR,ls='--',color='#ff7f0e',lw=1.2)
    ax.set_ylim(-1.2,len(labels)-.2)
    ax.text(base,-1.05,f'기준값 {base:.3f}',ha='center',fontsize=8.5,color='gray'); ax.text(THR,-0.7,f'임계값 {THR}',ha='center',fontsize=8.5,color='#ff7f0e')
    ax.set_xlabel('고장 확률 (빨강: 고장 예측을 높인 변수 / 파랑: 낮춘 변수)'); pts=[base,proba[i],THR]+[a for a,v in segs]+[a+v for a,v in segs]; ax.set_xlim(min(pts)-.1,max(pts)+.14)
    ax.set_title(f"{TITLE[k]} - UDI {raw.loc[i,'UDI']} ({raw.loc[i,'Type']}형) | 실제 {'고장' if yte[i] else '정상'} · 예측점수 {proba[i]:.3f} (기준값 {base:.3f} → 최종)",fontsize=10.5)
    fig.text(.5,.005,'SHAP은 모델 판단의 설명이며 실제 고장 원인을 증명하지 않습니다',ha='center',fontsize=9,color='#b00020')
    plt.tight_layout(rect=(0,.03,1,1)); plt.savefig(path,dpi=130); plt.close()
    assert abs(base+s.sum()-proba[i])<1e-6
    return s
cdat={}
for k,i in cases.items():
    s=waterfall(i,k,f'{out}/waterfall_{k}.png')
    o=np.argsort(-s); up=[Xte.columns[j] for j in o if s[j]>0.02][:2]; dn=[Xte.columns[j] for j in o[::-1] if s[j]<-0.02][:2]
    types=[t for t in ['TWF','HDF','PWF','OSF','RNF'] if raw.loc[i,t]==1]
    cdat[k]=dict(kind=k,title=TITLE[k],row=int(i),udi=int(raw.loc[i,'UDI']),type=raw.loc[i,'Type'],actual=int(yte[i]),proba=float(proba[i]),pred=int(pred[i]),fail_types=types,
        values={KOR[f][0]:val(i,f) for f in Xte.columns if not f.startswith('Type_')},
        shap=[dict(feature=KOR[f][0],value=val(i,f),shap=float(s[j])) for j,f in enumerate(Xte.columns)],
        up=[KOR[f][0] for f in up],down=[KOR[f][0] for f in dn])
    print(k,'↑',[KOR[f][0] for f in up],'↓',[KOR[f][0] for f in dn],'types',types)
json.dump(dict(base=base,thr=THR,cases=cdat,sample=dict(n=200,fail=nfail,comp=comp,top10=[dict(feature=KOR[k][0],v=float(v)) for k,v in imp[::-1].items()])),open(out+'/shap_cases.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
pd.DataFrame({'test_row':samp,'UDI':raw.loc[samp,'UDI'].values,'y_true':yte[samp],'proba':proba[samp],'kind':kind[samp]}).to_csv(out+'/shap_sample200_rows.csv',index=False)
pd.DataFrame(S,columns=Xte.columns).assign(test_row=samp).to_csv(out+'/shap_values_sample200.csv',index=False)
