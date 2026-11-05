"""XGBoost 학습 + RF 비교. 사용: python -I train_xgb_compare.py <전처리폴더> <RF폴더> <출력폴더>"""
import sys, json, joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
src,rfd,out=sys.argv[1:4]
Xtr=pd.read_csv(src+'/X_train.csv'); ytr=pd.read_csv(src+'/y_train.csv').iloc[:,0]
Xte=pd.read_csv(src+'/X_test.csv'); yte=pd.read_csv(src+'/y_test.csv').iloc[:,0]
params=dict(n_estimators=200,max_depth=6,learning_rate=0.1,eval_metric='aucpr',random_state=42,scale_pos_weight=1)
xgb=XGBClassifier(**params).fit(Xtr,ytr)
px=xgb.predict_proba(Xte)[:,1]
rf=joblib.load(rfd+'/rf_model.joblib'); pr=rf.predict_proba(Xte)[:,1]
assert (pd.read_csv(rfd+'/test_predictions.csv')['proba_failure'].values==pr).all()  # 저장된 RF 예측과 동일
B,R='#1f77b4','#d62728'
res={}
for nm,p in [('RandomForest',pr),('XGBoost',px)]:
    pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pred).ravel()
    res[nm]=dict(accuracy=accuracy_score(yte,pred),precision=precision_score(yte,pred),recall=recall_score(yte,pred),f1=f1_score(yte,pred),roc_auc=roc_auc_score(yte,p),pr_auc=average_precision_score(yte,p),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))
print(pd.DataFrame(res).round(4).to_string())
# 임계값 민감도: 재현율 0.85 달성 시 오경보
rows=[]
for nm,p in [('RandomForest',pr),('XGBoost',px)]:
    for t in [.5,.4,.3,.2,.1]:
        pred=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pred).ravel(); rows.append(dict(model=nm,threshold=t,recall=tp/(tp+fn),precision=tp/(tp+fp) if tp+fp else np.nan,FN=int(fn),FP=int(fp)))
    # 재현율>=0.85가 되는 가장 높은 임계값
    ts=np.unique(p)[::-1]
    for t in ts:
        pred=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pred).ravel()
        if tp/(tp+fn)>=.85: rows.append(dict(model=nm,threshold=float(t),recall=tp/(tp+fn),precision=tp/(tp+fp),FN=int(fn),FP=int(fp),note='재현율≥0.85 최고 임계값')); break
sens=pd.DataFrame(rows); print(sens.round(3).to_string())
# 2) 혼동행렬 나란히
fig,axs=plt.subplots(1,2,figsize=(11,5))
for ax,nm in zip(axs,['RandomForest','XGBoost']):
    r=res[nm]; cm=np.array([[r['TN'],r['FP']],[r['FN'],r['TP']]]); ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):
            lab=[['정확','오경보'],['놓친 고장','적발']][i][j]; ax.text(j,i,f'{cm[i,j]:,}\n({lab})',ha='center',va='center',fontsize=12,color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set_xticks([0,1]); ax.set_xticklabels(['예측: 정상','예측: 고장']); ax.set_yticks([0,1]); ax.set_yticklabels(['실제: 정상','실제: 고장'])
    ax.set_title(f"{nm}\n재현율 {r['recall']*100:.1f}% · 놓친 {r['FN']} · 오경보 {r['FP']}",fontsize=11)
fig.suptitle('혼동행렬 비교 (원본 Test 2,000건, 임계값 0.5)',fontsize=13); plt.tight_layout(); plt.savefig(out+'/confusion_matrix_compare.png',dpi=130); plt.close()
# ROC 비교
fig,ax=plt.subplots(figsize=(6,5.5))
for nm,p,c in [('RandomForest',pr,B),('XGBoost',px,R)]:
    f,t,_=roc_curve(yte,p); ax.plot(f,t,color=c,lw=2,label=f"{nm} (AUC={res[nm]['roc_auc']:.3f})")
ax.plot([0,1],[0,1],'--',color='gray'); ax.set_xlabel('거짓 양성률'); ax.set_ylabel('참 양성률(재현율)'); ax.set_title('ROC 곡선 비교'); ax.legend(loc='lower right'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(out+'/roc_compare.png',dpi=130); plt.close()
# 3) 중요도 비교
fi_rf=pd.Series(rf.feature_importances_,index=Xtr.columns); fi_x=pd.Series(xgb.feature_importances_,index=Xtr.columns)
kor={'air_temp':'공기 온도','process_temp':'공정 온도','rpm':'회전 속도','torque':'토크','tool_wear':'공구 마모','temp_diff':'temp_diff(파생)','power':'power(파생)','tool_torque_interaction':'tool_torque_interaction(파생)','Type_H':'Type_H','Type_L':'Type_L','Type_M':'Type_M'}
order=(fi_rf+fi_x).sort_values().index; y=np.arange(len(order)); h=.38
fig,ax=plt.subplots(figsize=(9,6.5))
ax.barh(y+h/2,fi_rf[order],h,color=B,label='RandomForest (불순도 감소 평균)'); ax.barh(y-h/2,fi_x[order],h,color=R,label='XGBoost (gain 기반, 합=1 정규화)')
ax.set_yticks(y); ax.set_yticklabels([kor[k] for k in order]); ax.set_xlabel('중요도 (각 모델 내 합=1, 모델 간 절대값 직접 비교는 부정확)')
ax.set_title('Feature Importance 비교\n※ 모델 판단에 사용된 정도이며 실제 고장 원인의 증명은 아님',fontsize=11); ax.legend(loc='lower right'); plt.tight_layout(); plt.savefig(out+'/feature_importance_compare.png',dpi=130); plt.close()
print(pd.DataFrame({'RF':fi_rf,'XGB':fi_x}).round(4).sort_values('XGB',ascending=False).to_string())
# 저장
xgb.save_model(out+'/xgb_model.json'); joblib.dump(xgb,out+'/xgb_model.joblib')
json.dump(dict(model='XGBClassifier',params=params,threshold=.5,features=list(Xtr.columns)),open(out+'/xgb_config.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
pd.DataFrame({'y_true':yte,'rf_proba':pr,'xgb_proba':px,'rf_pred_0.5':(pr>=.5).astype(int),'xgb_pred_0.5':(px>=.5).astype(int)}).to_csv(out+'/test_predictions_both.csv',index=False)
json.dump(dict(metrics=res,threshold_sensitivity=sens.to_dict('records'),importance=dict(rf=fi_rf.to_dict(),xgb=fi_x.to_dict())),open(out+'/compare_metrics.json','w',encoding='utf-8'),ensure_ascii=False,indent=1,default=float)
