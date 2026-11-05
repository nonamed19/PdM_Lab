"""RF 학습/평가. 사용: python -I train_rf.py <전처리폴더> <출력폴더>"""
import sys, json, joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, average_precision_score
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
src,out=sys.argv[1],sys.argv[2]
Xtr=pd.read_csv(src+'/X_train.csv'); ytr=pd.read_csv(src+'/y_train.csv').iloc[:,0]
Xte=pd.read_csv(src+'/X_test.csv'); yte=pd.read_csv(src+'/y_test.csv').iloc[:,0]
params=dict(n_estimators=200,random_state=42,class_weight='balanced')
rf=RandomForestClassifier(**params).fit(Xtr,ytr)
proba=rf.predict_proba(Xte)[:,1]; pred=(proba>=0.5).astype(int)
tn,fp,fn,tp=confusion_matrix(yte,pred).ravel()
rep=classification_report(yte,pred,target_names=['정상','고장'],output_dict=True,digits=4)
auc=roc_auc_score(yte,proba); ap=average_precision_score(yte,proba)
print('TN FP FN TP',tn,fp,fn,tp); print(classification_report(yte,pred,target_names=['정상','고장'],digits=4)); print('AUC',round(auc,4),'PR-AUC',round(ap,4))
B,R='#1f77b4','#d62728'
# 1) 혼동행렬
cm=np.array([[tn,fp],[fn,tp]]); fig,ax=plt.subplots(figsize=(6,5.2)); im=ax.imshow(cm,cmap='Blues')
for i in range(2):
    for j in range(2):
        lab=[['정상→정상\n(정확)','정상→고장\n(오경보)'],['고장→정상\n(놓친 고장)','고장→고장\n(적발)']][i][j]
        ax.text(j,i,f'{cm[i,j]:,}\n{lab}',ha='center',va='center',fontsize=12,color='white' if cm[i,j]>cm.max()/2 else 'black')
ax.set_xticks([0,1]); ax.set_xticklabels(['예측: 정상','예측: 고장']); ax.set_yticks([0,1]); ax.set_yticklabels(['실제: 정상','실제: 고장'])
ax.set_title(f'혼동행렬 (원본 Test 2,000건, 임계값 0.5)\n재현율 {tp/(tp+fn)*100:.1f}% · 놓친 고장 {fn}건 · 오경보 {fp}건',fontsize=11)
plt.tight_layout(); plt.savefig(out+'/confusion_matrix.png',dpi=130); plt.close()
# 3) ROC
fpr,tpr,_=roc_curve(yte,proba); fig,ax=plt.subplots(figsize=(6,5.5))
ax.plot(fpr,tpr,color=R,lw=2,label=f'랜덤포레스트 (AUC = {auc:.3f})'); ax.plot([0,1],[0,1],'--',color='gray',label='무작위 (AUC = 0.5)')
ax.plot(fp/(fp+tn),tp/(tp+fn),'o',color='k',ms=8,label='임계값 0.5 지점')
ax.set_xlabel('거짓 양성률 (오경보 비율)'); ax.set_ylabel('참 양성률 (재현율)'); ax.set_title('ROC 곡선 (원본 Test)'); ax.legend(loc='lower right'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(out+'/roc_curve.png',dpi=130); plt.close()
# 4) 중요도
imp=pd.Series(rf.feature_importances_,index=Xtr.columns).sort_values(ascending=False); top=imp.head(10)[::-1]
kor={'air_temp':'공기 온도','process_temp':'공정 온도','rpm':'회전 속도','torque':'토크','tool_wear':'공구 마모','temp_diff':'temp_diff (파생)','power':'power (파생)','tool_torque_interaction':'tool_torque_interaction (파생)','Type_H':'Type_H','Type_L':'Type_L','Type_M':'Type_M'}
fig,ax=plt.subplots(figsize=(8,5.5)); ax.barh([kor[k] for k in top.index],top.values,color='#4c78a8')
for i,v in enumerate(top.values): ax.text(v+.002,i,f'{v:.3f}',va='center')
ax.set_xlabel('중요도 (불순도 감소 기반)'); ax.set_title('Feature Importance 상위 10개\n※ 모델 판단에 사용된 정도이며 실제 고장 원인의 증명은 아님',fontsize=11)
plt.tight_layout(); plt.savefig(out+'/feature_importance.png',dpi=130); plt.close()
print(imp.round(4).to_string())
# 저장
joblib.dump(rf,out+'/rf_model.joblib')
json.dump(dict(model='RandomForestClassifier',params=params,threshold=0.5,train_shape=list(Xtr.shape),train_class_counts={int(k):int(v) for k,v in ytr.value_counts().items()},features=list(Xtr.columns)),open(out+'/rf_config.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
pd.DataFrame({'y_true':yte,'proba_failure':proba,'pred_0.5':pred}).to_csv(out+'/test_predictions.csv',index=False)
json.dump(dict(confusion=dict(TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp)),recall=float(tp/(tp+fn)),precision=float(tp/(tp+fp)) if tp+fp else None,f1=rep['고장']['f1-score'],accuracy=rep['accuracy'],roc_auc=float(auc),pr_auc=float(ap),report=rep,feature_importance=imp.to_dict()),open(out+'/rf_metrics.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
