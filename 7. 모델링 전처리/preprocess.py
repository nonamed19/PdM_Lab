"""AI4I 모델링 전처리: 분할 -> 스케일링 -> SMOTENC(Train만) -> Type 원핫. 사용: python -I preprocess.py <features.csv> <출력폴더>"""
import sys, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTENC

src, out = sys.argv[1], sys.argv[2]
raw = pd.read_csv(src, encoding='utf-8-sig')
COLS = {'Air temperature [K]':'air_temp','Process temperature [K]':'process_temp','Rotational speed [rpm]':'rpm',
        'Torque [Nm]':'torque','Tool wear [min]':'tool_wear','temp_diff':'temp_diff','power':'power',
        'tool_torque_interaction':'tool_torque_interaction'}   # XGBoost가 거부하는 [ ] 문자 제거
CONT = list(COLS.values())
X = raw[list(COLS)+['Type']].rename(columns=COLS)
y = raw['Machine failure']
def rate(s): return f'{int(s.sum())}/{len(s)} ({s.mean()*100:.2f}%)'
log = {}

# 1) 입력 선택
log['1.입력(원본)'] = dict(shape=X.shape, 열=list(X.columns), 고장=rate(y))
# 2) 분할
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
log['2.Train'] = dict(shape=Xtr.shape, 고장=rate(ytr)); log['2.Test'] = dict(shape=Xte.shape, 고장=rate(yte))
Xte_raw = Xte.copy()
# 3) 스케일링 (Train에서만 학습)
sc = StandardScaler().fit(Xtr[CONT])
Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
Xtr_s[CONT] = sc.transform(Xtr[CONT]); Xte_s[CONT] = sc.transform(Xte[CONT])
log['3.Train 스케일 후'] = dict(shape=Xtr_s.shape, 연속형평균=float(Xtr_s[CONT].mean().abs().max()), 연속형표준편차=float(Xtr_s[CONT].std(ddof=0).mean()))
# 4) SMOTENC (Train만)
cat_idx = [list(Xtr_s.columns).index('Type')]
sm = SMOTENC(categorical_features=cat_idx, random_state=42, sampling_strategy=1.0)
Xtr_r, ytr_r = sm.fit_resample(Xtr_s, ytr)
Xtr_r = pd.DataFrame(Xtr_r, columns=Xtr_s.columns); Xtr_r[CONT] = Xtr_r[CONT].astype(float); ytr_r = pd.Series(ytr_r, name='Machine failure')
log['4.Train SMOTENC 후'] = dict(shape=Xtr_r.shape, 고장=rate(ytr_r), 정상=int((ytr_r==0).sum()), 합성샘플=len(Xtr_r)-len(Xtr_s))
# 5) Type 원핫 (Train에서 학습, Test에는 transform만)
ohe = OneHotEncoder(categories=[['H','L','M']], sparse_output=False, handle_unknown='error').fit(Xtr_r[['Type']])
oh_names = [f'Type_{c}' for c in ohe.categories_[0]]
def finish(df):
    oh = pd.DataFrame(ohe.transform(df[['Type']]), columns=oh_names, index=df.index)
    return pd.concat([df[CONT].reset_index(drop=True), oh.reset_index(drop=True)], axis=1)
Xtr_f, Xte_f = finish(Xtr_r), finish(Xte_s)
log['5.Train 최종'] = dict(shape=Xtr_f.shape, 고장=rate(ytr_r)); log['5.Test 최종'] = dict(shape=Xte_f.shape, 고장=rate(yte))

# ---- 검증 ----
chk = {}
for nm, d in [('Train', Xtr_f), ('Test', Xte_f)]:
    o = d[oh_names]
    chk[f'{nm} 원핫 값이 0/1'] = bool(o.isin([0, 1]).all().all())
    chk[f'{nm} 행 합=1'] = bool((o.sum(axis=1) == 1).all())
chk['스케일러가 Train 통계만 사용'] = bool(np.allclose(sc.mean_, Xtr[CONT].mean().values) and not np.allclose(sc.mean_, X[CONT].mean().values))
chk['Test 행수 유지(합성 없음)'] = len(Xte_f) == len(yte) == 2000
chk['Test 고장 비율 유지'] = abs(yte.mean() - y.mean()) < 1e-3
chk['Test 원본 행 일치(역변환)'] = bool(np.allclose(sc.inverse_transform(Xte_f[CONT]), Xte_raw[CONT].values) and (ohe.inverse_transform(Xte_f[oh_names]).ravel() == Xte_raw['Type'].values).all())
chk['Test 인덱스가 원본과 동일'] = bool(raw.loc[Xte_raw.index, 'UDI'].is_unique and (raw.loc[Xte_raw.index, 'Machine failure'].values == yte.values).all())
chk['Train∩Test 겹침 없음'] = len(set(Xtr.index) & set(Xte.index)) == 0
chk['Train 원본행이 SMOTENC 후에도 앞에 그대로'] = bool(np.allclose(Xtr_r[CONT].values[:len(Xtr_s)], Xtr_s[CONT].values))
log['검증'] = chk

# ---- 저장 ----
Xtr_f.to_csv(f'{out}/X_train.csv', index=False); ytr_r.to_csv(f'{out}/y_train.csv', index=False)
Xte_f.to_csv(f'{out}/X_test.csv', index=False); yte.reset_index(drop=True).to_csv(f'{out}/y_test.csv', index=False)
Xte_raw.assign(UDI=raw.loc[Xte_raw.index,'UDI'].values, y=yte.values).to_csv(f'{out}/test_original_raw.csv', index=False)
joblib.dump(dict(scaler=sc, onehot=ohe, continuous_cols=CONT, onehot_cols=oh_names, feature_names=CONT+oh_names, rename=COLS), f'{out}/preprocessor.joblib')
json.dump(log, open(f'{out}/preprocess_log.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
for k, v in log.items(): print(k, v)
