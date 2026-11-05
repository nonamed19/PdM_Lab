"""경영진 대시보드 빌드. 기존 결과 파일에서 숫자를 읽어 dashboard.html에 내장하고 기존 보고 수치와 대조.
사용: python -I build_dashboard.py <PdM_Lab 루트> <템플릿> <출력 HTML>"""
import sys, json, numpy as np, pandas as pd
root, tpl, outp = sys.argv[1:4]
S1 = root + '/Step 1_데이터 탐색 & 이상 탐지'; S2 = root + '/Step 2_고장 분류 모델 & ROI 시나리오'
feat = pd.read_csv(S1 + '/3. 파생변수 생성/ai4i2020_features.csv', encoding='utf-8-sig')
pred = pd.read_csv(S2 + '/3. XGBoost 학습 및 모델 비교/test_predictions_both.csv')
rfm = json.load(open(S2 + '/2. 랜덤포레스트 학습/rf_metrics.json', encoding='utf-8'))
cmp_ = json.load(open(S2 + '/3. XGBoost 학습 및 모델 비교/compare_metrics.json', encoding='utf-8'))
cost = json.load(open(S2 + '/5. 도입 비용 시나리오/cost_scenario_results.json', encoding='utf-8'))
thr_csv = pd.read_csv(S2 + '/4. 임계값 탐색/selected_thresholds_metrics.csv', encoding='utf-8-sig')

N = len(feat); nfail = int(feat['Machine failure'].sum())
types = {t: int(feat[t].sum()) for t in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']}
unlabeled = int(((feat['Machine failure'] == 1) & (feat[list(types)].sum(axis=1) == 0)).sum())
def conf(t):
    p = pred.rf_proba.values >= t; y = pred.y_true.values == 1
    return dict(TP=int((p & y).sum()), FP=int((p & ~y).sum()), FN=int((~p & y).sum()), TN=int((~p & ~y).sum()), threshold=t)
THR = 0.44; c44, c50 = conf(THR), conf(0.5)
for c in (c44, c50): c['recall'] = c['TP'] / (c['TP'] + c['FN']); c['precision'] = c['TP'] / (c['TP'] + c['FP'])
base = cost['기준시나리오']
cols = {'air': 'Air temperature [K]', 'proc': 'Process temperature [K]', 'rpm': 'Rotational speed [rpm]', 'torque': 'Torque [Nm]', 'wear': 'Tool wear [min]'}
sensors = {k: [float(x) for x in feat[v].tolist()] for k, v in cols.items()}
assert (feat['UDI'].values == np.arange(1, N + 1)).all()
imp_rf = rfm['feature_importance']; imp_x = cmp_['importance']['xgb']

# --- 기존 결과와 대조 (기준값은 앞선 보고서에 적은 수치) ---
REF = [
 ('전체 관측 샘플 수', N, 10000, 'Step 1 · 1번 (기초 EDA)'),
 ('전체 고장 건수', nfail, 339, 'Step 1 · 1번'),
 ('고장률(%)', round(nfail / N * 100, 2), 3.39, 'Step 1 · 1번'),
 ('고장 유형 HDF', types['HDF'], 115, 'Step 1 · 1번'), ('고장 유형 OSF', types['OSF'], 98, 'Step 1 · 1번'),
 ('고장 유형 PWF', types['PWF'], 95, 'Step 1 · 1번'), ('고장 유형 TWF', types['TWF'], 46, 'Step 1 · 1번'), ('고장 유형 RNF', types['RNF'], 19, 'Step 1 · 1번'),
 ('Test 평가 건수', c44['TP'] + c44['FP'] + c44['FN'] + c44['TN'], 2000, 'Step 2 · 1번 (전처리)'),
 ('Test 고장 건수', c44['TP'] + c44['FN'], 68, 'Step 2 · 1번'),
 ('임계값 0.44 TP', c44['TP'], 58, 'Step 2 · 4번 (임계값 탐색)'), ('임계값 0.44 FP', c44['FP'], 41, 'Step 2 · 4번'),
 ('임계값 0.44 FN', c44['FN'], 10, 'Step 2 · 4번'), ('임계값 0.44 TN', c44['TN'], 1891, 'Step 2 · 4번'),
 ('임계값 0.44 재현율(%)', round(c44['recall'] * 100, 1), 85.3, 'Step 2 · 4번'),
 ('임계값 0.5 재현율(%)', round(c50['recall'] * 100, 1), 82.4, 'Step 2 · 2번 (RF 학습)'),
 ('연간 순절감액(억 원)', round(base['net'] / 10000, 2), 23.69, 'Step 2 · 5번 (도입 비용 시나리오)'),
 ('ROI(%)', round(base['roi'], 1), 1184.5, 'Step 2 · 5번'),
 ('RF 중요도 1위 rpm', round(imp_rf['rpm'], 3), 0.214, 'Step 2 · 2번'),
]
ver = []
for name, val, ref, src in REF:
    ok = abs(val - ref) < 1e-9; ver.append(dict(name=name, val=val, ref=ref, src=src, ok=ok))
    assert ok, (name, val, ref)
# 저장된 임계값 CSV와도 대조
r = thr_csv[(thr_csv.model == 'RandomForest') & (thr_csv['기준'] == 'Recall≥0.85 최대 임계값')].iloc[0]
assert (int(r.TP), int(r.FP), int(r.FN), int(r.TN)) == (c44['TP'], c44['FP'], c44['FN'], c44['TN']) and abs(r.threshold - THR) < 1e-9
print('대조', len(ver), '항목 모두 일치')

data = dict(N=N, nfail=nfail, types=types, unlabeled=unlabeled, conf={'0.44': c44, '0.5': c50},
            imp=dict(rf=imp_rf, xgb=imp_x), sensors=sensors, fail=feat['Machine failure'].astype(int).tolist(),
            cost=dict(assume=cost['가정'], base=base, breakeven=cost['손익분기_고장비용'],
                      sens=[dict(c=s['고장비용'], net=s['net'], roi=s['roi']) for s in cost['민감도_고장비용']],
                      meas=cost['측정'], before=cost['도입전후']['도입전'], after=cost['도입전후']['도입후']),
            ver=ver, model='RandomForest', test_n=2000)
html = open(tpl, encoding='utf-8').read().replace('/*DATA*/', json.dumps(data, ensure_ascii=False, separators=(',', ':')))
open(outp, 'w', encoding='utf-8').write(html); print('저장', outp, round(len(html) / 1024), 'KB')
