"""경영진 발표 HTML 빌드: 기존 결과 파일에서 수치를 읽어 executive_presentation.html에 내장하고, 기존 보고 수치와 대조(assert)한다.
사용: python -I build_presentation.py <PdM_Lab 루트> <발표 폴더>"""
import sys, json, csv
import numpy as np, pandas as pd
root, out = sys.argv[1:3]
S1 = root + '/Step 1_데이터 탐색 & 이상 탐지'; S2 = root + '/Step 2_고장 분류 모델 & ROI 시나리오'
feat = pd.read_csv(S1 + '/3. 파생변수 생성/ai4i2020_features.csv', encoding='utf-8-sig')
pred = pd.read_csv(S2 + '/3. XGBoost 학습 및 모델 비교/test_predictions_both.csv')
raw = pd.read_csv(S2 + '/1. 모델링 전처리/test_original_raw.csv')
cmpj = json.load(open(S2 + '/3. XGBoost 학습 및 모델 비교/compare_metrics.json', encoding='utf-8'))
cost = json.load(open(S2 + '/5. 도입 비용 시나리오/cost_scenario_results.json', encoding='utf-8'))
thr = pd.read_csv(S2 + '/4. 임계값 탐색/selected_thresholds_metrics.csv', encoding='utf-8-sig')
rfm = json.load(open(S2 + '/2. 랜덤포레스트 학습/rf_metrics.json', encoding='utf-8'))
checks = []
def chk(name, val, ref, src):
    try: ok = abs(float(val) - float(ref)) < 1e-9 if not isinstance(val, (tuple, str)) and not isinstance(ref, (tuple, str)) else str(val) == str(ref)
    except (TypeError, ValueError): ok = str(val) == str(ref)
    checks.append((name, val, ref, src, '일치' if ok else '불일치')); assert ok, (name, val, ref)

# ---------- 문제 발견 ----------
N = len(feat); F = int(feat['Machine failure'].sum())
types = {t: int(feat[t].sum()) for t in ['HDF', 'OSF', 'PWF', 'TWF', 'RNF']}
unlabeled = int(((feat['Machine failure'] == 1) & (feat[list(types)].sum(axis=1) == 0)).sum()); multi = int((feat[list(types)].sum(axis=1) > 1).sum())
chk('전체 샘플', N, 10000, 'Step 1·1번'); chk('고장 건수', F, 339, 'Step 1·1번'); chk('고장률(%)', f'{F / N * 100:.2f}', '3.39', 'Step 1·1번')
for t, v in {'HDF': 115, 'OSF': 98, 'PWF': 95, 'TWF': 46, 'RNF': 19}.items(): chk('고장 유형 ' + t, types[t], v, 'Step 1·1번')
chk('유형 라벨 합계', sum(types.values()), 373, 'Step 1·1번'); chk('유형 라벨 없는 고장', unlabeled, 9, 'Step 1·1번'); chk('2개 이상 유형 동시 표시', multi, 24, 'Step 1·1번')
tf = feat.groupby('Type')['Machine failure'].agg(['size', 'sum']).loc[['L', 'M', 'H']]
typefail = {k: dict(n=int(v['size']), f=int(v['sum']), rate=float(v['sum'] / v['size'] * 100)) for k, v in tf.iterrows()}
for k, r in {'L': '3.92', 'M': '2.77', 'H': '2.09'}.items(): chk('유형별 고장률 ' + k, f'{typefail[k]["rate"]:.2f}', r, 'Step 1·2번')
tq, rp = 'Torque [Nm]', 'Rotational speed [rpm]'
a, b = feat[feat['Machine failure'] == 0][tq], feat[feat['Machine failure'] == 1][tq]
sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2)); d_tq = (b.mean() - a.mean()) / sp
chk('토크 평균(정상)', f'{a.mean():.2f}', '39.63', 'Step 1·2번'); chk('토크 평균(고장)', f'{b.mean():.2f}', '50.17', 'Step 1·2번'); chk('토크 Cohen d', f'{d_tq:.2f}', '1.08', 'Step 1·2번')
def decile_rate(col, which):
    q = pd.qcut(feat[col].rank(method='first'), 10, labels=False); g = feat.groupby(q)['Machine failure'].mean() * 100
    return float(g.iloc[9] if which == 'top' else g.iloc[0])
dq_t, dq_r = decile_rate(tq, 'top'), decile_rate(rp, 'bottom')
chk('토크 상위 10% 고장률', f'{dq_t:.1f}', '18.4', 'Step 1·3번'); chk('회전 속도 하위 10% 고장률', f'{dq_r:.1f}', '16.6', 'Step 1·3번')
find = dict(tq_n=float(a.mean()), tq_f=float(b.mean()), d=float(d_tq), dec_t=dq_t, dec_r=dq_r, rpm_med_n=float(feat[feat['Machine failure'] == 0][rp].median()), rpm_med_f=float(feat[feat['Machine failure'] == 1][rp].median()))
chk('회전 속도 중앙값(정상)', int(find['rpm_med_n']), 1507, 'Step 1·2번'); chk('회전 속도 중앙값(고장)', int(find['rpm_med_f']), 1365, 'Step 1·2번')

# ---------- 모델 성능 (같은 Test, 같은 학습 데이터) ----------
y = pred.y_true.values; prob = {'rf': pred.rf_proba.values, 'xgb': pred.xgb_proba.values}
sel_thr = {'rf': 0.44, 'xgb': round(float(thr[(thr.model == 'XGBoost') & (thr['기준'] == 'Recall≥0.85 최대 임계값')].iloc[0].threshold), 4)}
sel_csv = {'rf': thr[(thr.model == 'RandomForest') & (thr['기준'] == 'Recall≥0.85 최대 임계값')].iloc[0], 'xgb': thr[(thr.model == 'XGBoost') & (thr['기준'] == 'Recall≥0.85 최대 임계값')].iloc[0]}
def at(p, t):
    pr = p >= t; tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum()); fn = int((~pr & (y == 1)).sum()); tn = int((~pr & (y == 0)).sum())
    P = tp / (tp + fp); R = tp / (tp + fn)
    return dict(thr=float(t), acc=(tp + tn) / len(y), prec=P, rec=R, f1=2 * P * R / (P + R), tp=tp, fp=fp, fn=fn, tn=tn)
models = {m: {'def': at(prob[m], 0.5), 'sel': at(prob[m], float(sel_csv[m].threshold))} for m in prob}
auc = {'rf': dict(roc=cmpj['metrics']['RandomForest']['roc_auc'], pr=cmpj['metrics']['RandomForest']['pr_auc']), 'xgb': dict(roc=cmpj['metrics']['XGBoost']['roc_auc'], pr=cmpj['metrics']['XGBoost']['pr_auc'])}
for m, name in [('rf', 'RandomForest'), ('xgb', 'XGBoost')]:       # 저장된 비교 지표(기본 0.5)·임계값 탐색 CSV와 대조
    for k, kk in [('acc', 'accuracy'), ('prec', 'precision'), ('rec', 'recall'), ('f1', 'f1')]:
        chk(f'{name} 0.5 {kk}', f'{models[m]["def"][k]:.4f}', f'{cmpj["metrics"][name][kk]:.4f}', 'Step 2·3번')
    r = sel_csv[m]; chk(f'{name} 선택 임계값 TP/FP/FN/TN', (models[m]['sel']['tp'], models[m]['sel']['fp'], models[m]['sel']['fn'], models[m]['sel']['tn']), (int(r.TP), int(r.FP), int(r.FN), int(r.TN)), 'Step 2·4번')
    chk(f'{name} 선택 임계값 재현율', f'{models[m]["sel"]["rec"]:.3f}', f'{r.recall:.3f}', 'Step 2·4번')
chk('RF ROC-AUC', f'{auc["rf"]["roc"]:.3f}', '0.974', 'Step 2·2번'); chk('XGB ROC-AUC', f'{auc["xgb"]["roc"]:.3f}', '0.977', 'Step 2·3번')
chk('RF 0.44 재현율(%)', f'{models["rf"]["sel"]["rec"] * 100:.1f}', '85.3', 'Step 2·4번'); chk('XGB 0.314 오경보', models['xgb']['sel']['fp'], 50, 'Step 2·4번')
chk('Test 건수', len(y), 2000, 'Step 2·1번'); chk('Test 고장', int(y.sum()), 68, 'Step 2·1번')
# 유형별 Test 분류 (임계값 0.44, 저장된 예측점수 사용)
ft = feat.set_index('UDI')[list(types)]; lab = ft.loc[raw.UDI.values]; p44 = prob['rf'] >= 0.44
typedet = {t: dict(n=int(((lab[t] == 1) & (y == 1)).sum()), tp=int(((lab[t] == 1) & (y == 1) & p44).sum())) for t in types}
chk('TWF Test 분류(0.44)', (typedet['TWF']['tp'], typedet['TWF']['n']), (3, 10), 'Step 3·6번')

# ---------- 비용 시나리오 ----------
A = cost['가정']; TP, FP = cost['측정']['TP'], cost['측정']['FP']
chk('비용 시나리오 TP/FP', (TP, FP), (models['rf']['sel']['tp'], models['rf']['sel']['fp']), 'Step 2·5번')
def scen(c): g = TP * (c - A['정비비_만원']) - FP * A['점검비_만원']; n = g - A['운영비_만원']; return dict(cost=c, prevent=TP * c, pm=TP * A['정비비_만원'], insp=FP * A['점검비_만원'], gross=g, sys=A['운영비_만원'], net=n, roi=n / A['운영비_만원'] * 100)
sens = [scen(s['고장비용']) for s in cost['민감도_고장비용']]
for s, j in zip(sens, cost['민감도_고장비용']): chk(f'순절감액 {s["cost"]}만 원', s['net'], j['net'], 'Step 2·5번'); chk(f'ROI {s["cost"]}만 원', round(s['roi'], 1), round(j['roi'], 1), 'Step 2·5번')
base = [s for s in sens if s['cost'] == A['고장비용_만원']][0]
chk('기준 순절감액(억)', f'{base["net"] / 10000:.2f}', '23.69', 'Step 2·5번'); chk('기준 ROI', f'{base["roi"]:.1f}', '1184.5', 'Step 2·5번'); chk('기준 절감액(억)', f'{base["gross"] / 10000:.2f}', '25.69', 'Step 2·5번')
chk('손익분기 고장비용(만원)', f'{cost["손익분기_고장비용"]:,.0f}', '916', 'Step 2·5번')

data = dict(N=N, F=F, types=types, unlabeled=unlabeled, multi=multi, labelsum=sum(types.values()), typefail=typefail, find=find, models=models, auc=auc, sel_thr=sel_thr,
            typedet=typedet, test_n=len(y), test_f=int(y.sum()),
            cost=dict(assume=A, tp=TP, fp=FP, sens=sens, base_cost=A['고장비용_만원'], breakeven=cost['손익분기_고장비용']))
tpl = open(out + '/presentation_template.html', encoding='utf-8').read()
html = tpl.replace('/*DATA*/', json.dumps(data, ensure_ascii=False, separators=(',', ':')))
open(out + '/executive_presentation.html', 'w', encoding='utf-8').write(html)
with open(out + '/number_check.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['항목', '발표 값', '기존 결과 값', '출처', '판정']); w.writerows(checks)
print('수치 대조', len(checks), '항목 모두 일치 · 저장', len(html) // 1024, 'KB')
