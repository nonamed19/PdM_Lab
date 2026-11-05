"""경영진 1페이지 보고서 빌드: 기존 결과 파일에서 수치를 읽어 보고서에 채우고, 기존 보고 수치와 대조(assert) 후
HTML/PDF 저장, PDF가 1페이지인지 확인, number_check.csv 저장.
사용: python -I build_report.py <PdM_Lab 루트> <보고서 폴더>"""
import sys, re, json, csv, pathlib
import numpy as np, pandas as pd
root, out = sys.argv[1:3]
S1 = root + '/Step 1_데이터 탐색 & 이상 탐지'; S2 = root + '/Step 2_고장 분류 모델 & ROI 시나리오'; S3 = root + '/Step 3_대시보드 & 보안_윤리'

feat = pd.read_csv(S1 + '/3. 파생변수 생성/ai4i2020_features.csv', encoding='utf-8-sig')
plog = json.load(open(S2 + '/1. 모델링 전처리/preprocess_log.json', encoding='utf-8'))
rfm = json.load(open(S2 + '/2. 랜덤포레스트 학습/rf_metrics.json', encoding='utf-8'))
thr = pd.read_csv(S2 + '/4. 임계값 탐색/selected_thresholds_metrics.csv', encoding='utf-8-sig')
cost = json.load(open(S2 + '/5. 도입 비용 시나리오/cost_scenario_results.json', encoding='utf-8'))
gsum = json.load(open(S3 + '/2. 교육용 경보 기준/grade_summary.json', encoding='utf-8'))
pred = pd.read_csv(S2 + '/3. XGBoost 학습 및 모델 비교/test_predictions_both.csv')
raw = pd.read_csv(S2 + '/1. 모델링 전처리/test_original_raw.csv')

# ---- 값 읽기 ----
N_ALL, F_ALL = len(feat), int(feat['Machine failure'].sum())
tr_keys = [k for k in plog if k.startswith('2.Train')][0]; res_keys = [k for k in plog if k.startswith('4.Train SMOTENC')][0]
TR_N, TR_RES = plog[tr_keys]['shape'][0], plog[res_keys]['shape'][0]
r44 = thr[(thr.model == 'RandomForest') & (thr['기준'] == 'Recall≥0.85 최대 임계값')].iloc[0]
r50 = thr[(thr.model == 'RandomForest') & (thr['기준'] == '기본 0.5')].iloc[0]
TEST_N = int(r44.TP + r44.FN + r44.FP + r44.TN); TEST_F = int(r44.TP + r44.FN)
base = cost['기준시나리오']; A = cost['가정']
g = {x['grade']: x for x in gsum['grades']}; gnames = list(g)         # 정상, 주의, 경고, 위험 순서
riskonly = gsum['cumulative'][2]
prev50 = [x for x in cost['민감도_예방률'] if abs(list(x.values())[0] - 0.5) < 1e-9][0]
# TWF: 임계값 0.44에서의 분류 (Test 라벨은 유형별 집계에만 사용)
ft = feat[['UDI', 'TWF']].set_index('UDI'); twf = ft.loc[raw.UDI.values, 'TWF'].values; y = pred.y_true.values; p44 = pred.rf_proba.values >= 0.44
TWF_N = int(((twf == 1) & (y == 1)).sum()); TWF_TP = int(((twf == 1) & (y == 1) & p44).sum())

V = dict(
 TEST_N=f'{TEST_N:,}', TEST_F=TEST_F, TP=int(r44.TP), FN=int(r44.FN), FP=int(r44.FP), TN=int(r44.TN), THR=f'{r44.threshold:.2f}', REC=f'{r44.recall * 100:.1f}', PREC=f'{r44.precision * 100:.1f}',
 REC50=f'{r50.recall * 100:.1f}', FN50=int(r50.FN), FP50=int(r50.FP), AUC=f'{rfm["roc_auc"]:.3f}',
 N_ALL=f'{N_ALL:,}', F_ALL=F_ALL, F_RATE=f'{F_ALL / N_ALL * 100:.2f}', TR_N=f'{TR_N:,}', TR_RES=f'{TR_RES:,}',
 ANN_N=f'{A["연간건수"]:,}', CF=f'{A["고장비용_만원"] / 10000 * 10000:,.0f}만 원/건', INSP=f'{A["점검비_만원"]:,}만 원', PM=f'{A["정비비_만원"]:,}만 원',
 SYS=f'{A["운영비_만원"] / 10000:.0f}', GROSS=f'{base["gross"] / 10000:.2f}', NET=f'{base["net"] / 10000:.2f}', ROI=f'{base["roi"]:,.1f}',
 NET50=f'{prev50["net"] / 10000:.2f}', ROI50=f'{prev50["roi"]:,.1f}', BE=f'{cost["손익분기_고장비용"]:,.0f}',
 G0=f'{g[gnames[0]]["samples"]:,}', G1=g[gnames[1]]['samples'], G2=g[gnames[2]]['samples'], G3=g[gnames[3]]['samples'],
 RISKONLY_MISS=TEST_F - riskonly['적발'], RISKONLY_MISS_PCT=f'{(TEST_F - riskonly["적발"]) / TEST_F * 100:.1f}', TWF_N=TWF_N, TWF_TP=TWF_TP,
)

# ---- 기존 보고 수치와 대조 (앞선 보고서·화면에 적은 값) ----
REF = [  # (표시 항목, 보고서 값, 기존 보고 값, 출처)
 ('전체 샘플 수', N_ALL, 10000, 'Step 1·1번'), ('전체 고장 건수', F_ALL, 339, 'Step 1·1번'), ('고장률(%)', V['F_RATE'], '3.39', 'Step 1·1번'),
 ('학습 데이터(분할 후)', TR_N, 8000, 'Step 2·1번'), ('학습 데이터(SMOTENC 후)', TR_RES, 15458, 'Step 2·1번'),
 ('Test 평가 건수', TEST_N, 2000, 'Step 2·1번'), ('Test 고장 건수', TEST_F, 68, 'Step 2·1번'),
 ('임계값', V['THR'], '0.44', 'Step 2·4번'), ('TP', V['TP'], 58, 'Step 2·4번'), ('놓친 고장 FN', V['FN'], 10, 'Step 2·4번'), ('오경보 FP', V['FP'], 41, 'Step 2·4번'),
 ('재현율(%)', V['REC'], '85.3', 'Step 2·4번'), ('정밀도(%)', V['PREC'], '58.6', 'Step 2·4번'), ('ROC-AUC', V['AUC'], '0.974', 'Step 2·2번'),
 ('기본 0.5 재현율(%)', V['REC50'], '82.4', 'Step 2·2번'), ('기본 0.5 놓친 고장', V['FN50'], 12, 'Step 2·2번'), ('기본 0.5 오경보', V['FP50'], 38, 'Step 2·2번'),
 ('연간 관측 건수 가정', A['연간건수'], 2000, 'Step 2·5번'), ('연간 고장 건수 가정', A['연간고장'], 68, 'Step 2·5번'), ('예방률 가정', A['예방률'], 1.0, 'Step 2·5번'),
 ('고장 비용(만 원)', A['고장비용_만원'], 5000, 'Step 2·5번'), ('점검비(만 원)', A['점검비_만원'], 100, 'Step 2·5번'), ('정비비(만 원)', A['정비비_만원'], 500, 'Step 2·5번'), ('운영비(만 원)', A['운영비_만원'], 20000, 'Step 2·5번'),
 ('절감액(운영비 차감 전, 억 원)', V['GROSS'], '25.69', 'Step 2·5번'), ('연간 순절감액(운영비 차감 후, 억 원)', V['NET'], '23.69', 'Step 2·5번'), ('ROI(%)', V['ROI'], '1,184.5', 'Step 2·5번'),
 ('손익분기 고장 비용(만 원)', V['BE'], '916', 'Step 2·5번'), ('예방 50% 순절감액(억 원)', V['NET50'], '10.64', 'Step 2·5번'), ('예방 50% ROI(%)', V['ROI50'], '532.0', 'Step 2·5번'),
 ('경보 등급 건수 정상', V['G0'], '1,866', 'Step 3·2번'), ('경보 등급 건수 주의', V['G1'], 57, 'Step 3·2번'), ('경보 등급 건수 경고', V['G2'], 35, 'Step 3·2번'), ('경보 등급 건수 위험', V['G3'], 42, 'Step 3·2번'),
 ('위험만 대응 시 놓친 고장', V['RISKONLY_MISS'], 31, 'Step 3·2번'), ('위험만 대응 시 놓친 비율(%)', V['RISKONLY_MISS_PCT'], '45.6', 'Step 3·2번'),
]
# 계산식 재검증: 순절감액 = TP×(고장비용−정비비) − FP×점검비 − 운영비
calc_gross = V['TP'] * (A['고장비용_만원'] - A['정비비_만원']) - V['FP'] * A['점검비_만원']; calc_net = calc_gross - A['운영비_만원']
assert calc_gross == base['gross'] and calc_net == base['net'] and abs(calc_net / A['운영비_만원'] * 100 - base['roi']) < 1e-9
rows = []
for name, val, ref, src in REF:
    ok = str(val) == str(ref); rows.append((name, val, ref, src, '일치' if ok else '불일치')); assert ok, (name, val, ref)
rows.append(('계산식 재계산: 순절감액(만 원)', calc_net, base['net'], 'TP×(고장비용−정비비)−FP×점검비−운영비', '일치'))
rows.append(('계산식 재계산: ROI(%)', round(calc_net / A['운영비_만원'] * 100, 1), round(base['roi'], 1), '순절감액÷운영비×100', '일치'))
print('수치 대조', len(rows), '항목 모두 일치;', 'TWF', TWF_TP, '/', TWF_N)

# ---- HTML 생성 ----
html = open(out + '/report_template.html', encoding='utf-8').read()
html = re.sub(r'%%([A-Z0-9_]+)%%', lambda m: str(V[m.group(1)]), html)
assert not re.search(r'%%[A-Z]', html); html = html.replace('%%', '%')           # 리터럴 % (100%% 등)
open(out + '/executive_report.html', 'w', encoding='utf-8').write(html)
with open(out + '/number_check.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['항목', '보고서 값', '기존 결과 값', '출처/계산식', '판정']); w.writerows(rows)

# ---- PDF (A4 1페이지) ----
from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    b = pw.chromium.launch(channel='chrome', headless=True); pg = b.new_page()
    pg.goto(pathlib.Path(out + '/executive_report.html').resolve().as_uri()); pg.emulate_media(media='print')
    pg.pdf(path=out + '/executive_report.pdf', format='A4', print_background=True, margin=dict(top='9mm', bottom='9mm', left='10mm', right='10mm'))
    b.close()
import fitz
doc = fitz.open(out + '/executive_report.pdf'); print('PDF 페이지 수:', len(doc))
doc[0].get_pixmap(dpi=110).save(out + '/executive_report_preview.png')
assert len(doc) == 1, '1페이지를 넘음'
