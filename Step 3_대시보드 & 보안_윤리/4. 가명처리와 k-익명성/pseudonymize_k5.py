"""가명처리 + k-익명성(k>=5) 시연 (공개 합성 데이터 AI4I 2020).
사용: python -I pseudonymize_k5.py <원본 CSV> <공유용 출력 폴더> <비공개 대응표 폴더>
- UDI -> 무작위 대체 ID, Product ID -> 대체 시리얼, 행 순서는 무작위로 섞음
- 준식별자: Type + 센서 5개 (구간 일반화, 양 끝은 열린 구간), k<5 클래스의 레코드는 억제(삭제)
- 대응표는 비공개 폴더에만 저장하고 공유용 처리본에는 넣지 않음
※ 가명처리와 k=5는 완전 익명화나 외부 공유 승인을 뜻하지 않음 (교육용 시연)"""
import sys, json, secrets, string, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
src, out, priv = sys.argv[1:4]
K = 5
raw = pd.read_csv(src, encoding='utf-8-sig')
SENS = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
SHORT = {'Air temperature [K]': 'air_temp', 'Process temperature [K]': 'process_temp', 'Rotational speed [rpm]': 'rpm', 'Torque [Nm]': 'torque', 'Tool wear [min]': 'tool_wear'}
KOR = {'air_temp': '공기 온도 [K]', 'process_temp': '공정 온도 [K]', 'rpm': '회전 속도 [rpm]', 'torque': '토크 [Nm]', 'tool_wear': '공구 마모 [min]'}
FLAGS = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
EDGES = {'Air temperature [K]': [299, 301], 'Process temperature [K]': [309.5, 310.5], 'Rotational speed [rpm]': [1450, 1600], 'Torque [Nm]': [35, 45], 'Tool wear [min]': [70, 140]}
QI = ['Type'] + SENS

# ---------- 원본의 재식별 위험 (정확한 값 기준) ----------
orig_sz = raw.groupby(QI)['UDI'].transform('size')
orig = dict(classes=int(raw.groupby(QI).ngroups), min_k=int(orig_sz.min()), unique_pct=float((orig_sz == 1).mean() * 100))
print('원본 준식별자(정확값):', orig)

# ---------- 구간 일반화 ----------
def label(c, e, i):
    u = c.split('[')[1].rstrip(']'); f = lambda v: f'{v:g}'
    return f'{f(e[0])} {u} 미만' if i == 0 else (f'{f(e[1])} {u} 이상' if i == len(e) else f'{f(e[i-1])} 이상 ~ {f(e[i])} 미만 {u}')
G = raw[['Type']].copy(); mid = pd.DataFrame(index=raw.index)
for c in SENS:
    e = EDGES[c]; idx = np.digitize(raw[c], e)
    G[SHORT[c] + '_구간'] = [label(c, e, i) for i in idx]
    w = e[1] - e[0]; m = [e[0] - w / 2] + [(e[i] + e[i + 1]) / 2 for i in range(len(e) - 1)] + [e[-1] + w / 2]   # 구간 대표값(열린 구간은 인접 구간 폭의 절반)
    mid[SHORT[c]] = np.array(m)[idx]
qi_g = list(G.columns)
size = G.groupby(qi_g)['Type'].transform('size'); sup = (size < K)
print('일반화 후 클래스', G.groupby(qi_g).ngroups, '최소 k(억제 전)', int(size.min()), '억제 대상 레코드', int(sup.sum()), '(고장', int(raw.loc[sup, 'Machine failure'].sum()), ')')
keep = ~sup
Gk = G[keep]; sz2 = Gk.groupby(qi_g)['Type'].transform('size')
assert sz2.min() >= K, '억제 후에도 k<5 클래스가 남음'
print('억제 후 클래스', Gk.groupby(qi_g).ngroups, '최소 k', int(sz2.min()), '남은 레코드', int(keep.sum()))

# ---------- l-다양성(민감 속성: 고장 여부) ----------
cls = Gk.assign(fail=raw.loc[keep, 'Machine failure'].values).groupby(qi_g)['fail'].agg(['size', 'sum'])
allfail = cls[cls['sum'] == cls['size']]; allnorm = cls[cls['sum'] == 0]
ldiv = dict(classes=int(len(cls)), homogeneous_normal_classes=int(len(allnorm)), homogeneous_normal_records=int(allnorm['size'].sum()),
            homogeneous_fail_classes=int(len(allfail)), homogeneous_fail_records=int(allfail['size'].sum()))
ldiv['homogeneous_total_records'] = ldiv['homogeneous_normal_records'] + ldiv['homogeneous_fail_records']
ldiv['homogeneous_total_pct'] = ldiv['homogeneous_total_records'] / int(keep.sum()) * 100
print('l-다양성(l=2) 위반(고장 여부가 클래스 내 모두 같음):', ldiv)

# ---------- 가명 ID 생성 (무작위, 재현 불가) + 행 순서 섞기 ----------
def uniq(n, fn):
    s = set()
    while len(s) < n: s.add(fn())
    return list(s)
alpha = string.ascii_uppercase + string.digits
ids = uniq(len(raw), lambda: 'S-' + ''.join(secrets.choice(alpha) for _ in range(8)))
sers = uniq(len(raw), lambda: 'SN-' + ''.join(secrets.choice(alpha) for _ in range(10)))
secrets.SystemRandom().shuffle(ids); secrets.SystemRandom().shuffle(sers)
mapping = pd.DataFrame({'new_sample_id': ids, 'original_UDI': raw.UDI.values, 'new_serial': sers, 'original_Product_ID': raw['Product ID'].values, 'suppressed_in_shared_file': sup.values})
shared = pd.concat([pd.DataFrame({'sample_id': ids, 'serial': sers}), G, raw[FLAGS]], axis=1)[keep.values]
order = np.random.default_rng(secrets.randbits(64)).permutation(len(shared))        # 원본 행 순서 추적 방지
shared = shared.iloc[order].reset_index(drop=True)
shared.to_csv(out + '/processed_k5.csv', index=False, encoding='utf-8-sig')
mapping.to_csv(priv + '/id_mapping_PRIVATE.csv', index=False, encoding='utf-8-sig')
# 공유용 처리본에 원본 식별자/대응표가 없는지 검사
txt = open(out + '/processed_k5.csv', encoding='utf-8-sig').read()
header = txt.splitlines()[0]   # 무작위 ID 문자열에 'UDI' 같은 글자가 우연히 들어갈 수 있으므로 본문이 아닌 열 이름으로 검사
assert not (set(raw['Product ID']) & set(shared.serial)) and 'original' not in header and 'UDI' not in header and 'Product ID' not in header
assert not (set(map(str, raw.UDI)) & set(shared.sample_id))
assert (shared.groupby(qi_g)['Type'].transform('size') >= K).all()
print('공유용 처리본', shared.shape, '| 대응표(비공개)', mapping.shape)

# ---------- 원본 vs 처리본 비교 (같은 레코드 기준) ----------
R = raw[keep]; M = mid[keep]
stat = []
for c in SENS:
    s = SHORT[c]; stat.append(dict(변수=KOR[s], 원본평균=R[c].mean(), 처리본평균=M[s].mean(), 원본표준편차=R[c].std(), 처리본표준편차=M[s].std(), 원본고유값수=int(R[c].nunique()), 처리본구간수=int(M[s].nunique())))
stat = pd.DataFrame(stat); print(stat.round(2).to_string())
fr = lambda d: d['Machine failure'].mean() * 100
fail = dict(원본전체=fr(raw), 처리본=fr(R), 억제된_레코드=int(sup.sum()), 억제된_고장=int(raw.loc[sup, 'Machine failure'].sum()), 억제_레코드_고장률=fr(raw[sup]))
tfail = pd.DataFrame({'원본': raw.groupby('Type')['Machine failure'].mean() * 100, '처리본': R.groupby('Type')['Machine failure'].mean() * 100, '원본건수': raw.groupby('Type').size(), '처리본건수': R.groupby('Type').size()}).loc[['L', 'M', 'H']]
print(fail); print(tfail.round(2))
corr = pd.DataFrame({KOR[SHORT[c]]: [np.corrcoef(R[c], R['Machine failure'])[0, 1], np.corrcoef(M[SHORT[c]], R['Machine failure'])[0, 1]] for c in SENS}, index=['원본', '처리본']).T
cc = lambda d, a, b: np.corrcoef(d[a], d[b])[0, 1]
pair = dict(토크_회전속도_원본=cc(R, 'Torque [Nm]', 'Rotational speed [rpm]'), 토크_회전속도_처리본=cc(M, 'torque', 'rpm'))
print(corr.round(3)); print(pair)
# 단순 RF 기준선(같은 레코드·같은 분할): 원본 센서 vs 일반화 대표값
y = R['Machine failure'].values; Xo = R[SENS].values; Xg = M.values
tr, te = train_test_split(np.arange(len(R)), test_size=.2, stratify=y, random_state=42)
auc = {}
for nm, X in [('원본 센서 5개', Xo), ('일반화 대표값 5개', Xg)]:
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced_subsample').fit(X[tr], y[tr]); p = rf.predict_proba(X[te])[:, 1]
    auc[nm] = dict(roc_auc=roc_auc_score(y[te], p), pr_auc=average_precision_score(y[te], p))
print(auc)

# ---------- 대안 설계 비교(개인정보-유용성 trade-off) ----------
ALT = {'A (선택) 센서별 3구간': EDGES,
       'E 더 거친 일반화': {'Air temperature [K]': [300], 'Process temperature [K]': [310], 'Rotational speed [rpm]': [1400, 1650], 'Torque [Nm]': [33, 47], 'Tool wear [min]': [70, 150]},
       'F 더 세밀한 일반화': {'Air temperature [K]': [298, 300, 302], 'Process temperature [K]': [309, 310.5], 'Rotational speed [rpm]': [1400, 1500, 1700], 'Torque [Nm]': [30, 40, 50], 'Tool wear [min]': [50, 100, 160]}}
alt = []
for nm, ed in ALT.items():
    g = raw[['Type']].copy()
    for c in SENS: g[c] = np.digitize(raw[c], ed[c])
    s_ = g.groupby(list(g.columns))['Type'].transform('size'); sp = s_ < K
    alt.append(dict(설계=nm, 구간수=sum(len(v) + 1 for v in ed.values()), 클래스수=int(g[~sp].groupby(list(g.columns)).ngroups), 억제레코드=int(sp.sum()), 억제율=sp.mean() * 100, 억제된_고장=int(raw.loc[sp, 'Machine failure'].sum())))
alt = pd.DataFrame(alt); print(alt.round(2).to_string())

json.dump(dict(k=K, orig=orig, edges=EDGES, classes_before_suppression=int(G.groupby(qi_g).ngroups), classes_after=int(Gk.groupby(qi_g).ngroups), min_k_after=int(sz2.min()), suppressed=int(sup.sum()),
               kept=int(keep.sum()), ldiv=ldiv, fail=fail, stat=stat.to_dict('records'), type_fail=tfail.reset_index().to_dict('records'), corr=corr.reset_index().to_dict('records'), pair=pair, auc=auc, alt=alt.to_dict('records')),
          open(out + '/k_anonymity_report.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)

# ---------- 그림 ----------
WARN = '※ 교육용 시연 · 가명처리/k=5는 완전 익명화나 외부 공유 승인이 아님'
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axs[0]; bins = np.arange(0.5, 40.5, 1)
ax.hist(np.clip(orig_sz, 1, 40), bins=bins, color='#d62728', alpha=.7, label='원본(정확한 값): 모든 레코드 k=1')
ax.hist(np.clip(sz2, 1, 40), bins=bins, color='#4c78a8', alpha=.8, label='일반화 + 억제 후')
ax.axvline(K - .5, ls='--', color='k'); ax.text(K, ax.get_ylim()[1] * .85, f'k = {K}', fontsize=11)
ax.set_xlabel('같은 준식별자 조합을 가진 레코드 수 (클래스 크기, 40 초과는 40으로 표시)'); ax.set_ylabel('레코드 수'); ax.set_yscale('log'); ax.legend(); ax.set_title('준식별자 클래스 크기 분포')
ax = axs[1]; w = .38; xs = np.arange(3)
ax.bar(xs - w / 2, tfail['원본'], w, color='#999999', label='원본'); ax.bar(xs + w / 2, tfail['처리본'], w, color='#4c78a8', label='처리본(억제 후)')
for i, (a, b) in enumerate(zip(tfail['원본'], tfail['처리본'])): ax.text(i - w / 2, a + .05, f'{a:.2f}%', ha='center', fontsize=9); ax.text(i + w / 2, b + .05, f'{b:.2f}%', ha='center', fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels([f'Type {t}' for t in ['L', 'M', 'H']]); ax.set_ylabel('고장률 [%]'); ax.legend(); ax.set_title(f'제품 유형별 고장률 (전체 {fail["원본전체"]:.2f}% → {fail["처리본"]:.2f}%)')
fig.suptitle('k-익명성 달성 결과와 고장률 변화  |  ' + WARN, fontsize=11); plt.tight_layout(); plt.savefig(out + '/k_anonymity_overview.png', dpi=130); plt.close()

fig, axs = plt.subplots(1, 5, figsize=(18, 3.9))
for ax, c in zip(axs, SENS):
    ax.hist(raw[c], bins=40, color='#999999'); e = EDGES[c]
    for v in e: ax.axvline(v, color='#d62728', ls='--')
    sh = np.bincount(np.digitize(R[c], e), minlength=3) / len(R) * 100; lo, hi = raw[c].min(), raw[c].max(); ys = ax.get_ylim()[1]
    for xx, s_, hh in zip([(lo + e[0]) / 2, (e[0] + e[1]) / 2, (e[1] + hi) / 2], sh, [.92, .80, .92]): ax.text(xx, ys * hh, f'{s_:.0f}%', ha='center', color='#d62728', fontsize=11, fontweight='bold')
    ax.set_title(KOR[SHORT[c]]); ax.set_xlabel('원본 값 (빨간 점선 = 구간 경계)')
axs[0].set_ylabel('원본 샘플 수'); fig.suptitle('일반화로 사라지는 정보: 원본 분포와 3개 구간 (빨간 %는 처리본 구간별 비율)  |  ' + WARN, fontsize=11)
plt.tight_layout(); plt.savefig(out + '/generalization_bins.png', dpi=130); plt.close()
