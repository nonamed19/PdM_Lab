"""교육용 경보 등급 분류/시각화. 사용: python -I alert_grades.py <test_predictions_both.csv> <test_original_raw.csv> <출력폴더>"""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
pred, raw, out = sys.argv[1:4]
p = pd.read_csv(pred); r = pd.read_csv(raw)
assert (p.y_true.values == r.y.values).all()                      # 같은 행 순서인지 확인
s = np.round(p.rf_proba.values, 9)                                 # 부동소수 오차로 경계값이 하위 등급에 빠지지 않게 반올림
GR = ['정상', '주의', '경고', '위험']; LO = {'정상': 0.0, '주의': 0.30, '경고': 0.60, '위험': 0.85}
grade = np.select([s >= .85, s >= .60, s >= .30], ['위험', '경고', '주의'], '정상')   # 경계값은 상위 등급
df = pd.DataFrame({'sample_id_UDI': r.UDI.values, 'rf_score': p.rf_proba.values, 'grade': grade, 'actual_failure': p.y_true.values})
df.to_csv(out + '/test_alert_grades.csv', index=False, encoding='utf-8-sig')
t = df.groupby('grade').agg(samples=('actual_failure', 'size'), observed_failures=('actual_failure', 'sum')).reindex(GR)
t['share_pct'] = (t.samples / len(df) * 100).round(2); t['observed_failure_share_pct'] = (t.observed_failures / t.samples * 100).round(1)
t.to_csv(out + '/grade_distribution.csv', encoding='utf-8-sig'); print(t)
for b in (.30, .60, .85): print('경계값', b, '정확히 같은 샘플', int((s == b).sum()), '→ 상위 등급 확인:', sorted(set(grade[s == b])))
assert t.samples.sum() == 2000 and t.observed_failures.sum() == 68
# 누적 기준(참고): 어느 등급 이상을 알림으로 볼 때
cum = []
for g in ['주의', '경고', '위험']:
    m = s >= LO[g]; tp = int((m & (p.y_true.values == 1)).sum()); fp = int((m & (p.y_true.values == 0)).sum())
    cum.append(dict(기준=f'{g} 이상', 알림=int(m.sum()), 적발=tp, 오경보=fp, 재현율=round(tp / 68 * 100, 1))); 
print(pd.DataFrame(cum))
json.dump(dict(grades=t.reset_index().to_dict('records'), cumulative=cum), open(out + '/grade_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
COL = {'정상': '#4c78a8', '주의': '#e6c229', '경고': '#f28e2b', '위험': '#d62728'}
WARN = '※ 교육용 예시 등급 · 점수는 실제 고장 확률로 보정되지 않음 · 행 번호는 설비 ID/시간이 아님'
# 그림 1: 등급별 분포
fig, ax = plt.subplots(figsize=(9, 5.2))
bars = ax.bar(GR, t.samples.values, color=[COL[g] for g in GR], width=.6)
for b, g in zip(bars, GR):
    n = int(t.samples[g]); ax.text(b.get_x() + b.get_width() / 2, n + 25, f'{n:,}건\n({n / 2000 * 100:.1f}%)', ha='center', fontsize=11, fontweight='bold' if g == '위험' else None)
ax.annotate(f'위험 등급 관측 샘플 {int(t.samples["위험"])}건', xy=(3, t.samples['위험'] + 190), xytext=(1.7, 900), fontsize=11, color='#d62728', arrowprops=dict(arrowstyle='->', color='#d62728'))
ax.set_ylim(0, 2150); ax.set_ylabel('Test 샘플 수 (건)'); ax.set_xlabel('경보 등급 (경계값은 상위 등급)')
ax.set_xticks(range(4)); ax.set_xticklabels([f'{g}\n{l}' for g, l in zip(GR, ['< 30%', '30~60%', '60~85%', '≥ 85%'])])
ax.set_title('Test 2,000건의 교육용 경보 등급 분포 (랜덤포레스트 점수 기준)', fontsize=12); fig.text(.5, .01, WARN, ha='center', fontsize=9, color='#b00020')
plt.tight_layout(rect=(0, .03, 1, 1)); plt.savefig(out + '/grade_distribution.png', dpi=130); plt.close()
# 그림 2: 점수 히스토그램과 경계
fig, ax = plt.subplots(figsize=(9, 4.8)); bins = np.round(np.linspace(0, 1, 41), 9)
for g, lo, hi in [('정상', 0, .3), ('주의', .3, .6), ('경고', .6, .85), ('위험', .85, 1.0001)]:
    m = (s >= lo) & (s < hi) if g != '위험' else s >= lo; ax.hist(s[m], bins=bins, color=COL[g], label=f'{g} {int(m.sum()):,}건')
for b in (.3, .6, .85): ax.axvline(b, ls='--', color='gray', lw=1)
ax.set_yscale('log'); ax.set_xlabel('랜덤포레스트 예측점수 (0~1, 확률로 보정되지 않음)'); ax.set_ylabel('샘플 수 (로그 스케일)'); ax.legend(loc='upper center')
ax.set_title('예측점수 분포와 등급 경계 (30% · 60% · 85%)', fontsize=12); fig.text(.5, .01, WARN, ha='center', fontsize=9, color='#b00020')
plt.tight_layout(rect=(0, .04, 1, 1)); plt.savefig(out + '/score_histogram_grades.png', dpi=130); plt.close()
