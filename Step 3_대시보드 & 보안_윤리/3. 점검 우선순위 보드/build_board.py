"""점검 우선순위 보드 빌드. 저장된 예측 결과·경보 등급에서 값을 읽어 HTML에 내장 (실제 고장 라벨은 내장하지 않음).
사용: python -I build_board.py <PdM_Lab 루트> <템플릿> <출력 HTML>"""
import sys, json, numpy as np, pandas as pd
root, tpl, outp = sys.argv[1:4]
S2 = root + '/Step 2_고장 분류 모델 & ROI 시나리오'; S3 = root + '/Step 3_대시보드 & 보안_윤리'
raw = pd.read_csv(S2 + '/1. 모델링 전처리/test_original_raw.csv')
gr = pd.read_csv(S3 + '/2. 교육용 경보 기준/test_alert_grades.csv', encoding='utf-8-sig')
pol = pd.read_csv(S3 + '/2. 교육용 경보 기준/alert_policy.csv', encoding='utf-8-sig')
pred = pd.read_csv(S2 + '/3. XGBoost 학습 및 모델 비교/test_predictions_both.csv')
assert len(raw) == len(gr) == len(pred) == 2000
assert (raw.UDI.values == gr.sample_id_UDI.values).all()                          # 행 정렬 확인
assert np.allclose(gr.rf_score.values, pred.rf_proba.values)                      # 저장된 예측점수와 동일
# 등급은 저장된 정의로 다시 계산해 저장된 등급 파일과 대조 (경계값은 상위 등급)
s = np.round(pred.rf_proba.values, 9)
g2 = np.select([s >= .85, s >= .60, s >= .30], ['위험', '경고', '주의'], '정상')
assert (g2 == gr.grade.values).all()

names = ['정상', '주의', '경고', '위험']
ranges = {'정상': '점수 < 0.30', '주의': '0.30 ≤ 점수 < 0.60', '경고': '0.60 ≤ 점수 < 0.85', '위험': '점수 ≥ 0.85'}
grades = []
for n in names:
    r = pol[pol['등급'] == n].iloc[0]
    grades.append(dict(name=n, range=ranges[n], check=r['담당자 확인 내용'], action=r['점검 조치'], sla1=r['확인 SLA(예시)'], sla2=r['조치 SLA(예시)'], esc=r['에스컬레이션(예시)']))
rows = []
for i in range(len(raw)):
    q = raw.iloc[i]
    rows.append([int(q.UDI), q.Type, round(float(q.air_temp), 1), round(float(q.process_temp), 1), int(q.rpm), round(float(q.torque), 1), int(q.tool_wear),
                 round(float(q.temp_diff), 1), round(float(q.power), 1), round(float(q.tool_torque_interaction), 1), round(float(gr.rf_score[i]), 6), names.index(g2[i])])
data = dict(grades=grades, rows=rows)    # 실제 고장 라벨(y)은 포함하지 않음
html = open(tpl, encoding='utf-8').read().replace('/*DATA*/', json.dumps(data, ensure_ascii=False, separators=(',', ':')))
assert 'actual_failure' not in html
open(outp, 'w', encoding='utf-8').write(html)
print('저장', outp, round(len(html) / 1024), 'KB', pd.Series(g2).value_counts().reindex(names).to_dict())
