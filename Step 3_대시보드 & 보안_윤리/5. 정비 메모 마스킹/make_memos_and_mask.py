"""가상 정비 메모 20건 생성 -> 민감정보 탐지/마스킹 -> 정답 대조 -> 저장.  (수업용 가상정보)
사용: python -I make_memos_and_mask.py <출력 폴더> <비공개 대응표 폴더>
  - 담당자 이름/이메일(example.com)/전화번호/내부 설비번호(EQ-TEST-###)는 모두 가상 정보
  - 정답 목록은 메모 작성 시 표시한 마커에서 직접 추출(탐지 코드와 독립)
  - 담당자·연락처는 토큰형 마스킹, 설비번호는 일관된 무작위 대체번호로 교체(대응표는 비공개 폴더에만 저장)
  - 고장 증상·정비 내용 문구는 유지되어야 하며, 유지 여부를 검사"""
import sys, re, os, csv, json, secrets, string
out, priv = sys.argv[1:3]
os.makedirs(out + '/classroom_only', exist_ok=True)

# ---- 1) 가상 메모 (마커: {N:이름} {E:이메일} {P:전화} {Q:설비번호} {S:고장 증상} {A:정비 내용}) ----
RAW = [
 ('2025-03-03', "[{Q:EQ-TEST-001}] {S:회전 속도 급감}과 {S:토크 상승} 확인. {A:공구 교체}와 {A:벨트 장력 재조정} 완료. 담당자 {N:김가상} 책임, 연락처 {P:010-0000-0101}."),
 ('2025-03-04', "{Q:EQ-TEST-002}에서 {S:공구 마모 과다} 경고. {A:공구 교체} 후 시험 가공 정상. 작성자: {N:이테스트} 대리 ({E:lee.test@example.com})."),
 ('2025-03-05', "{Q:EQ-TEST-003} {S:방열 불량}으로 공정 온도 310.5 K까지 상승. {A:냉각수 보충}, {A:필터 청소}. 문의는 {N:박예시} 주임 {P:010 0000 0103}."),
 ('2025-03-06', "{Q:EQ-TEST-001} 재점검: {S:소음 증가} 확인, {A:베어링 윤활} 실시. 확인자 {N:김가상} 책임 / {E:gasang.kim@example.com}."),
 ('2025-03-07', "{Q:EQ-TEST-004}, {Q:EQ-TEST-005} 동시 {S:전력 변동} 발생. {A:전원부 점검}과 {A:센서 재교정}. 연락 {P:01000000104} ({N:최샘플} 팀장)."),
 ('2025-03-10', "{Q:EQ-TEST-006} 가동 중 {S:진동 증가}, 회전 속도 1,450 rpm에서 토크 58.8 Nm. {A:커플링 정렬} 후 개선. 담당 {N:정연습} 대리, {P:031-000-0105}."),
 ('2025-03-11', "{Q:EQ-TEST-002} {S:공구 마모 과다} 재발. 공구 마모 202 min. {A:공구 교체} 및 {A:마모 한계 재설정}. 작성 {N:이테스트} 대리 {E:test.lee02@example.com}."),
 ('2025-03-12', "{Q:EQ-TEST-007} {S:토크 상승}과 {S:온도 상승} 동시 발생. {A:윤활유 교환}. 로트 LOT-2025-0312 가공 중단 없음. 담당자 {N:한모의} 주임 연락처 {P:010-0000-0106}."),
 ('2025-03-13', "{Q:EQ-TEST-008} {S:냉각 불량} 의심. {A:냉각 팬 청소}와 {A:냉각수 보충}. 확인 {N:오가짜} 책임 ({E:ogajja.sample@example.com})."),
 ('2025-03-14', "{Q:EQ-TEST-003} {S:방열 불량} 재발, {A:열교환기 청소}. 이전 담당 {N:박예시} 주임, 이번 담당 {N:서더미} 대리 {P:010 0000 0107}."),
 ('2025-03-17', "{Q:EQ-TEST-009} 이송축 {S:소음 증가}, {A:그리스 주입} 실시. 점검자 {N:윤하나} 대리 {E:yoon.hana@example.com}."),
 ('2025-03-18', "{Q:EQ-TEST-010} {S:회전 속도 불안정}. {A:인버터 파라미터 점검}, {A:엔코더 케이블 교체}. 담당 {N:김가상} 책임 {P:010-0000-0108}."),
 ('2025-03-19', "{Q:EQ-TEST-004} {S:전력 변동} 재발. 전압 측정 후 {A:전원 필터 교체}. {N:최샘플} 팀장 대신 {N:임시험} 주임이 대응 ({P:01000000109})."),
 ('2025-03-20', "{Q:EQ-TEST-011} {S:공구 마모 과다}와 {S:표면 거칠기 악화}. {A:공구 교체}, {A:절삭 조건 재조정}. 작성자 {N:정연습} 대리 {E:practice.jung@example.com}."),
 ('2025-03-21', "{Q:EQ-TEST-005} {S:과부하 경보}. {A:부하 센서 재교정}과 {A:클램프 압력 조정}. 확인 {N:한모의} 주임, 전화 {P:031-000-0110}."),
 ('2025-03-24', "{Q:EQ-TEST-012} 가동 시작 직후 {S:토크 상승}. {A:구동 벨트 교체}. 담당자 {N:이테스트} 대리 {E:lee.test@example.com} / {P:010 0000 0111}."),
 ('2025-03-25', "{Q:EQ-TEST-006} {S:진동 증가} 재확인, {A:베어링 교체}. 연락처 {P:010-0000-0112}, 담당 {N:정연습} 대리."),
 ('2025-03-26', "{Q:EQ-TEST-007} {S:온도 상승} 재발. {A:냉각 라인 점검}과 {A:온도 센서 교체}. 작성 {N:서더미} 대리 ({E:seo.dummy@example.com})."),
 ('2025-03-27', "{Q:EQ-TEST-001}, {Q:EQ-TEST-008} {S:소음 증가} 점검. {A:베어링 윤활}과 {A:커버 체결 확인}. 담당자 {N:오가짜} 책임 {P:01000000113}."),
 ('2025-03-28', "{Q:EQ-TEST-010} {S:회전 속도 불안정} 해소 확인. 모니터링 지속. 점검 {N:윤하나} 대리 {E:yoon.hana@example.com}, {P:031-000-0114}."),
]
TYPES = {'N': '담당자 이름', 'E': '이메일', 'P': '전화번호', 'Q': '내부 설비번호'}
memos = []
for i, (date, tpl) in enumerate(RAW, 1):
    text = ''; truth = []; keep = []; pos = 0
    for m in re.finditer(r'\{([NEPQSA]):([^}]+)\}', tpl):
        text += tpl[pos:m.start()]; s = len(text); text += m.group(2); e = len(text); pos = m.end()
        if m.group(1) in TYPES: truth.append(dict(type=m.group(1), value=m.group(2), start=s, end=e))
        else: keep.append(m.group(2))
    text += tpl[pos:]
    memos.append(dict(id=f'M{i:02d}', date=date, text=text, truth=truth, keep=keep))

# ---- 2) 탐지 (정답과 독립적인 규칙) ----
ROSTER = ['김가상', '이테스트', '박예시', '최샘플', '정연습', '한모의', '오가짜', '서더미']     # 가상 담당자 명단
SURN = '김이박최정한오서윤임강조장홍'
RULES = {
 'E': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
 'P': re.compile(r'(?<!\d)(?<!\d,)0\d{1,2}[-\s.]?\d{3,4}[-\s.]?\d{4}(?!\d)(?!,\d)'),   # 천 단위 쉼표 숫자(1,450 등)는 제외, 문장부호 쉼표는 허용
 'Q': re.compile(r'EQ-TEST-\d{3}'),
 'N1': re.compile('|'.join(ROSTER)),                                             # 명단에 있는 이름
 'N2': re.compile(r'(?<![가-힣])[' + SURN + r'][가-힣]{2}(?=\s(?:책임|대리|주임|팀장))'),   # 성씨 + 2글자 + 직함 (명단에 없는 이름용)
}
PRI = {'E': 0, 'P': 1, 'Q': 2, 'N': 3}
def detect(text):
    c = []
    for k, rx in RULES.items():
        for m in rx.finditer(text): c.append(dict(type=k[0], value=m.group(), start=m.start(), end=m.end()))
    c.sort(key=lambda s: (s['start'], PRI[s['type']])); res = []
    for s in c:                                            # 겹치면 우선순위가 높은 것 하나만
        if res and s['start'] < res[-1]['end']: continue
        res.append(s)
    return res

# ---- 3) 마스킹/대체 ----
alpha = string.ascii_uppercase + string.digits; eqmap = {}
def eq_replace(v):
    if v not in eqmap:
        while True:
            c = 'EQ-' + ''.join(secrets.choice(alpha) for _ in range(4))
            if c not in eqmap.values() and not c.startswith('EQ-TEST'): break
        eqmap[v] = c
    return eqmap[v]
MASK = {'N': '[담당자]', 'E': '[이메일]', 'P': '[전화번호]'}
def apply(text, spans):
    o = ''; pos = 0; outspans = []
    for s in spans:
        o += text[pos:s['start']]; rep = eq_replace(s['value']) if s['type'] == 'Q' else MASK[s['type']]
        outspans.append(dict(s, rep=rep, ostart=len(o), oend=len(o) + len(rep))); o += rep; pos = s['end']
    return o + text[pos:], outspans

stat = {k: dict(truth=0, detected=0, tp=0, fn=0, fp=0) for k in TYPES}; miss = []; extra = []
for m in memos:
    det = detect(m['text']); m['masked'], m['spans'] = apply(m['text'], det)
    tset = {(t['type'], t['start'], t['end']) for t in m['truth']}; dset = {(d['type'], d['start'], d['end']) for d in det}
    for t in m['truth']: stat[t['type']]['truth'] += 1
    for d in det: stat[d['type']]['detected'] += 1
    for t in tset & dset: stat[t[0]]['tp'] += 1
    for t in tset - dset: stat[t[0]]['fn'] += 1; miss.append((m['id'], t))
    for d in dset - tset: stat[d[0]]['fp'] += 1; extra.append((m['id'], d))
    m['detected_n'] = {k: sum(1 for d in det if d['type'] == k) for k in TYPES}
print({k: stat[k] for k in stat}); print('누락', miss, '오탐', extra)

# ---- 4) 검증: 원문 값 잔존, 증상/정비 문구 유지, 대체번호 일관성 ----
leak = [(m['id'], t['value']) for m in memos for t in m['truth'] if t['value'] in m['masked']]
lost = [(m['id'], k) for m in memos for k in m['keep'] if k not in m['masked']]
assert not leak, leak
assert not lost, lost
assert not re.search(r'EQ-TEST|example\.com|010|031-', ' '.join(m['masked'] for m in memos))
assert len(set(eqmap.values())) == len(eqmap) == len({t['value'] for m in memos for t in m['truth'] if t['type'] == 'Q'})
print('원문 잔존 0, 증상/정비 문구 유지', sum(len(m['keep']) for m in memos), '개 모두 확인, 설비번호', len(eqmap), '종 -> 대체번호 일관')

# ---- 5) 저장 ----
with open(out + '/masked_memos.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['memo_id', 'date', 'masked_text'] + [f'detected_{k}' for k in TYPES])
    for m in memos: w.writerow([m['id'], m['date'], m['masked']] + [m['detected_n'][k] for k in TYPES])
with open(out + '/classroom_only/memos_original_FICTIONAL.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['memo_id', 'date', 'original_text'])
    for m in memos: w.writerow([m['id'], m['date'], m['text']])
with open(out + '/classroom_only/ground_truth.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['memo_id', 'type', 'type_name', 'value', 'start', 'end'])
    for m in memos:
        for t in m['truth']: w.writerow([m['id'], t['type'], TYPES[t['type']], t['value'], t['start'], t['end']])
os.makedirs(priv, exist_ok=True)
with open(priv + '/equipment_mapping_PRIVATE.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['original_equipment_no', 'replacement_no'])
    for k, v in sorted(eqmap.items()): w.writerow([k, v])
json.dump(dict(types=TYPES, stat=stat, eq_count=len(eqmap), memos=[dict(id=m['id'], date=m['date'], text=m['text'], masked=m['masked'], spans=m['spans'], truth=m['truth'], keep=m['keep'], n=m['detected_n']) for m in memos]),
          open(out + '/classroom_only/viewer_data.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('저장 완료')
