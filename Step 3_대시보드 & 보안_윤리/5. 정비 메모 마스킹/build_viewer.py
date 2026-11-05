"""privacy_viewer.html 빌드: classroom_only/viewer_data.json을 템플릿에 내장.
사용: python -I build_viewer.py <폴더(5. 정비 메모 마스킹)>"""
import sys
d = sys.argv[1]
t = open(d + '/viewer_template.html', encoding='utf-8').read()
data = open(d + '/classroom_only/viewer_data.json', encoding='utf-8').read()
open(d + '/privacy_viewer.html', 'w', encoding='utf-8').write(t.replace('/*DATA*/', data))
print('저장', d + '/privacy_viewer.html')
