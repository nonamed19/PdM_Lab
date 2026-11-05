"""샘플 PDF 생성: 브라우저 인쇄(PDF로 저장)와 같은 경로(인쇄 상태 진입 → 인쇄용 CSS)로 출력한다.
사용: python -I make_sample_pdf.py <발표 폴더>   (기본 선택 조건: 랜덤포레스트 · 탐색적 임계값 · 고장 비용 5,000만 원)"""
import sys, pathlib
from playwright.sync_api import sync_playwright
d = sys.argv[1]
with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome', headless=True); pg = b.new_page(viewport={'width': 1920, 'height': 1080})
    pg.goto(pathlib.Path(d + '/executive_presentation.html').resolve().as_uri()); pg.wait_for_timeout(2500)
    pg.evaluate("window.__deck.enterPrint()")
    pg.pdf(path=d + '/executive_presentation_sample.pdf', prefer_css_page_size=True, print_background=True)
    pg.evaluate("window.__deck.exitPrint()"); b.close()
print('저장 완료')
