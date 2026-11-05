# 데이터셋 다운로드

**질문:** UCI Machine Learning Repository에서 AI4I 2020 Predictive Maintenance 데이터셋(https://archive.ics.uci.edu/dataset/601)을 다운로드해서 이 폴더에 저장하고, 파일 목록과 데이터 크기를 확인해줘.

**답변:** `D:\PdM_Lab\1. 데이터 다운로드\`에 다운로드하고 압축을 풀었습니다.

| 파일 | 크기 |
|---|---|
| `ai4i2020.csv` | 522,048 bytes (약 510 KB) |
| `dataset.zip` | 522,170 bytes (약 510 KB) |

- 헤더를 뺀 10,000행, 14개 컬럼입니다.
- 컬럼: UDI, Product ID, Type, 공기 온도, 공정 온도, 회전 속도, 토크, 공구 마모, Machine failure, TWF, HDF, PWF, OSF, RNF
- CSV 맨 앞에 BOM이 있습니다. pandas로 읽을 때는 `encoding='utf-8-sig'`를 사용합니다.
