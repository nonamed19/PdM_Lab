# PdM Lab

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC6B23)

PdM Lab은 AI4I 2020 합성 설비 데이터를 이용해 예지보전 분석 과정을 실습하는 프로젝트입니다. 센서 데이터를 탐색하고 이상 탐지 및 고장 분류 모델을 비교한 뒤, 임계값과 비용 시나리오를 검토합니다. 분석 결과는 대시보드, 점검 보드, 보고서와 개인정보 보호 실습 자료로 확인할 수 있습니다.

이 프로젝트의 분류 결과는 **현재 관측값의 정상·고장 분류**입니다. 미래 고장을 사전에 예측한 성과로 해석할 수 없으며, 합성 데이터의 규칙과 경보 기준은 실제 설비에 그대로 적용할 수 없습니다.

## 주요 기능

- **데이터 탐색**: 센서 분포, 제품 유형별 고장률, 파생변수와 고장의 관계를 확인합니다.
- **이상 탐지와 고장 분류**: Isolation Forest 분석 결과를 확인하고, Random Forest와 XGBoost를 학습·비교합니다. 전처리는 학습 데이터에만 SMOTENC를 적용합니다.
- **임계값과 비용 분석**: 분류 임계값별 정밀도와 재현율을 비교하고, 가정에 따른 도입 비용과 절감액을 계산합니다.
- **결과 설명과 시각화**: SHAP 분석, 센서 탐색 화면, 경영진 대시보드와 점검 우선순위 보드를 제공합니다.
- **정보 보호 실습**: 합성 데이터의 가명처리와 k-익명성, 가상 정비 메모의 민감정보 마스킹을 시연합니다.
- **보고서**: 분석 결과를 담은 1페이지 보고서와 발표용 HTML 및 PDF를 제공합니다.

임계값 탐색은 테스트 데이터의 라벨을 사용하므로 해당 결과는 독립 평가가 아닙니다. 비용과 ROI는 시나리오의 가정에 따른 값이며 실제 절감 실적이 아닙니다. 데이터 생성 규칙에 기반한 위험 플래그는 정답과 직접 연결될 수 있으므로 모델 성능을 해석할 때 누수에 유의해야 합니다.

## 기술 구성

| 구분 | 사용 기술 및 파일 |
| --- | --- |
| 데이터 분석 | Python, pandas, NumPy, Matplotlib |
| 모델링 | scikit-learn, imbalanced-learn, XGBoost, SHAP |
| 데이터 및 결과 | CSV, JSON, joblib, PNG |
| 화면 및 보고서 | HTML, JavaScript, PDF |
| PDF 생성 | Playwright, Chrome, PyMuPDF |

## 시작하기

### 저장소 준비

```bash
git clone https://github.com/nonamed19/PdM_Lab.git
cd PdM_Lab
```

저장소에 원본 데이터와 주요 분석 결과가 포함되어 있습니다. Python 설치 없이도 아래 HTML 파일을 브라우저에서 열어 저장된 결과를 확인할 수 있습니다.

- [센서 탐색 화면](<Step 1_데이터 탐색 & 이상 탐지/5. 센서 탐색 화면/sensor_explorer.html>)
- [고장 예측 설명 화면](<Step 2_고장 분류 모델 & ROI 시나리오/6. SHAP 설명/prediction_explainer.html>)
- [경영진 대시보드](<Step 3_대시보드 & 보안_윤리/1. 경영진 대시보드/dashboard.html>)
- [점검 우선순위 보드](<Step 3_대시보드 & 보안_윤리/3. 점검 우선순위 보드/inspection_board.html>)
- [정비 메모 마스킹 화면](<Step 3_대시보드 & 보안_윤리/5. 정비 메모 마스킹/privacy_viewer.html>)
- [경영진 1페이지 보고서](<Step 3_대시보드 & 보안_윤리/6. 경영진 1페이지 보고서/executive_report.html>)
- [경영진 발표 보고서](<Step 3_대시보드 & 보안_윤리/7. 경영진 발표 보고서/executive_presentation.html>)

### 분석 스크립트 실행 환경

모델 학습을 다시 실행하려면 Python과 다음 라이브러리가 필요합니다. 저장소에 의존성 버전 고정 파일이 없으므로 아래 명령은 코드에서 확인한 패키지를 설치합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas matplotlib scikit-learn imbalanced-learn xgboost joblib shap
```

Windows PowerShell에서는 가상환경을 `.venv\Scripts\Activate.ps1`로 활성화합니다. 그래프 스크립트는 `Malgun Gothic` 글꼴을 지정하므로 다른 환경에서는 한글 글꼴 설정이 필요할 수 있습니다.

### 전처리와 모델 학습 예시

다음은 저장소 루트에서 실행하는 Bash 예시입니다. 저장된 파생변수 CSV를 입력으로 사용하고, 기존 결과를 덮어쓰지 않도록 `work/`에 새 결과를 생성합니다.

```bash
mkdir -p work/preprocess work/rf work/xgb work/threshold work/cost
S1="Step 1_데이터 탐색 & 이상 탐지"
S2="Step 2_고장 분류 모델 & ROI 시나리오"

python -I "$S2/1. 모델링 전처리/preprocess.py" \
  "$S1/3. 파생변수 생성/ai4i2020_features.csv" work/preprocess
python -I "$S2/2. 랜덤포레스트 학습/train_rf.py" work/preprocess work/rf
python -I "$S2/3. XGBoost 학습 및 모델 비교/train_xgb_compare.py" \
  work/preprocess work/rf work/xgb
python -I "$S2/4. 임계값 탐색/threshold_explore.py" \
  work/xgb/test_predictions_both.csv work/threshold
python -I "$S2/5. 도입 비용 시나리오/cost_scenario.py" \
  work/threshold/selected_thresholds_metrics.csv work/cost
```

실행 후 `work/rf/`의 모델과 평가 결과, `work/xgb/`의 비교 결과, `work/threshold/`의 임계값별 지표, `work/cost/`의 비용 시나리오를 확인합니다. 초기 탐색 및 파생변수 단계는 저장된 결과와 각 단계의 설명을 참고합니다.

보고서 PDF를 다시 생성하려면 추가로 `playwright`와 `pymupdf` 패키지 및 Chrome이 필요합니다. 각 스크립트의 인자와 상세 실행 방법은 해당 단계의 README를 참고하세요.

## 사용 흐름

1. [데이터 다운로드 및 구성](<Step 0_작업 폴더 확인 & 데이터 다운로드/1. 데이터 다운로드/README.md>)을 확인하고 센서 분포와 파생변수를 살펴봅니다.
2. [Isolation Forest 결과](<Step 1_데이터 탐색 & 이상 탐지/4. Isolation Forest 이상탐지/README.md>)를 검토하고, 저장된 파생변수 데이터로 분류 모델을 학습합니다.
3. [모델 비교](<Step 2_고장 분류 모델 & ROI 시나리오/3. XGBoost 학습 및 모델 비교/README.md>)와 [임계값 탐색](<Step 2_고장 분류 모델 & ROI 시나리오/4. 임계값 탐색/README.md>) 결과를 확인합니다.
4. [비용 시나리오](<Step 2_고장 분류 모델 & ROI 시나리오/5. 도입 비용 시나리오/README.md>)의 가정을 검토하고 대시보드와 보고서를 엽니다.
5. [가명처리 실습](<Step 3_대시보드 & 보안_윤리/4. 가명처리와 k-익명성/README.md>)과 [정비 메모 마스킹 실습](<Step 3_대시보드 & 보안_윤리/5. 정비 메모 마스킹/README.md>)을 확인합니다. 가명처리와 k-익명성 시연은 완전한 익명화나 외부 공유 승인을 의미하지 않습니다.

## 프로젝트 구조

| 경로 | 역할 |
| --- | --- |
| `Step 0_작업 폴더 확인 & 데이터 다운로드/` | AI4I 원본 데이터와 데이터셋 설명 |
| `Step 1_데이터 탐색 & 이상 탐지/` | EDA, 파생변수, 이상 탐지 결과와 센서 탐색 화면 |
| `Step 2_고장 분류 모델 & ROI 시나리오/` | 전처리, 모델 학습·비교, 임계값, 비용 및 SHAP 분석 |
| `Step 3_대시보드 & 보안_윤리/` | 대시보드, 경보·점검 보드, 정보 보호 실습과 경영진 보고서 |

각 단계의 하위 폴더에 분석 설명, 관련 스크립트와 결과 파일이 함께 저장되어 있습니다.
