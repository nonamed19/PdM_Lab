# 모델링을 위한 전처리 (Train/Test 분할 · 스케일링 · SMOTENC · 원핫)

**질문:** 목표를 Machine failure로 두고, 입력은 센서 5개·파생변수 3개(temp_diff, power, tool_torque_interaction)·Type만 사용해 전처리해줘. 순서는 (1) 입력 선택 → (2) 80/20 분할(stratify, random_state=42) → (3) 연속형 8개만 Train에서 StandardScaler 학습 후 변환 → (4) Train에만 SMOTENC(random_state=42)로 1:1 → (5) Type 원핫. 각 단계의 shape과 고장 비율, 원핫 검증, RF·XGBoost 공용 처리본·전처리기 저장, 원본 Test 유지 확인까지 해줘.

**답변:** 요청한 순서대로 처리했고, 모든 검증을 통과했습니다. Test는 합성 없이 원본 2,000건 그대로입니다.

- 전처리 코드: [preprocess.py](preprocess.py) (`python -I preprocess.py <features.csv> <출력폴더>`로 재현)

> ⚠️ 입력 변수 중 power, tool_torque_interaction, temp_diff는 합성 데이터 생성 규칙에서 나온 변수입니다. 이 데이터에서는 성능이 비현실적으로 높게 나올 수 있으며, 실제 설비 성능이 아닙니다. 제외 대상(UDI, Product ID, TWF~RNF, hdf_risk·pwf_risk·osf_risk)은 입력에 넣지 않았습니다.

### 1) 단계별 shape과 고장 비율

| 단계 | 데이터 | shape | 고장 | 고장 비율 |
|---|---|---|---|---|
| 1. 입력 선택 | 전체 | (10,000, 9) | 339 | 3.39% |
| 2. 분할 | Train | (8,000, 9) | 271 | 3.39% |
| 2. 분할 | Test | (2,000, 9) | 68 | 3.40% |
| 3. 스케일링 | Train | (8,000, 9) | 271 | 3.39% (연속형 평균 ≈ 0, 표준편차 = 1) |
| 4. SMOTENC | Train | (15,458, 9) | 7,729 | **50.00%** (정상 7,729 : 고장 7,729) |
| 5. 원핫 최종 | Train | (15,458, 11) | 7,729 | 50.00% |
| 5. 원핫 최종 | Test | (2,000, 11) | 68 | 3.40% (원본 분포 유지) |

- 합성 샘플은 7,458건입니다. 원래 Train 고장 271건에 합성 7,458건을 더해 정상 수와 맞췄습니다.
- 입력 9열은 연속형 8개 + Type입니다. 원핫 후에는 연속형 8개 + Type_H, Type_L, Type_M으로 11열입니다.
- 열 이름에서 `[K]`, `[rpm]` 같은 대괄호를 제거했습니다(`air_temp`, `process_temp`, `rpm`, `torque`, `tool_wear` 등). XGBoost는 열 이름에 `[`, `]`, `<`가 있으면 오류가 나기 때문입니다.

### 2) Type 원핫 검증

| 검증 | Train | Test |
|---|---|---|
| Type_H/L/M 모두 0 또는 1 | ✅ | ✅ |
| 모든 행의 합 = 1 | ✅ | ✅ |

### 3) 원본 Test 유지 검증

| 검증 항목 | 결과 |
|---|---|
| Test 행수 2,000건 그대로 (합성 없음) | ✅ |
| Test 고장 비율이 원본과 같음 (3.40% vs 3.39%) | ✅ |
| StandardScaler가 Train 통계만 사용 (전체 평균과 다름) | ✅ |
| 스케일된 Test를 역변환하면 원본 값과 일치 (최대 오차 1.8e-12) | ✅ |
| 독립적으로 다시 분할한 Test와 UDI·고장 라벨이 일치 | ✅ |
| Train과 Test 사이에 겹치는 행 없음 | ✅ |
| SMOTENC 후에도 Train 원본 행이 앞쪽에 그대로 있음 | ✅ |
| NaN 없음, 모든 열 float64, 저장 후 다시 읽어도 열 순서 동일 | ✅ |

### 4) 저장 파일

| 파일 | 내용 |
|---|---|
| `X_train.csv`, `y_train.csv` | SMOTENC·원핫까지 끝난 Train (15,458행) |
| `X_test.csv`, `y_test.csv` | 변환만 적용한 원본 Test (2,000행) |
| `test_original_raw.csv` | 스케일링 전 원본 Test 값 + UDI + 라벨 (대조용) |
| `preprocessor.joblib` | StandardScaler, OneHotEncoder(H/L/M 고정), 열 이름 목록 |
| `preprocess_log.json` | 단계별 shape·비율·검증 결과 기록 |

RF와 XGBoost 모두 `X_train.csv`/`X_test.csv`를 그대로 읽어 쓰면 됩니다. 새 데이터를 예측할 때는 `preprocessor.joblib`의 scaler와 onehot을 `transform`만 적용합니다.

### 5) 주의할 점: SMOTENC가 Type 분포를 바꿨습니다

| Train 클래스 | Type_H | Type_L | Type_M |
|---|---|---|---|
| 정상 | 10.0% | 59.9% | 30.1% |
| 고장 (SMOTENC 후) | **0.2%** | **84.1%** | 15.7% |

- 원본 고장 데이터의 Type 비율은 L 69%, M 24%, H 6%입니다. SMOTENC는 범주형 열을 이웃 중 가장 흔한 값으로 채우기 때문에, 합성된 고장 샘플이 대부분 L이 되었습니다.
- 그 결과 모델이 "Type=L이면 고장"이라는 가짜 규칙을 학습할 수 있습니다. 원본 Test에서는 이 규칙이 성립하지 않으므로 성능이 떨어질 수 있습니다.
- 이후 모델링에서 Type의 중요도가 비정상적으로 높게 나오면 이 영향을 의심해야 합니다. 필요하면 Type을 제외한 모델, 또는 `class_weight`/`scale_pos_weight` 방식과 비교해 볼 수 있습니다.
- 요청하신 순서(스케일링 → SMOTENC → 원핫)는 그대로 지켰고 순서는 바꾸지 않았습니다.
- XGBoost는 아직 설치하지 않았습니다. 이번 단계는 전처리만 포함하며, 처리본은 XGBoost에서 바로 읽을 수 있는 형식입니다.
