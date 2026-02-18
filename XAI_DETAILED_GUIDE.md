# XAI 에러 수정 상세 가이드

## 🐛 발생했던 에러

```
RuntimeError: The size of tensor a (5) must match the size of tensor b (4) at non-singleton dimension 3
File "/home/sung145/miniconda3/lib/python3.13/site-packages/shap/explainers/_deep/deep_pytorch.py", line 372, in nonlinear_1d
```

## 🔍 에러의 원인 분석

### 1. **DeepExplainer의 한계**
- SHAP의 `DeepExplainer`는 모델의 내부 구조를 추적해야 함
- TimesNet 모델의 복잡한 FFT 연산과 동적 padding으로 인해 형태 변환이 일어남
- 배경 데이터와 테스트 데이터의 차원이 호환되지 않음

### 2. **입력 데이터 차원 불일치**
```
배경 데이터: (N_bg, seq_len, n_features) = (100, 16, 7)
테스트 데이터: (N_test, seq_len, n_features) = (100, 16, 7)

↓ FFT와 padding 과정에서

실제 처리: (batch_size, 5) vs (batch_size, 4) ← 차원 불일치!
```

### 3. **모델의 동적 형태 변환**
```python
# TimesNet의 forward pass
def forward(self, x):
    # x shape: (batch, seq_len, features)
    enc_out = self.enc_embedding(x, None)
    
    for i in range(self.layer):
        enc_out = self.layer_norm(self.model[i](enc_out))
        # TimesBlock 내부에서 FFT + padding + reshape
        # → 차원이 예측 불가능하게 변함
    
    output = output.reshape(output.shape[0], -1)
    output = self.projection(output)  # (batch, num_classes)
    return output
```

## ✅ 적용된 해결책

### 1️⃣ **Explainer 변경: DeepExplainer → KernelExplainer**

#### 변경 전 (문제 발생)
```python
explainer = shap.DeepExplainer(self.model, background_tensor)
shap_values = explainer.shap_values(X_test_tensor)  # ❌ 에러!
```

#### 변경 후 (안정적)
```python
# KernelExplainer: 모델 독립적인 블랙박스 방식
explainer = shap.KernelExplainer(
    predict_flat,  # 래핑된 예측 함수
    background_subsample,  # 배경 데이터
    link="logit"
)
shap_values = explainer.shap_values(test_subsample)  # ✓ 작동!
```

### 2️⃣ **데이터 평탄화 (Flattening)**

#### 변경 전
```python
# 3D 배열 직접 전달
background_data.shape = (100, 16, 7)  # ❌ 차원 불일치 문제
X_test.shape = (100, 16, 7)
```

#### 변경 후
```python
# 1D로 평탄화 → 예측 함수에서 복원
background_data_flat = background_data.reshape(background_data.shape[0], -1)
# (100, 112) = (100, 16*7)

X_test_flat = X_test.reshape(X_test.shape[0], -1)
# (100, 112) = (100, 16*7)

# SHAP은 평탄화된 데이터로 안정적으로 작동
```

### 3️⃣ **래퍼 함수로 자동 차원 변환**

```python
def predict_flat(x):
    """
    입력: (batch_size, flattened_features)
    출력: (batch_size, num_classes)
    """
    # 자동으로 3D로 복원
    batch_size = x.shape[0]
    x_reshaped = x.reshape(batch_size, 16, 7)  # 원래 shape으로 복원
    
    # 모델에 전달
    return model_predict_wrapper(x_reshaped)
```

### 4️⃣ **에러 처리 및 폴백**

```python
try:
    # KernelExplainer 시도
    explainer = shap.KernelExplainer(predict_flat, background_subsample)
    shap_values = explainer.shap_values(test_subsample)
    print("✓ SHAP values computed successfully")
    
except Exception as e:
    # 실패 시 그래디언트 기반 방법으로 폴백
    print(f"Warning: SHAP computation failed: {e}")
    print("Using alternative approach: Computing feature importance from gradients...")
    
    # 그래디언트 기반 특성 중요도 계산
    # (더 빠르지만 덜 정확함)
```

## 📊 성능 비교

| 항목 | 변경 전 | 변경 후 |
|------|--------|--------|
| **Explainer** | DeepExplainer | KernelExplainer |
| **안정성** | ❌ 에러 발생 | ✅ 안정적 |
| **호환성** | 모델 의존적 | 모델 독립적 |
| **계산 속도** | 빠름 | 중간 |
| **정확도** | 높음 | 충분함 |
| **폴백 지원** | ❌ 없음 | ✅ 있음 |

## 🚀 사용 방법

### 1. XAI 분석을 포함하여 학습 및 테스트

```bash
# 기본 설정
python run.py --use_xai

# 샘플 수 지정
python run.py --use_xai --xai_num_samples 100

# 전체 옵션 예시
python run.py \
  --model TimesNet \
  --data OHT_fire \
  --is_training 1 \
  --train_epochs 10 \
  --use_xai \
  --xai_num_samples 100
```

### 2. 결과 확인

결과는 `./xai_results/{setting}/` 폴더에 저장됩니다:

```
xai_results/
└── TSC_dm64_dff128_topk3_sl16_nminmax/
    ├── xai_analysis.json           # ✓ 상세 분석 데이터
    ├── xai_summary.txt             # ✓ 텍스트 요약 보고서
    ├── feature_importance.png      # ✓ 시각화 (클래스별)
    ├── X_test.npy                  # 테스트 입력
    ├── y_test.npy                  # 테스트 레이블
    ├── shap_values_class_0.npy     # SHAP values (Class 0)
    ├── shap_values_class_1.npy     # SHAP values (Class 1)
    ├── shap_values_class_2.npy     # SHAP values (Class 2)
    └── shap_values_class_3.npy     # SHAP values (Class 3)
```

### 3. Streamlit 대시보드에서 확인

```bash
streamlit run streamlit_app.py
```

대시보드의 **"🔍 XAI Analysis"** 탭에서:
- 모델 정확도
- 각 클래스별 특성 중요도
- 상위 10개 중요 특성 (시각화 + 수치)
- SHAP 통계

## 📈 출력 예시

### xai_summary.txt
```
SHAP XAI Analysis Summary
========================
Model: TSC_dm64_dff128_topk3_sl16_nminmax
Total Samples Analyzed: 100
Number of Classes: 4
Model Accuracy on Test Set: 0.8500

Input Shape: (100, 16, 7)
Flattened Feature Dimension: (100, 112)

Feature Importance (Mean |SHAP value|):

  Class 0:
    Feature 42: 0.234567
    Feature 53: 0.198765
    Feature 28: 0.176543
    ...
```

### xai_analysis.json
```json
{
  "total_samples": 100,
  "num_classes": 4,
  "model_accuracy": 0.85,
  "feature_importance": {
    "class_0": [0.234, 0.198, 0.176, ...],
    "class_1": [0.156, 0.143, 0.132, ...],
    ...
  },
  "predictions": [0, 1, 2, ...],
  "true_labels": [0, 1, 2, ...],
  "prediction_probabilities": [[0.9, 0.05, ...], ...]
}
```

## 🔧 고급 옵션

### 계산량 조절

큰 데이터셋의 경우 샘플 수를 줄이세요:

```bash
# 빠른 분석 (작은 샘플)
python run.py --use_xai --xai_num_samples 20

# 정밀한 분석 (큰 샘플)
python run.py --use_xai --xai_num_samples 200
```

### 메모리 최적화

`exp_classification.py`에서 다음 값을 조정:

```python
# 약 50줄 근처
background_subsample = shap.sample(background_data_flat, min(50, ...))  # ← 이 값 조정
test_subsample = X_test_flat[:min(20, X_test_flat.shape[0])]  # ← 이 값 조정
```

## ✨ 주요 개선 사항 요약

| # | 개선 사항 | 효과 |
|---|---------|------|
| 1 | KernelExplainer 사용 | ✓ 에러 해결 |
| 2 | 데이터 평탄화 | ✓ 차원 호환성 |
| 3 | 예측 래퍼 함수 | ✓ 자동 차원 변환 |
| 4 | 에러 처리 | ✓ 안정성 향상 |
| 5 | 계산 최적화 | ✓ 성능 향상 |
| 6 | 결과 시각화 | ✓ 해석성 향상 |

## 📝 문제 해결

### Q: XAI 분석이 너무 느립니다
**A:** `--xai_num_samples` 값을 줄여보세요
```bash
python run.py --use_xai --xai_num_samples 50
```

### Q: 메모리 부족 에러가 발생합니다
**A:** `exp_classification.py`에서 서브샘플링 값 조정:
```python
# 줄 약 350 근처
background_subsample = shap.sample(background_data_flat, min(20, ...))
test_subsample = X_test_flat[:min(10, X_test_flat.shape[0])]
```

### Q: Streamlit에서 XAI 결과가 보이지 않습니다
**A:** 다음을 확인하세요:
1. XAI 분석이 완료되었는가 (`--use_xai` 플래그 사용)
2. `./xai_results/{model_setting}/` 폴더가 존재하는가
3. JSON 파일이 생성되었는가

## 🎯 다음 단계

1. ✅ 수정된 코드로 XAI 분석 실행
2. ✅ 결과 확인 및 검증
3. ✅ Streamlit 대시보드에서 시각화 확인
4. ⏭️ 프로덕션 배포

---

**마지막 업데이트**: 2026-02-18  
**버전**: 1.0  
**상태**: ✅ 완료 및 검증됨
