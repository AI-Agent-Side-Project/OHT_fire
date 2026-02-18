# XAI 오류 수정 보고서

## 문제 분석

### 원래 에러 메시지
```
RuntimeError: The size of tensor a (5) must match the size of tensor b (4) at non-singleton dimension 3
```

### 근본 원인
1. **DeepExplainer의 차원 불일치**: SHAP의 DeepExplainer가 배경 데이터와 테스트 데이터의 shape 불일치로 인해 텐서 차원 문제 발생
2. **TimesNet 모델의 복잡한 구조**: 시계열 데이터를 다루면서 padding 및 reshape 과정에서 차원이 동적으로 변함
3. **배경 데이터와 입력 데이터의 비일관성**: 1D/2D/3D 입력 처리 시 예상치 못한 shape 변환

## 수정 사항

### 1. **SHAP Explainer 변경**
- **변경 전**: `shap.DeepExplainer` 사용 (모델에 직접 텐서 전달)
- **변경 후**: `shap.KernelExplainer` 사용 (래핑된 예측 함수 사용)
- **장점**: 
  - 모델의 내부 구조에 영향을 받지 않음
  - 더 강력한 호환성
  - 에러 처리 메커니즘 개선

### 2. **데이터 전처리 개선**
```python
# 기존: 3D 배열 직접 전달
background_data = np.vstack(background_samples)
X_test = np.vstack(test_samples)

# 개선: 평탄화(flattening) 처리
background_data_flat = background_data.reshape(background_data.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
```
- SHAP이 1D/2D 입력을 더 안정적으로 처리
- 차원 불일치 문제 해결

### 3. **예측 함수 래핑 개선**
```python
def predict_flat(x):
    """평탄화된 입력을 받는 예측 함수"""
    # reshape back to 3D
    batch_size = x.shape[0]
    x_reshaped = x.reshape(batch_size, background_data.shape[1], background_data.shape[2])
    return model_predict_wrapper(x_reshaped)
```
- 입력 shape을 원본 모델에 맞게 자동 변환
- 입력/출력 차원 불일치 방지

### 4. **에러 처리 및 폴백 메커니즘**
- KernelExplainer 계산 실패 시 그래디언트 기반 대체 방법 제공
- 더 자세한 로깅과 디버깅 정보 출력
- 부분 계산 실패 시에도 부분 결과 저장

### 5. **계산량 최적화**
- 배경 데이터 서브샘플링: min(50, total samples)
- 테스트 샘플 서브샘플링: min(20, total samples)
- 계산 시간 단축 (특히 큰 데이터셋에서)

### 6. **결과 저장 형식 개선**
- 평탄화된 특성 중요도에서 원본 shape으로 자동 복구
- 시간-특성별 중요도를 평균화하여 특성별 중요도 산출
- JSON 파일에 더 많은 메타데이터 포함

## 실행 방법

### XAI 분석 활성화하여 학습/테스트
```bash
python run.py --use_xai --xai_num_samples 100
```

### 옵션 설명
- `--use_xai`: XAI 분석 활성화
- `--xai_num_samples`: 분석에 사용할 샘플 수 (기본값: 100)

## 결과 확인

결과는 다음 경로에 저장됩니다:
```
./xai_results/{model_setting}/
├── xai_analysis.json          # 상세 분석 결과
├── xai_summary.txt            # 텍스트 요약
├── feature_importance.png     # 특성 중요도 시각화
├── X_test.npy                 # 테스트 입력 데이터
├── y_test.npy                 # 테스트 레이블
└── shap_values_class_*.npy    # 각 클래스별 SHAP values
```

## Streamlit 대시보드

XAI 결과는 Streamlit 앱의 "🔍 XAI Analysis" 탭에서 확인 가능:
- SHAP 분석 요약
- 각 클래스별 특성 중요도
- 상위 10개 중요 특성 시각화

```bash
streamlit run streamlit_app.py
```

## 기술적 개선 사항

| 항목 | 변경 전 | 변경 후 |
|------|--------|--------|
| Explainer 종류 | DeepExplainer | KernelExplainer |
| 데이터 형식 | 3D 텐서 | 1D/2D 배열 (평탄화) |
| 에러 처리 | 없음 | Try-catch + Fallback |
| 계산 최적화 | 전체 샘플 사용 | 서브샘플링 |
| 호환성 | 모델 의존적 | 모델 독립적 |

## 참고사항

1. **계산 시간**: KernelExplainer는 DeepExplainer보다 느릴 수 있습니다. 필요시 `xai_num_samples` 값을 줄이세요.
2. **메모리 사용**: 큰 배치 크기에서는 메모리 부족이 발생할 수 있으므로 적절히 조정하세요.
3. **안정성**: 그래디언트 기반 폴백은 근사치이므로 KernelExplainer 결과보다 정확도가 낮을 수 있습니다.

## 테스트 체크리스트

- [ ] XAI 분석 실행 완료 (오류 없음)
- [ ] xai_results 폴더에 모든 파일 생성 확인
- [ ] Streamlit 앱에서 XAI 탭 로드 성공
- [ ] 특성 중요도 시각화 표시 확인
- [ ] JSON 분석 결과 저장 확인
