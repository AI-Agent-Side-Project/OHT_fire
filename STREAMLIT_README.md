# 🔥 OHT Fire - AI Prediction & XAI Dashboard

Streamlit 기반의 인터랙티브 대시보드로 모델 예측 결과와 SHAP XAI 분석을 시각화합니다.

## 설치

### 1. Streamlit 및 의존성 설치
```bash
pip install -r requirements_streamlit.txt
```

### 2. 모델 학습 및 예측 실행
```bash
# XAI 분석 포함하여 실행
python run.py --is_training=0 --use_xai --xai_num_samples=100
```

## 사용 방법

### Streamlit 앱 실행
```bash
streamlit run streamlit_app.py
```

앱이 자동으로 브라우저에서 열릴 것입니다. (기본값: http://localhost:8501)

## 주요 기능

### 📊 Tab 1: Model Predictions
- **Model Metrics**: 전체 샘플 수, 클래스 수, 정확도 표시
- **Prediction Distribution**: 예측된 클래스와 실제 클래스 분포 비교
- **Confidence Distribution**: 모델 신뢰도 분포 시각화
- **Detailed Predictions**: 상세한 예측 결과 테이블 (처음 50개 샘플)

### 🔍 Tab 2: XAI Analysis
- **XAI Summary**: SHAP 분석 요약 보고서
- **SHAP Statistics**: 각 클래스별 SHAP values 통계
- 특성 중요도 계산 결과

### 📈 Tab 3: Feature Importance
- **시각화**: Feature importance 플롯
- **상위 특성**: 각 클래스별 상위 10개 중요 특성 표시
- **수치 데이터**: 모든 특성의 중요도 수치 표시

### ⚠️ Tab 4: Alarm & Insights
- **알람 규칙 설정**: 신뢰도 임계값, 고위험/중위험 클래스 선택
- **자동 알람 생성**:
  - 🔴 고위험 클래스 감지
  - 🟠 중위험 클래스 감지
  - ⚠️ 낮은 신뢰도 감지
- **알람 통계**: 생성된 알람의 개수 및 분포
- **상세 알람 리스트**: 각 알람의 원인과 위험 수준
- **권장사항**: 시스템이 생성한 자동 권장사항

## 데이터 구조

앱이 다음 디렉토리에서 데이터를 읽습니다:

```
./test_results/
├── {model_name}/
│   └── (예측 결과 파일)
│
./xai_results/
└── {model_name}/
    ├── xai_analysis.json (분석 결과)
    ├── xai_summary.txt (요약 보고서)
    ├── feature_importance.png (특성 중요도 이미지)
    ├── X_test.npy (테스트 데이터)
    ├── y_test.npy (실제 레이블)
    ├── shap_values_class_0.npy
    ├── shap_values_class_1.npy
    └── ...
```

## 알람 규칙 커스터마이징

Streamlit 사이드바에서 다음을 설정할 수 있습니다:

1. **Confidence Threshold**: 신뢰도가 이 값 이하일 때 알람 발생
2. **High-Risk Fire Classes**: 고위험으로 분류할 클래스
3. **Medium-Risk Fire Classes**: 중위험으로 분류할 클래스

## 예시 실행 순서

```bash
# 1. 모델 훈련 및 예측 (XAI 분석 포함)
python run.py --is_training=1 --use_xai --xai_num_samples=100

# 2. Streamlit 대시보드 실행
streamlit run streamlit_app.py

# 3. 브라우저에서 http://localhost:8501 접속
```

## 주의사항

- Streamlit 앱을 처음 실행할 때 모델과 데이터를 로드하는 데 시간이 걸릴 수 있습니다.
- 대용량 데이터셋의 경우 SHAP 분석에 시간이 소요될 수 있습니다.
- GPU가 있으면 XAI 분석 속도가 향상됩니다.

## 문제 해결

### 포트 충돌
```bash
streamlit run streamlit_app.py --server.port 8502
```

### 캐시 초기화
```bash
streamlit cache clear
```

### 모든 기능 활성화로 실행
```bash
streamlit run streamlit_app.py --logger.level=debug
```

## 라이선스

© 2026 OHT Fire AI Project
