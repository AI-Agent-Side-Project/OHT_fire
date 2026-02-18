import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="OHT-Fire AI Agent",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Session State 초기화
# ============================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = {}  # {csv_file: deque of results}

if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# ============================================
# 데이터 스택 관리 함수
# ============================================
def update_prediction_stack(model_name, csv_files, max_history=1000):
    """
    CSV 파일 기반으로 예측 결과를 스택에 저장
    - CSV 파일이 변경되면 새로 시작
    - 같은 CSV 파일이면 결과를 누적
    """
    current_csv_key = tuple(sorted(csv_files))
    
    # 모델이 변경되었거나 CSV 파일이 변경된 경우 초기화
    if st.session_state.current_model != model_name or current_csv_key not in st.session_state.prediction_history:
        st.session_state.current_model = model_name
        st.session_state.prediction_history[current_csv_key] = {
            'data': deque(maxlen=max_history),
            'csv_files': csv_files,
            'model': model_name
        }
    
    return st.session_state.prediction_history[current_csv_key]

def add_prediction_to_stack(stack, prediction_data):
    """예측 데이터를 스택에 추가"""
    stack['data'].append(prediction_data)

def get_stacked_predictions(stack):
    """스택된 모든 예측 데이터 반환"""
    if stack['data']:
        # 최신 데이터 순서대로 반환
        return list(stack['data'])
    return []

def _display_sample_explanation(explanation, sample_idx):
    """
    샘플별 설명 표시 (Instance-level SHAP explanation)
    """
    # 기본 정보
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pred_class = explanation['predicted_class']
        pred_prob = explanation['predicted_probability']
        st.metric(
            "Predicted Class",
            f"Class {pred_class}",
            f"{pred_prob:.1%}"
        )
    
    with col2:
        true_class = explanation['true_class']
        is_correct = explanation['is_correct']
        status = "✅ Correct" if is_correct else "❌ Wrong"
        st.metric(
            "True Class",
            f"Class {true_class}",
            status
        )
    
    with col3:
        contrast_class = explanation['contrast_class']
        contrast_prob = explanation['contrast_probability']
        st.metric(
            "2nd Highest Class",
            f"Class {contrast_class}",
            f"{contrast_prob:.1%}"
        )
    
    with col4:
        margin = explanation['predicted_probability'] - explanation['contrast_probability']
        st.metric(
            "Confidence Margin",
            f"{margin:.1%}",
            delta=None
        )
    
    st.markdown("---")
    
    # 확률 분포
    st.subheader("Class Probability Distribution")
    probs_dict = explanation['all_class_probabilities']
    class_labels = sorted(probs_dict.keys())
    class_probs = [probs_dict[label] for label in class_labels]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#ff6b6b' if label == f'class_{explanation["predicted_class"]}' else '#4dabf7' 
              for label in class_labels]
    bars = ax.bar(class_labels, class_probs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'Predicted Class Probabilities - Sample {sample_idx}', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 확률 값 표시
    for bar, prob in zip(bars, class_probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{prob:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # 예측에 기여한 상위 특성 (Instance-level)
    st.subheader(f"🎯 Top Features That Predicted Class {explanation['predicted_class']}")
    
    top_features = explanation['top_contributing_features']
    
    # 상위 특성 시각화
    fig, ax = plt.subplots(figsize=(10, 5))
    feature_names = [f"Feature {f['feature_idx']}" for f in top_features]
    contributions = [f['contribution_magnitude'] for f in top_features]
    colors_features = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(feature_names)))
    
    bars = ax.barh(feature_names, contributions, color=colors_features, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Contribution Magnitude |SHAP|', fontsize=11, fontweight='bold')
    ax.set_title(f'Top Contributing Features for Class {explanation["predicted_class"]} - Sample {sample_idx}', 
                fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    # 값 표시
    for bar, contrib in zip(bars, contributions):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{contrib:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 특성별 기여도 테이블
    st.markdown("#### Feature Contribution Details")
    feature_contribution_df = pd.DataFrame({
        'Rank': range(1, len(top_features) + 1),
        'Feature Index': [f['feature_idx'] for f in top_features],
        'Contribution Score': [f['contribution_magnitude'] for f in top_features],
        'Relative Importance': [f"{f['contribution_magnitude']/contributions[0]*100:.1f}%" for f in top_features]
    })
    st.dataframe(feature_contribution_df, use_container_width=True, hide_index=True)
    
    # 모든 특성의 기여도 (히트맵)
    st.markdown("#### All Features Contribution Heatmap")
    all_contributions = explanation['feature_contribution_scores']
    
    # reshape for visualization (seq_len x num_features 형태 가정)
    # 만약 단순 1D라면 그대로 표시
    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow([all_contributions], cmap='RdYlGn_r', aspect='auto')
    ax.set_xlabel('Feature Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sample', fontsize=11, fontweight='bold')
    ax.set_title(f'All Features Contribution Scores for Class {explanation["predicted_class"]}', 
                fontsize=12, fontweight='bold')
    ax.set_yticks([0])
    ax.set_yticklabels([f'Sample {sample_idx}'])
    
    plt.colorbar(im, ax=ax, label='|SHAP value|')
    plt.tight_layout()
    st.pyplot(fig)
    
    # 해석
    st.markdown("#### 📋 Interpretation")
    explanation_text = f"""
    **Summary:** This sample was predicted as **Class {explanation['predicted_class']}** 
    with a confidence of **{explanation['predicted_probability']:.1%}**.
    
    **Top Contributing Features:** The model's decision was primarily influenced by:
    """
    for i, feature in enumerate(top_features[:3], 1):
        explanation_text += f"\n{i}. Feature {feature['feature_idx']} (score: {feature['contribution_magnitude']:.4f})"
    
    explanation_text += f"""
    
    **Prediction Confidence:** The margin between the predicted class and the second-highest class 
    is {explanation['predicted_probability'] - explanation['contrast_probability']:.1%}, 
    indicating a {'strong' if explanation['predicted_probability'] - explanation['contrast_probability'] > 0.3 else 'moderate' if explanation['predicted_probability'] - explanation['contrast_probability'] > 0.15 else 'weak'} prediction.
    
    **Correctness:** This prediction is {'✅ CORRECT' if explanation['is_correct'] else '❌ INCORRECT'}.
    """
    
    st.info(explanation_text)

# 커스텀 CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    .alert-high {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff0000;
    }
    .alert-medium {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .alert-low {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# 제목
st.title("🔥 OHT Fire - AI Prediction & XAI Dashboard with Data Stacking")
st.markdown("---")

# 사이드바 설정
st.sidebar.title("⚙️ Settings")

# 1. 모델 선택
available_models = []
results_dir = Path('./test_results')
if results_dir.exists():
    available_models = [d.name for d in results_dir.iterdir() if d.is_dir()]

selected_model = st.sidebar.selectbox(
    "Select Model",
    available_models,
    help="Choose a trained model to analyze"
)

# 2. XAI 결과 로드
xai_dir = Path('./xai_results') / selected_model
xai_available = xai_dir.exists()

st.sidebar.info(
    f"✓ XAI Results Available" if xai_available else "✗ XAI Results Not Available"
)

# 데이터 스택 설정
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Stacking Settings")

max_history = st.sidebar.slider(
    "Maximum Stacked Records",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
    help="Maximum number of predictions to keep in history"
)

if st.sidebar.button("🔄 Clear History", help="Clear all stacked prediction data"):
    st.session_state.prediction_history = {}
    st.session_state.current_model = None
    st.success("History cleared!")

# 메인 컨텐츠
if selected_model:
    
    # Tab 생성
    tab1, tab2, tab2_5, tab3, tab4, tab5 = st.tabs([
        "📊 Model Predictions",
        "🔍 XAI Analysis",
        "🎯 Sample Explanation",
        "📈 Feature Importance",
        "⚠️ Alarm & Insights",
        "📚 Prediction History"
    ])
    
    # ============================================
    # TAB 1: Model Predictions
    # ============================================
    with tab1:
        st.subheader("Model Prediction Results")
        
        # 예측 결과 파일 로드
        analysis_file = xai_dir / 'xai_analysis.json' if xai_available else None
        
        if analysis_file and analysis_file.exists():
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            
            # 데이터 스택 업데이트
            csv_files = ["test_data"]  # 실제로는 file_mapping에서 가져올 수 있음
            stack = update_prediction_stack(selected_model, csv_files, max_history)
            add_prediction_to_stack(stack, analysis_data)
            
            # 메트릭 표시
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Samples",
                    analysis_data['total_samples'],
                    help="Number of test samples analyzed"
                )
            
            with col2:
                st.metric(
                    "Number of Classes",
                    analysis_data['num_classes'],
                    help="Number of fire classes"
                )
            
            with col3:
                accuracy = analysis_data['model_accuracy']
                st.metric(
                    "Model Accuracy",
                    f"{accuracy:.2%}",
                    help="Accuracy on test dataset"
                )
            
            with col4:
                stacked_count = len(list(stack['data']))
                st.metric(
                    "Stacked Records",
                    stacked_count,
                    help=f"Number of stacked predictions (max: {max_history})"
                )
            
            # 예측 결과 분포
            st.subheader("Prediction Distribution")
            
            predictions = np.array(analysis_data['predictions'])
            true_labels = np.array(analysis_data['true_labels'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 예측 클래스 분포
                pred_dist = pd.Series(predictions).value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = plt.cm.Set3(range(len(pred_dist)))
                bars = ax.bar(
                    [f'Class {i}' for i in pred_dist.index],
                    pred_dist.values,
                    color=colors,
                    edgecolor='black',
                    linewidth=1.5
                )
                ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
                ax.set_title('Predicted Class Distribution', fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                # 값 라벨
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # 실제 클래스 분포
                true_dist = pd.Series(true_labels).value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = plt.cm.Set3(range(len(true_dist)))
                bars = ax.bar(
                    [f'Class {i}' for i in true_dist.index],
                    true_dist.values,
                    color=colors,
                    edgecolor='black',
                    linewidth=1.5
                )
                ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
                ax.set_title('True Class Distribution', fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                # 값 라벨
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown("---")
            
            # Confidence Distribution
            st.subheader("Prediction Confidence Distribution")
            
            probs = np.array(analysis_data['prediction_probabilities'])
            max_probs = np.max(probs, axis=1)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(max_probs, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(max_probs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(max_probs):.3f}')
            ax.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title('Model Prediction Confidence Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # 상세 테이블
            st.subheader("Detailed Predictions (First 50 samples)")
            
            results_df = pd.DataFrame({
                'Sample ID': range(len(predictions[:50])),
                'Predicted Class': predictions[:50],
                'True Class': true_labels[:50],
                'Correct': (predictions[:50] == true_labels[:50]).astype(int),
                'Confidence': np.max(probs[:50], axis=1),
            })
            
            st.dataframe(results_df, use_container_width=True)
        
        else:
            st.warning("No prediction results found. Please run the model first.")
    
    # ============================================
    # TAB 2: XAI Analysis
    # ============================================
    with tab2:
        st.subheader("SHAP-based Explainability Analysis")
        
        if xai_available:
            # Summary 텍스트 표시
            summary_file = xai_dir / 'xai_summary.txt'
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_text = f.read()
                st.text(summary_text)
            
            st.markdown("---")
            
            # SHAP Values 통계
            st.subheader("SHAP Values Statistics")
            
            # 각 클래스의 SHAP values 로드
            shap_files = list(xai_dir.glob('shap_values_class_*.npy'))
            
            if shap_files:
                shap_stats = {}
                for shap_file in sorted(shap_files):
                    class_idx = shap_file.stem.split('_')[-1]
                    shap_values = np.load(shap_file)
                    
                    shap_stats[f'Class {class_idx}'] = {
                        'Mean |SHAP|': np.mean(np.abs(shap_values)),
                        'Std |SHAP|': np.std(np.abs(shap_values)),
                        'Max |SHAP|': np.max(np.abs(shap_values)),
                        'Shape': f"{shap_values.shape}",
                    }
                
                stats_df = pd.DataFrame(shap_stats).T
                st.dataframe(stats_df, use_container_width=True)
        else:
            st.warning("XAI analysis results not found. Please run with --use_xai flag.")
    
    # ============================================
    # TAB 2.5: Sample Explanation (Instance-level)
    # ============================================
    with tab2_5:
        st.subheader("🎯 Why Was This Sample Predicted as This Class?")
        
        if xai_available:
            # sample_explanations.json 로드
            sample_exp_file = xai_dir / 'sample_explanations.json'
            
            if sample_exp_file.exists():
                with open(sample_exp_file, 'r') as f:
                    sample_explanations = json.load(f)
                
                st.markdown("""
                This section explains **why the model made a specific prediction** for each sample.
                Instead of showing global feature importance, it shows the actual features that contributed
                to each sample being classified as a particular class.
                """)
                
                # 샘플 선택 UI
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    sample_selection = st.selectbox(
                        "Select a sample to explain:",
                        range(len(sample_explanations)),
                        format_func=lambda i: f"Sample {i} → Predicted: Class {sample_explanations[i]['predicted_class']} (True: Class {sample_explanations[i]['true_class']})"
                    )
                
                with col2:
                    show_all = st.checkbox("Show all samples", False)
                
                st.markdown("---")
                
                if show_all:
                    # 모든 샘플 표시
                    for idx, exp in enumerate(sample_explanations):
                        with st.expander(
                            f"📌 Sample {idx}: Predicted as **Class {exp['predicted_class']}** "
                            f"(Prob: {exp['predicted_probability']:.2%}) | "
                            f"True: Class {exp['true_class']} {'✅' if exp['is_correct'] else '❌'}"
                        ):
                            _display_sample_explanation(exp, idx)
                else:
                    # 선택된 샘플만 표시
                    exp = sample_explanations[sample_selection]
                    _display_sample_explanation(exp, sample_selection)
                    
            else:
                st.warning("Sample explanations file not found. Please re-run XAI analysis.")
        else:
            st.warning("XAI analysis results not found. Please run with --use_xai flag.")
    
    # ============================================
    # TAB 4: Feature Importance
    # ============================================
    with tab4:
        st.subheader("Feature Importance Analysis")
        
        if xai_available:
            # Feature importance 이미지 표시
            importance_img = xai_dir / 'feature_importance.png'
            if importance_img.exists():
                st.image(str(importance_img), use_column_width=True, caption="Feature Importance based on SHAP")
            
            st.markdown("---")
            
            # Feature importance 수치 표시
            analysis_file = xai_dir / 'xai_analysis.json'
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    analysis_data = json.load(f)
                
                st.subheader("Feature Importance Values")
                
                importance_dict = analysis_data['feature_importance']
                
                # 각 클래스별 상위 특성
                for class_name, importance_values in importance_dict.items():
                    st.subheader(f"📌 {class_name}")
                    
                    # DataFrame으로 변환
                    importance_df = pd.DataFrame({
                        'Feature': [f'Feature {i}' for i in range(len(importance_values))],
                        'Importance': importance_values
                    }).sort_values('Importance', ascending=False)
                    
                    # 상위 10개
                    top_10 = importance_df.head(10)
                    
                    col1, col2 = st.columns([1.5, 1])
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        colors = plt.cm.viridis(np.linspace(0, 1, len(top_10)))
                        bars = ax.barh(top_10['Feature'], top_10['Importance'], color=colors)
                        ax.set_xlabel('Mean |SHAP value|', fontsize=11, fontweight='bold')
                        ax.set_title(f'Top 10 Important Features - {class_name}', fontsize=12, fontweight='bold')
                        ax.invert_yaxis()
                        
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width, bar.get_y() + bar.get_height()/2.,
                                   f'{width:.6f}', ha='left', va='center', fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col2:
                        st.dataframe(top_10, use_container_width=True)
                    
                    st.markdown("---")
        else:
            st.warning("XAI analysis results not found.")
    
    # ============================================
    # TAB 5: Alarm & Insights
    # ============================================
    with tab5:
        st.subheader("⚠️ Alarm System & Risk Assessment")
        
        if analysis_file and analysis_file.exists():
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            
            predictions = np.array(analysis_data['predictions'])
            true_labels = np.array(analysis_data['true_labels'])
            probs = np.array(analysis_data['prediction_probabilities'])
            
            # 알람 규칙 설정
            st.subheader("Alarm Rules Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                confidence_threshold = st.slider(
                    "Confidence Threshold for High Alert",
                    0.0, 1.0, 0.7,
                    help="Alert when confidence is below this threshold"
                )
            
            with col2:
                high_fire_classes = st.multiselect(
                    "High-Risk Fire Classes",
                    range(analysis_data['num_classes']),
                    default=[3],  # 가장 높은 등급
                    help="Classes considered as high-risk fire"
                )
            
            with col3:
                medium_fire_classes = st.multiselect(
                    "Medium-Risk Fire Classes",
                    range(analysis_data['num_classes']),
                    default=[2],
                    help="Classes considered as medium-risk fire"
                )
            
            st.markdown("---")
            
            # 알람 생성
            st.subheader("Generated Alarms")
            
            # 알람 카운트
            high_risk_count = 0
            medium_risk_count = 0
            low_confidence_count = 0
            
            alarm_data = []
            
            for idx in range(len(predictions)):
                pred = predictions[idx]
                confidence = np.max(probs[idx])
                
                alarms = []
                risk_level = "🟢 LOW"
                
                # 고위험 체크
                if pred in high_fire_classes:
                    alarms.append("🔴 HIGH RISK CLASS")
                    risk_level = "🔴 CRITICAL"
                    high_risk_count += 1
                
                # 중위험 체크
                elif pred in medium_fire_classes:
                    alarms.append("🟠 MEDIUM RISK CLASS")
                    risk_level = "🟠 HIGH"
                    medium_risk_count += 1
                
                # 낮은 신뢰도 체크
                if confidence < confidence_threshold:
                    alarms.append(f"⚠️ LOW CONFIDENCE ({confidence:.2%})")
                    if risk_level == "🟢 LOW":
                        risk_level = "🟡 MEDIUM"
                    low_confidence_count += 1
                
                if alarms:  # 알람이 있는 경우만
                    alarm_data.append({
                        'Sample ID': idx,
                        'Predicted Class': pred,
                        'Confidence': f"{confidence:.2%}",
                        'Alarms': ' | '.join(alarms),
                        'Risk Level': risk_level,
                        'True Class': true_labels[idx],
                        'Correct': '✓' if pred == true_labels[idx] else '✗'
                    })
            
            # 알람 통계
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🔴 High Risk Detected",
                    high_risk_count,
                    help="Number of high-risk fire predictions"
                )
            
            with col2:
                st.metric(
                    "🟠 Medium Risk Detected",
                    medium_risk_count,
                    help="Number of medium-risk fire predictions"
                )
            
            with col3:
                st.metric(
                    "⚠️ Low Confidence",
                    low_confidence_count,
                    help="Number of predictions with low confidence"
                )
            
            with col4:
                total_alarms = len(alarm_data)
                st.metric(
                    "🚨 Total Alarms",
                    total_alarms,
                    help="Total number of triggered alarms"
                )
            
            st.markdown("---")
            
            # 알람 상세 보기
            if alarm_data:
                st.subheader("Triggered Alarms (Detailed List)")
                
                alarm_df = pd.DataFrame(alarm_data)
                
                # 위험도별 색상 표시
                st.dataframe(
                    alarm_df,
                    use_container_width=True,
                    height=400
                )
                
                st.markdown("---")
                
                # 알람 분포
                st.subheader("Alarm Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    risk_counts = alarm_df['Risk Level'].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors_map = {'🔴 CRITICAL': '#ff0000', '🟠 HIGH': '#ff9900', '🟡 MEDIUM': '#ffff00', '🟢 LOW': '#00ff00'}
                    colors = [colors_map.get(risk, '#999999') for risk in risk_counts.index]
                    
                    ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                          colors=colors, startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
                    ax.set_title('Alarm Distribution by Risk Level', fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    # 알람 유형 분석
                    all_alarms = []
                    for alarm_list in alarm_df['Alarms']:
                        all_alarms.extend([a.strip() for a in alarm_list.split('|')])
                    
                    alarm_counts = pd.Series(all_alarms).value_counts()
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.barh(range(len(alarm_counts)), alarm_counts.values, color='coral', edgecolor='black')
                    ax.set_yticks(range(len(alarm_counts)))
                    ax.set_yticklabels(alarm_counts.index)
                    ax.set_xlabel('Count', fontsize=11, fontweight='bold')
                    ax.set_title('Alarm Type Distribution', fontsize=12, fontweight='bold')
                    ax.invert_yaxis()
                    
                    for i, v in enumerate(alarm_counts.values):
                        ax.text(v, i, f' {int(v)}', va='center', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                st.markdown("---")
                
                # 권장사항
                st.subheader("💡 Recommendations")
                
                recommendations = []
                
                if high_risk_count > 0:
                    recommendations.append(
                        f"🔴 **URGENT**: {high_risk_count} high-risk fire predictions detected. "
                        f"Immediate action required for fire prevention and response."
                    )
                
                if medium_risk_count > 0:
                    recommendations.append(
                        f"🟠 **WARNING**: {medium_risk_count} medium-risk fire predictions detected. "
                        f"Enhanced monitoring and precautions recommended."
                    )
                
                if low_confidence_count > 0:
                    recommendations.append(
                        f"⚠️ **CAUTION**: {low_confidence_count} predictions have low confidence scores. "
                        f"Manual verification recommended for these cases."
                    )
                
                if not recommendations:
                    recommendations.append("✅ **GOOD NEWS**: No alarms triggered. System status normal.")
                
                for rec in recommendations:
                    st.info(rec)
            
            else:
                st.success("✅ No alarms triggered! All predictions are within safe parameters.")
        
        else:
            st.warning("No prediction results found.")
    
    # ============================================
    # TAB 6: Prediction History (신규)
    # ============================================
    with tab6:
        st.subheader("📚 Stacked Prediction History")
        
        # 현재 스택 정보
        csv_key = None
        for key, stack_info in st.session_state.prediction_history.items():
            if stack_info['model'] == selected_model:
                csv_key = key
                break
        
        if csv_key and st.session_state.prediction_history[csv_key]['data']:
            stack = st.session_state.prediction_history[csv_key]
            stacked_data = get_stacked_predictions(stack)
            
            st.info(
                f"📊 **Model**: {stack['model']}\n\n"
                f"**CSV Files**: {', '.join(stack['csv_files'])}\n\n"
                f"**Total Stacked Records**: {len(stacked_data)}"
            )
            
            st.markdown("---")
            
            # 시간에 따른 정확도 추이
            if len(stacked_data) > 1:
                st.subheader("Accuracy Trend Over Time")
                
                accuracies = [d['model_accuracy'] for d in stacked_data]
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(range(len(accuracies)), accuracies, marker='o', linewidth=2, markersize=8, color='steelblue')
                ax.axhline(np.mean(accuracies), color='red', linestyle='--', linewidth=2, label=f'Average: {np.mean(accuracies):.3f}')
                ax.set_xlabel('Prediction Index', fontsize=11, fontweight='bold')
                ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
                ax.set_title('Model Accuracy Over Stacked Predictions', fontsize=12, fontweight='bold')
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("---")
            
            # 클래스 분포 누적
            st.subheader("Cumulative Class Predictions")
            
            all_predictions = []
            for d in stacked_data:
                all_predictions.extend(d['predictions'])
            
            cumulative_dist = pd.Series(all_predictions).value_counts().sort_index()
            
            col1, col2 = st.columns([1.5, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.Set3(range(len(cumulative_dist)))
                bars = ax.bar(
                    [f'Class {i}' for i in cumulative_dist.index],
                    cumulative_dist.values,
                    color=colors,
                    edgecolor='black',
                    linewidth=1.5
                )
                ax.set_ylabel('Total Predictions', fontsize=11, fontweight='bold')
                ax.set_title('Cumulative Class Distribution', fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.dataframe(
                    cumulative_dist.to_frame('Count'),
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # 상세 히스토리 테이블
            st.subheader("Detailed History")
            
            history_records = []
            for idx, data in enumerate(stacked_data):
                history_records.append({
                    'Index': idx + 1,
                    'Total Samples': data['total_samples'],
                    'Accuracy': f"{data['model_accuracy']:.2%}",
                    'High Risk': sum(1 for p in data['predictions'] if p == 3),
                    'Medium Risk': sum(1 for p in data['predictions'] if p == 2),
                })
            
            history_df = pd.DataFrame(history_records)
            st.dataframe(history_df, use_container_width=True)
            
            # CSV 다운로드
            st.markdown("---")
            csv_export = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History as CSV",
                data=csv_export,
                file_name=f"prediction_history_{selected_model}.csv",
                mime="text/csv"
            )
        
        else:
            st.info("ℹ️ No stacked prediction history yet. Run a prediction to start collecting data.")

else:
    st.info("👈 Please select a model from the sidebar to get started.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p style='color: #888'>OHT Fire - AI Prediction & XAI Dashboard with Data Stacking | Powered by Streamlit</p>
        <p style='color: #aaa; font-size: 0.8em'>© 2026 | Model: TimesNet | XAI: SHAP | Stacking: Deque</p>
    </div>
    """,
    unsafe_allow_html=True
)
