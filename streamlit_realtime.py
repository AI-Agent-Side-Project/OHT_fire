"""
Streamlit 실시간 대시보드

사용법:
    streamlit run streamlit_realtime.py
"""

import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import time
import pickle
import pdb
import os
import traceback
import joblib

from agents import (
    DataLoaderAgent,
    InferenceAgent,
    XAIAgent,
    ReportAgent,
    DashboardAgent,
    RealtimeOrchestrator
)

# ============================================
# 페이지 설정
# ============================================
st.set_page_config(
    page_title="🔥 OHT Fire Monitor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔥 OHT Fire - 실시간 Multi-Agent 모니터링 시스템")
st.markdown("---")

# ============================================
# Session State 초기화
# ============================================
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'csv_path' not in st.session_state:
    st.session_state.csv_path = None
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'timeseries_data' not in st.session_state:
    st.session_state.timeseries_data = None
if 'error_history' not in st.session_state:
    st.session_state.error_history = []
if 'window_size' not in st.session_state:
    st.session_state.window_size = 16
if 'checkpoint_path' not in st.session_state:
    st.session_state.checkpoint_path = "checkpoints/TSC_dm64_dff128_topk3_sl16_nminmax"
if 'device' not in st.session_state:
    st.session_state.device = "cpu"
if 'iteration' not in st.session_state:
    st.session_state.iteration = 0
if 'current_window' not in st.session_state:
    st.session_state.current_window = None
if 'last_results' not in st.session_state:
    st.session_state.last_results = []
if 'last_report_id' not in st.session_state:
    st.session_state.last_report_id = None
if 'pending_reports' not in st.session_state:
    st.session_state.pending_reports = {}  # {report_id: report_data}

# ============================================
# 사이드바 설정
# ============================================
with st.sidebar:
    st.header("⚙️ 설정")
    
    # CSV 파일 선택
    csv_files = sorted(Path('database/OHT/Test/Data').glob('*.csv'))
    if csv_files:
        selected_csv = st.selectbox(
            "📁 CSV 파일 선택",
            [str(f.name) for f in csv_files],
            key="csv_select"
        )
        st.session_state.csv_path = Path('database/OHT/Test/Data') / selected_csv
    else:
        st.warning("❌ CSV 파일을 찾을 수 없습니다")
        st.session_state.csv_path = None
    
    st.divider()
    
    # 모니터링 제어
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🟢 시작", width='stretch', key="start_btn"):
            st.session_state.is_running = True
            st.session_state.orchestrator = None  # 새로 시작할 때 재초기화
            st.rerun()
    
    with col2:
        if st.button("🛑 중지", width='stretch', key="stop_btn"):
            st.session_state.is_running = False
            st.rerun()
    
    st.divider()
    
    # 설정
    checkpoint_path = st.text_input(
        "📦 체크포인트 경로",
        value="checkpoints/TSC_dm64_dff128_topk3_sl16_nminmax",
        key="checkpoint_input"
    )
    
    device = st.selectbox(
        "🖥️ 디바이스",
        ["cpu", "cuda"],
        key="device_select"
    )
    
    window_size = st.slider(
        "📊 윈도우 크기",
        min_value=16,
        max_value=256,
        value=16,
        key="window_slider"
    )
    st.session_state.window_size = window_size
    st.session_state.checkpoint_path = checkpoint_path
    st.session_state.device = device


# ============================================
# 유틸리티 함수
# ============================================

def load_timeseries_from_csv(csv_path: str) -> pd.DataFrame:
    """CSV 파일에서 시계열 데이터 로드"""
    try:
        df = pd.read_csv(csv_path)
        if df.shape[0] == 0:
            st.error("CSV 파일이 비어있습니다")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"CSV 로드 실패: {e}")
        return pd.DataFrame()


def load_error_history() -> pd.DataFrame:
    """에러 이력 로드 (session state에서)"""
    if not st.session_state.error_history:
        return pd.DataFrame(columns=["timestamp", "severity", "message"])
    return pd.DataFrame(st.session_state.error_history)


def build_report_for_time(end_ts: int, window_data: pd.DataFrame) -> str:
    """선택된 시간에 대한 리포트 생성"""
    report = f"""
### 📊 분석 리포트 (Sample: {end_ts})

**데이터 통계:**
- 샘플 수: {len(window_data)}

**상태:**
- 🟢 정상 모니터링 중
- 이상 감지: 없음
    """
    return report


def plot_window(df_win: pd.DataFrame):
    """윈도우 데이터를 라인 차트로 시각화"""
    if df_win.empty or len(df_win) < 2:
        fig = go.Figure()
        fig.add_annotation(text="데이터 없음")
        return fig
    
    # 숫자형 열만 필터링
    numeric_cols = ['NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4', 
                    'ex_temperature', 'ex_humidity', 'ex_illuminance']
    
    # 컬럼이 실제로 있는지 확인
    available_cols = [col for col in numeric_cols if col in df_win.columns]
    
    # 첫 번째 숫자형 열을 기준으로 그래프 생성
    fig = go.Figure()
    for col in available_cols:
        fig.add_trace(go.Scatter(
            x=list(range(len(df_win))),
            y=df_win[col],
            mode='lines+markers',
            name=str(col)
        ))
    
    fig.update_layout(
        height=350,
        title="시계열 데이터",
        xaxis_title="시간 (샘플)",
        yaxis_title="값",
        hovermode='x unified'
    )
    return fig


# ============================================
# 메인 콘텐츠
# ============================================

# Orchestrator 초기화
if st.session_state.is_running and st.session_state.csv_path:
    if st.session_state.orchestrator is None:
        with st.spinner("🔄 시스템 초기화 중..."):
            try:
                scaler_pkl_path = f"{st.session_state.checkpoint_path}/scaler.pkl"
                
                # 에이전트 생성
                data_loader = DataLoaderAgent(
                    csv_path=str(st.session_state.csv_path),
                    window_size=st.session_state.window_size,
                    delay=1.0,
                    norm_method='minmax',
                    scaler=None,
                    scaler_path=scaler_pkl_path
                )
                
                inference = InferenceAgent(
                    checkpoint_path=st.session_state.checkpoint_path,
                    device=st.session_state.device
                )
                
                feature_names = ['NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4', 
                    'ex_temperature', 'ex_humidity', 'ex_illuminance']
                
                # InferenceAgent에서 실제 모델 예측 함수 가져오기
                model_predict = inference.get_predict_function()
                
                # 저장된 배경 데이터 로드 (Normal class 샘플)
                background_data_path = os.path.join(
                    st.session_state.checkpoint_path, 
                    'background_data.pkl'
                )
                background_data = None
                if os.path.exists(background_data_path):
                    background_data = joblib.load(background_data_path)
                    st.info(f"✓ 배경 데이터 로드됨: {background_data.shape}")
                else:
                    st.warning(f"⚠ 배경 데이터 파일 없음: {background_data_path}")
                    st.info("💡 run.py로 모델 학습 시 자동 생성됩니다")
                
                xai = XAIAgent(
                    model=model_predict,
                    feature_names=feature_names,
                    background_data=background_data
                )
                
                report = ReportAgent()
                dashboard = DashboardAgent()
                
                # Orchestrator 설정
                orchestrator = RealtimeOrchestrator()
                orchestrator.register_agent('data_loader', data_loader)
                orchestrator.register_agent('inference', inference)
                orchestrator.register_agent('xai', xai)
                orchestrator.register_agent('report', report)
                orchestrator.register_agent('dashboard', dashboard)
                
                # 초기화
                asyncio.run(orchestrator.initialize())
                
                st.session_state.orchestrator = orchestrator
                st.success("✓ 시스템 초기화 완료")
            
            except Exception as e:
                st.error(f"❌ 초기화 실패: {e}")
                st.session_state.is_running = False
    
    # 초기 데이터 로드
    if st.session_state.timeseries_data is None:
        with st.spinner("📂 데이터 로딩 중..."):
            try:
                df = load_timeseries_from_csv(str(st.session_state.csv_path))
                st.session_state.timeseries_data = df
                if len(df) > 0:
                    st.info(f"✓ {len(df)} 샘플 로드됨")
            except Exception as e:
                st.error(f"데이터 로딩 실패: {e}")
                st.session_state.is_running = False
    
    df = st.session_state.timeseries_data
    
    if df is not None and len(df) > 0:
        orchestrator = st.session_state.orchestrator
        
        # ========================================
        # 실시간 파이프라인 실행 (한 스텝)
        # ========================================
        
        # Placeholder 생성 (결과 업데이트용)
        placeholder_result = st.empty()
        placeholder_chart = st.empty()
        
        # 한 스텝 파이프라인 실행
        try:
            result = asyncio.run(orchestrator.run_pipeline())
            
            if result['success']:
                st.session_state.iteration += 1
                st.session_state.last_results.append(result)
                
                # Inference 데이터 추출
                inf_step = result['steps'].get('inference', {})
                inf_data = inf_step.get('data', {})
                dl_step = result['steps'].get('data_loader', {})
                dl_data = dl_step.get('data', {})
                
                # Severity가 0이 아니면 에러 히스토리에 기록
                severity_level = inf_data.get('predicted_class', 0)
                if severity_level != 0:
                    severity_text = {0: 'Normal', 1: 'Grey Zone', 2: 'Warning', 3: 'Danger'}.get(severity_level, 'Unknown')
                    st.session_state.error_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'severity': severity_text,
                        'message': f"Index: {dl_data.get('index', 'N/A')}, Class: {inf_data.get('class_name', 'Unknown')}"
                    })
                
                # Report 결과 처리
                report_step = result['steps'].get('report', {})
                report_data = report_step.get('data', {})
                should_report = report_data.get('should_report', False)
                report_id = report_data.get('report_id', '')
                
                # 상세 결과
                with placeholder_result.container():
                    st.markdown("### 📊 상세 결과")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Inference (모델 추론)**")
                        st.json({
                            "시간(ms)": inf_data.get('timestamp', 0),
                            "Severity": inf_data.get('class_name', 'N/A')
                        })
                    
                    with col2:
                        st.markdown("**Report 상태**")
                        if should_report:
                            st.success(f"✓ 보고서 생성됨: {report_id}")
                            st.session_state.last_report_id = report_id
                            st.session_state.pending_reports[report_id] = report_data
                        else:
                            reason = report_data.get('reason', '알 수 없음')
                            st.info(f"ℹ 보고서 미생성: {reason}")
                
                # 현재 윈도우 시각화
                # DataLoader에서 생성된 윈도우 데이터 추출
                data_loader_result = result['steps'].get('data_loader', {})
                window_data_list = data_loader_result.get('data', {}).get('window_data', None)
                feature_names = data_loader_result.get('data', {}).get('feature_names', [])
                
                if window_data_list is not None and len(window_data_list) > 0:
                    # 리스트를 DataFrame으로 변환
                    window_df = pd.DataFrame(
                        window_data_list,
                        columns=feature_names if feature_names else [f'Col_{i}' for i in range(len(window_data_list[0]))]
                    )
                    st.session_state.current_window = window_df
                    with placeholder_chart.container():
                        st.markdown("### 📈 현재 윈도우 데이터 (sliding window)")
                        st.plotly_chart(plot_window(window_df), width='stretch')
            else:
                st.error(f"❌ 파이프라인 오류: {result.get('error', 'Unknown')}")
        
        except Exception as e:
            st.error(f"❌ 실행 오류: {e}")
            import traceback
            st.code(traceback.format_exc())
        
        # ========================================
        # 탭 UI (히스토리 보기)
        # ========================================
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📊 결과 히스토리", "📋 에러 이력", "📄 보고서 관리"])
        
        with tab1:
            st.subheader("최근 실시간 결과들 (최대 20개)")
            
            if st.session_state.last_results:
                # 최근 20개만 표시
                recent_results = st.session_state.last_results[-20:]
                
                # 결과 요약 테이블
                summary_data = []
                for i, res in enumerate(recent_results, 1):
                    dl_step = res['steps'].get('data_loader', {})
                    dl_data = dl_step.get('data', {})
                    inference_step = res['steps'].get('inference', {})
                    inf_data = inference_step.get('data', {})
                    
                    severity_level = inf_data.get('predicted_class', 0)
                    severity_text = {0: 'Normal', 1: 'Grey Zone', 2: 'Warning', 3: 'Danger'}.get(severity_level, 'Unknown')
                    
                    summary_data.append({
                        "시간(ms)": dl_data.get('timestamp', 0),
                        "Severity": severity_text
                    })
                
                st.dataframe(pd.DataFrame(summary_data), width = 'stretch')
            else:
                st.info("결과 히스토리가 없습니다")
        
        with tab2:
            st.subheader("에러 이력 및 알림 (Severity != Normal)")
            
            err_df = load_error_history()
            
            if err_df.empty:
                st.info("🟢 현재까지 이상 감지 없음 (모두 Normal)")
            else:
                # 타임스탬프를 보기 좋게 포맷팅
                err_df_display = err_df.copy()
                err_df_display['timestamp'] = pd.to_datetime(err_df_display['timestamp']).dt.strftime('%H:%M:%S')
                st.dataframe(err_df_display, width='stretch', height=400)
                
                # 통계
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 이상 감지", len(err_df))
                with col2:
                    danger_count = len(err_df[err_df['severity'] == 'Danger'])
                    st.metric("🚨 Danger", danger_count)
                with col3:
                    warning_count = len(err_df[err_df['severity'] == 'Warning'])
                    st.metric("⚠️ Warning", warning_count)
        
        with tab3:
            st.subheader("📄 생성된 보고서 관리")
            
            if st.session_state.pending_reports:
                # 대기 중인 보고서 목록
                st.markdown("#### 대기 중인 보고서 (사용자 조치 입력 필요)")
                
                for report_id, report_data in st.session_state.pending_reports.items():
                    with st.expander(f"📋 {report_id} - {report_data.get('level_name', 'Unknown')}", expanded=True):
                        # 1차 보고서 표시
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**현재 상태**")
                            st.write(f"• {report_data.get('level_name', 'N/A')}")
                            st.write(f"• 시간: {report_data.get('timestamp', 'N/A')}")
                        
                        with col2:
                            st.markdown("**1차 분석 (초안)**")
                            st.info(report_data.get('interpretation_text', 'N/A'))
                        
                        # 상위 특성 표시
                        if report_data.get('top_features'):
                            st.markdown("**영향 특성**")
                            for feat in report_data['top_features'][:3]:
                                st.write(f"• {feat['name']}: {feat['contribution']*100:.1f}%")
                        
                        # 권장사항 표시
                        if report_data.get('recommendations'):
                            st.markdown("**권장사항**")
                            for rec in report_data['recommendations']:
                                st.write(f"• {rec}")
                        
                        # 사용자 조치 입력
                        st.markdown("---")
                        st.markdown("**실제 조치 입력**")
                        user_action = st.text_area(
                            "이상을 해결하기 위해 실시한 조치를 입력하세요:",
                            key=f"action_{report_id}",
                            height=100,
                            placeholder="예: 온도 센서 교정 완료, 케이블 재연결 등..."
                        )
                        
                        col_submit, col_cancel = st.columns(2)
                        with col_submit:
                            if st.button("✅ 최종 보고서 생성", key=f"submit_{report_id}"):
                                if not user_action.strip():
                                    st.error("조치 내용을 입력해주세요")
                                else:
                                    with st.spinner("최종 보고서 생성 중..."):
                                        try:
                                            # Finalize report 호출
                                            finalize_result = asyncio.run(
                                                orchestrator.agents['report'].finalize_report(report_id, user_action)
                                            )
                                            if finalize_result['success']:
                                                final_report = finalize_result['final_report']
                                                st.success("✓ 최종 보고서가 생성되었습니다")
                                                
                                                # 최종 보고서 표시
                                                st.markdown("**최종 분석 결과**")
                                                st.info(final_report.get('final_interpretation', 'N/A'))
                                                
                                                # 보고서 제거
                                                del st.session_state.pending_reports[report_id]
                                                st.rerun()
                                            else:
                                                st.error(f"오류: {finalize_result.get('error', 'Unknown')}")
                                        except Exception as e:
                                            st.error(f"오류 발생: {e}")
                        
                        with col_cancel:
                            if st.button("❌ 취소", key=f"cancel_{report_id}"):
                                del st.session_state.pending_reports[report_id]
                                st.rerun()
            
            else:
                st.info("현재 대기 중인 보고서가 없습니다")

        st.rerun()
    
    else:
        st.warning("❌ 로드된 데이터가 없습니다")

else:
    st.info("🔄 CSV 파일을 선택하고 '시작' 버튼을 클릭하세요")
