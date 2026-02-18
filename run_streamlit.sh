#!/bin/bash

# OHT Fire - Streamlit Dashboard 실행 스크립트

echo "🔥 OHT Fire - AI Prediction & XAI Dashboard"
echo "=========================================="
echo ""

# 현재 디렉토리 확인
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found in current directory"
    echo "Please run this script from the OHT_fire directory"
    exit 1
fi

# 의존성 설치 여부 확인
echo "📦 Checking dependencies..."

if ! python -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip install -r requirements_streamlit.txt
else
    echo "✓ Dependencies already installed"
fi

echo ""
echo "🚀 Starting Streamlit Dashboard..."
echo "========================================"
echo "📍 Dashboard will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "========================================"
echo ""

# Streamlit 실행
streamlit run streamlit_app.py
