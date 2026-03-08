#!/bin/bash
# OHT_fire Report Agent 개선사항 - 빠른 시작 가이드

echo "=================================================="
echo "OHT_fire Report Agent 개선사항 적용"
echo "=================================================="
echo ""

echo "📋 변경사항 요약:"
echo "1. ✅ Normal 상태에서 report 생성 제외"
echo "2. ✅ 이전 결과와 다를 때만 report 생성"
echo "3. ✅ RAG (VectorDB) 기반 과거 보고서 참고"
echo "4. ✅ LLM 1차 보고서 (초안) 생성"
echo "5. ✅ 사용자 조치 입력 기반 최종 보고서"
echo "6. ✅ VectorDB에 최종 보고서 저장"
echo "7. ✅ 불필요한 정보 제거 (confidence, probabilities)"
echo ""

echo "📦 설치 단계:"
echo "1. 필수 패키지 설치:"
echo "   $ pip install -r requirements_streamlit.txt"
echo ""

echo "2. Ollama 실행 (선택사항 - LLM 사용 시):"
echo "   $ ollama serve"
echo "   그 다음 다른 터미널에서:"
echo "   $ ollama pull mistral"
echo ""

echo "3. 테스트 실행:"
echo "   $ python test_report_changes.py"
echo ""

echo "4. Streamlit 실행:"
echo "   $ streamlit run streamlit_realtime.py"
echo ""

echo "📁 생성된 파일/디렉토리:"
echo "  report_history/              ← 보고서 히스토리"
echo "  ├── draft/                   ← 1차 보고서 (초안)"
echo "  └── final/                   ← 최종 보고서"
echo "  report_db/                   ← VectorDB"
echo "  REPORT_IMPROVEMENTS.md       ← 상세 문서"
echo "  CHANGES_SUMMARY.txt          ← 변경사항 요약"
echo "  test_report_changes.py       ← 테스트 코드"
echo ""

echo "🔧 수정된 파일:"
echo "  agents/report_agent.py       ← 주요 수정"
echo "  agents/orchestrator.py       ← 조건부 report 생성"
echo "  streamlit_realtime.py        ← 사용자 조치 입력 UI"
echo "  requirements_streamlit.txt   ← 의존성 추가"
echo ""

echo "📊 Streamlit UI 변경사항:"
echo "  - 새 탭 추가: '📄 보고서 관리'"
echo "  - 대기 중인 보고서 목록 표시"
echo "  - 사용자 조치 입력 인터페이스"
echo "  - 최종 보고서 생성 및 저장"
echo ""

echo "🚀 주요 기능:"
echo ""
echo "1. 조건부 Report 생성"
echo "   - Normal (0): 생성 안함"
echo "   - Grey (1): 생성"
echo "   - Warning (2): 생성"
echo "   - Danger (3): 생성"
echo "   - 이전과 같은 상태: 생성 안함"
echo ""

echo "2. 2단계 보고서 프로세스"
echo "   a) 1차: LLM이 현재 상태 분석 (초안)"
echo "      - interpretation_text"
echo "      - recommendations"
echo "   b) 2차: 사용자 조치 입력 후 최종 보고서 생성"
echo "      - final_interpretation"
echo "      - 파일 저장 + VectorDB 저장"
echo ""

echo "3. RAG (Retrieval Augmented Generation)"
echo "   - VectorDB (Chroma)에 최종 보고서 저장"
echo "   - 유사 보고서 검색 및 참고"
echo "   - LLM 프롬프트에 포함"
echo ""

echo "📝 간단한 테스트:"
echo ""
echo "Python 코드:"
echo '```python'
echo 'import asyncio'
echo 'from agents import ReportAgent'
echo 'from datetime import datetime'
echo ''
echo 'async def test():'
echo '    agent = ReportAgent(use_vectordb=False)'
echo '    await agent.initialize()'
echo '    '
echo '    # Danger 상태'
echo '    result = await agent.execute({'
echo '        "predicted_class": 3,'
echo '        "top_features": ['
echo '            {"name": "PM10", "contribution": 0.4},'
echo '        ],'
echo '        "timestamp": datetime.now().isoformat(),'
echo '        "feature_names": ["NTC", "PM10", "PM2.5", ...]'
echo '    })'
echo '    '
echo '    if result["data"]["should_report"]:'
echo '        print(f"✓ Report 생성됨: {result[\"data\"][\"report_id\"]}")'
echo '    else:'
echo '        print(f"ℹ Report 미생성: {result[\"data\"][\"reason\"]}")'
echo ''
echo 'asyncio.run(test())'
echo '```'
echo ""

echo "❓ FAQ:"
echo ""
echo "Q: Ollama가 없어도 되나요?"
echo "A: 네, 없어도 기본 해석이 제공됩니다. LLM 기능만 미사용됩니다."
echo ""

echo "Q: VectorDB를 원하지 않으면?"
echo "A: use_vectordb=False로 설정하면 RAG 기능 없이 동작합니다."
echo ""

echo "Q: 보고서를 데이터베이스에 저장하려면?"
echo "A: report_history/ 디렉토리의 JSON 파일을 읽어서 DB에 저장하면 됩니다."
echo ""

echo "Q: 최종 보고서는 자동 생성 안되나요?"
echo "A: 현재는 사용자가 조치를 입력해야 최종 보고서가 생성됩니다."
echo "   향후 자동 생성 기능을 추가할 수 있습니다."
echo ""

echo "=================================================="
echo "✅ 모든 변경사항이 적용되었습니다!"
echo "=================================================="
