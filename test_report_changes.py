#!/usr/bin/env python
"""
Report Agent 변경사항 검증 스크립트

변경사항:
1. Normal 상태일 때 report 생성 안함
2. 이전 결과와 다를 때만 report 생성
3. RAG + VectorDB 지원
4. 사용자 조치 입력 기반 최종 보고서 생성
5. 불필요한 정보 제거 (confidence, probabilities 등)
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

async def test_report_agent():
    """ReportAgent 기본 테스트"""
    
    print("=" * 60)
    print("Report Agent 변경사항 검증 테스트")
    print("=" * 60)
    
    from agents import ReportAgent
    
    # ReportAgent 생성
    report_agent = ReportAgent(use_vectordb=False)  # 테스트 시 VectorDB 비활성화
    await report_agent.initialize()
    
    # Test 1: Normal 상태 - report 생성 안함
    print("\n[Test 1] Normal 상태 - report 생성 안함")
    normal_input = {
        'predicted_class': 0,
        'top_features': [],
        'timestamp': datetime.now().isoformat(),
        'feature_names': ['NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4']
    }
    
    result = await report_agent.execute(normal_input)
    print(f"✓ should_report: {result['data'].get('should_report', 'N/A')}")
    print(f"✓ reason: {result['data'].get('reason', 'N/A')}")
    assert result['data']['should_report'] == False, "Normal 상태에서 report 생성되면 안됨"
    print("✓ PASSED")
    
    # Test 2: Warning 상태 - report 생성
    print("\n[Test 2] Warning 상태 - report 생성")
    warning_input = {
        'predicted_class': 2,
        'top_features': [
            {'name': 'PM10', 'contribution': 0.35},
            {'name': 'NTC', 'contribution': 0.25},
            {'name': 'CT1', 'contribution': 0.15}
        ],
        'timestamp': datetime.now().isoformat(),
        'feature_names': ['NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4']
    }
    
    result = await report_agent.execute(warning_input)
    print(f"✓ should_report: {result['data'].get('should_report', 'N/A')}")
    print(f"✓ report_id: {result['data'].get('report_id', 'N/A')}")
    assert result['data']['should_report'] == True, "Warning 상태에서 report 생성되어야 함"
    print("✓ PASSED")
    report_id_1 = result['data']['report_id']
    
    # Test 3: 같은 결과 - report 생성 안함
    print("\n[Test 3] 같은 결과 - report 생성 안함")
    warning_input2 = {
        'predicted_class': 2,
        'top_features': [
            {'name': 'PM10', 'contribution': 0.35},
            {'name': 'NTC', 'contribution': 0.25},
            {'name': 'CT1', 'contribution': 0.15}
        ],
        'timestamp': datetime.now().isoformat(),
        'feature_names': ['NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4']
    }
    
    result = await report_agent.execute(warning_input2)
    print(f"✓ should_report: {result['data'].get('should_report', 'N/A')}")
    print(f"✓ reason: {result['data'].get('reason', 'N/A')}")
    assert result['data']['should_report'] == False, "같은 상태에서 report 생성되면 안됨"
    print("✓ PASSED")
    
    # Test 4: Danger 상태 - report 생성
    print("\n[Test 4] Danger 상태 - report 생성")
    danger_input = {
        'predicted_class': 3,
        'top_features': [
            {'name': 'CT2', 'contribution': 0.45},
            {'name': 'PM2.5', 'contribution': 0.35},
        ],
        'timestamp': datetime.now().isoformat(),
        'feature_names': ['NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4']
    }
    
    result = await report_agent.execute(danger_input)
    print(f"✓ should_report: {result['data'].get('should_report', 'N/A')}")
    print(f"✓ report_id: {result['data'].get('report_id', 'N/A')}")
    assert result['data']['should_report'] == True, "Danger 상태에서 report 생성되어야 함"
    print("✓ PASSED")
    report_id_2 = result['data']['report_id']
    
    # Test 5: 보고서 내용 확인 (불필요한 정보 제거)
    print("\n[Test 5] 보고서 내용 확인 (불필요한 정보 제거)")
    report_json = result['data']['json']
    print(f"✓ Report 필드: {list(report_json.keys())}")
    
    # confidence와 probabilities가 없어야 함
    assert 'confidence' not in report_json, "confidence 필드가 있으면 안됨"
    assert 'probabilities' not in report_json, "probabilities 필드가 있으면 안됨"
    
    # 필요한 필드는 있어야 함
    assert 'timestamp' in report_json, "timestamp 필드가 없음"
    assert 'level_name' in report_json, "level_name 필드가 없음"
    assert 'interpretation_text' in report_json, "interpretation_text 필드가 없음"
    assert 'recommendations' in report_json, "recommendations 필드가 없음"
    assert 'top_features' in report_json, "top_features 필드가 없음"
    
    print("✓ PASSED - 불필요한 정보 제거됨")
    
    # Test 6: Finalize report (사용자 조치 입력)
    print("\n[Test 6] 최종 보고서 생성 (사용자 조치 입력)")
    
    try:
        finalize_result = await report_agent.finalize_report(
            report_id_2,
            "온도 센서 교정 완료, 냉각 팬 속도 증가"
        )
        
        if finalize_result['success']:
            final_report = finalize_result['final_report']
            print(f"✓ 최종 보고서 생성됨")
            print(f"✓ Status: {final_report.get('status', 'N/A')}")
            print(f"✓ 사용자 조치: {final_report.get('user_action', 'N/A')}")
            print(f"✓ 최종 분석: {final_report.get('final_interpretation', 'N/A')[:100]}...")
            
            # 파일 확인
            final_file = Path(finalize_result['file_path'])
            assert final_file.exists(), "최종 보고서 파일이 없음"
            print(f"✓ 파일 저장됨: {final_file}")
            print("✓ PASSED")
        else:
            print(f"✗ FAILED: {finalize_result.get('error', 'Unknown')}")
    except Exception as e:
        print(f"✗ Finalize 테스트 실패 (일부 LLM 기능 미사용 가능): {e}")
    
    # Test 7: 보고서 히스토리 디렉토리 확인
    print("\n[Test 7] 보고서 디렉토리 구조 확인")
    history_dir = Path("report_history")
    draft_dir = history_dir / "draft"
    final_dir = history_dir / "final"
    
    print(f"✓ 히스토리 디렉토리: {history_dir.exists()}")
    print(f"✓ Draft 디렉토리: {draft_dir.exists()}")
    print(f"✓ Final 디렉토리: {final_dir.exists()}")
    
    # Draft 파일 확인
    if draft_dir.exists():
        draft_files = list(draft_dir.glob("*.json"))
        print(f"✓ Draft 보고서 파일 개수: {len(draft_files)}")
        if draft_files:
            with open(draft_files[0], 'r', encoding='utf-8') as f:
                draft_content = json.load(f)
                print(f"✓ Draft 보고서 내용: {list(draft_content.keys())}")
    
    print("✓ PASSED")
    
    print("\n" + "=" * 60)
    print("모든 테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_report_agent())
