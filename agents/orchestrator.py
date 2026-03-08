"""
RealtimeOrchestrator: 모든 에이전트 조율 및 워크플로우 관리

생명주기:
1. DataLoader → 센서 데이터 로드
2. Inference → 모델 예측
3. XAI → 해석
4. Anomaly → 이상 감지
5. Report → 보고서 생성
6. Dashboard → UI 업데이트
"""

import asyncio
import pdb
from typing import Optional, Dict, Any, List
from .base_agent import BaseAgent
from .data_loader_agent import DataLoaderAgent
from .inference_agent import InferenceAgent
from .xai_agent import XAIAgent
from .report_agent import ReportAgent
from .dashboard_agent import DashboardAgent


class RealtimeOrchestrator:
    """실시간 Multi-Agent 오케스트레이터"""

    def __init__(self):
        """초기화"""
        self.agents: Dict[str, BaseAgent] = {}
        self.is_running = False
        self.logger_name = "Orchestrator"

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """에이전트 등록"""
        self.agents[name] = agent
        print(f"✓ {name} 에이전트 등록됨")

    async def initialize(self) -> bool:
        """모든 에이전트 초기화"""
        print("\n🚀 Agent System 초기화 시작...")
        print("=" * 50)
        
        for name, agent in self.agents.items():
            success = await agent.initialize()
            if not success:
                print(f"✗ {name} 초기화 실패")
                return False
            print(f"✓ {name} 초기화 완료")
        
        print("=" * 50)
        print("✓ 모든 에이전트 초기화 완료\n")
        return True

    async def run_pipeline(self, base_time: Optional[str] = None) -> Dict[str, Any]:
        """
        5-단계 파이프라인 실행

        Returns:
            {
                'success': bool,
                'data_loader': {...},
                'inference': {...},
                'xai': {...},
                'report': {...},
                'dashboard': {...},
                'total_time': float
            }
        """
        
        import time
        start_time = time.time()
        
        result = {
            'success': True,
            'steps': {},
            'total_time': 0
        }
        
        try:
            # Step 1: DataLoader
            print("\n📥 Step 1: 데이터 로딩...")
            data_loader: DataLoaderAgent = self.agents.get('data_loader')
            step1_result = await data_loader.execute({
                'base_time': base_time or ''
            })
            result['steps']['data_loader'] = step1_result
            if not step1_result['success']:
                raise Exception(f"DataLoader 실패: {step1_result['error']}")
            print(f"✓ {step1_result['execution_time']:.1f}ms")
            data = step1_result['data']
            
            # Step 2: Inference
            print("🧠 Step 2: 모델 추론...")
            inference: InferenceAgent = self.agents.get('inference')
            step2_result = await inference.execute(data)
            result['steps']['inference'] = step2_result
            if not step2_result['success']:
                raise Exception(f"Inference 실패: {step2_result['error']}")
            print(f"✓ {step2_result['execution_time']:.1f}ms")
            
            inference_data = step2_result['data']
            
            # Step 3: XAI
            print("🔍 Step 3: XAI 해석...")
            xai: XAIAgent = self.agents.get('xai')
            # XAI에 필요한 입력 데이터 준비
            xai_input = {
                **data,
                **inference_data,
                'feature_names': data.get('feature_names', [])
            }
            step3_result = await xai.execute(xai_input)
            result['steps']['xai'] = step3_result
            if not step3_result['success']:
                raise Exception(f"XAI 실패: {step3_result['error']}")
            print(f"✓ {step3_result['execution_time']:.1f}ms")
            
            xai_data = step3_result['data']
            
            # Step 4: Report Generation (Classification + XAI만 사용, 조건부 생성)
            print("📄 Step 4: 보고서 생성...")
            report: ReportAgent = self.agents.get('report')
            report_input = {
                **data,
                **inference_data,
                **xai_data,
                'timestamp': data.get('timestamp', '')
            }
            step4_result = await report.execute(report_input)
            result['steps']['report'] = step4_result
            if not step4_result['success']:
                raise Exception(f"Report 실패: {step4_result['error']}")
            print(f"✓ {step4_result['execution_time']:.1f}ms")
            
            # Report가 생성되었는지 확인 (should_report flag)
            report_data = step4_result['data']
            should_report = report_data.get('should_report', False)
            
            if should_report:
                print(f"  → 보고서 생성됨: {report_data.get('report_id', 'N/A')}")
            else:
                reason = report_data.get('reason', '알 수 없음')
                print(f"  → 보고서 생성 안함: {reason}")
            
            # Step 5: Dashboard Update
            print("📊 Step 5: 대시보드 업데이트...")
            dashboard: DashboardAgent = self.agents.get('dashboard')
            dashboard_input = {
                **report_input,
                'summary': report_data.get('summary', '')
            }
            step5_result = await dashboard.execute(dashboard_input)
            result['steps']['dashboard'] = step5_result
            if not step5_result['success']:
                raise Exception(f"Dashboard 실패: {step5_result['error']}")
            print(f"✓ {step5_result['execution_time']:.1f}ms")
            
            # 최종 결과
            result['total_time'] = time.time() - start_time
            
            # 상태 레벨 표시
            level = inference_data.get('severity_level', 0)
            level_emoji = {0: '✅', 1: '⚠️', 2: '🔔', 3: '🚨'}.get(level, '❓')
            print(f"\n{level_emoji} 결과: {inference_data.get('level_name', 'Unknown')}")
            print(f"⏱️ 전체 소요 시간: {result['total_time']*1000:.1f}ms")
            
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            print(f"\n✗ 파이프라인 오류: {e}")
        
        return result

    async def run_continuous(self, duration: Optional[int] = None) -> None:
        """
        연속 모니터링 실행

        Args:
            duration: 실행 시간 (초). None이면 무한 실행
        """
        
        import time
        self.is_running = True
        start_time = time.time()
        iteration = 0
        
        print("\n🔄 연속 모니터링 시작...")
        print(f"{'=' * 60}\n")
        
        try:
            while self.is_running:
                iteration += 1
                print(f"\n{'='*60}")
                print(f"🔵 Iteration {iteration}")
                print(f"{'='*60}")
                
                # 파이프라인 실행
                result = await self.run_pipeline()
                
                if not result['success']:
                    print(f"⚠️ 경고: {result.get('error', 'Unknown error')}")
                
                # 지속 시간 체크
                if duration and (time.time() - start_time) > duration:
                    print(f"\n📍 설정된 시간({duration}초) 도달 - 종료합니다.")
                    break
                
                # 다음 사이클 대기 (1초)
                await asyncio.sleep(0.1)  # UI 갱신 용
        
        except KeyboardInterrupt:
            print("\n\n⚠️ 사용자가 중단했습니다.")
        
        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        """모든 에이전트 정리"""
        print("\n🛑 Agent System 종료 중...")
        print("=" * 50)
        
        for name, agent in self.agents.items():
            await agent.cleanup()
            print(f"✓ {name} 정리 완료")
        
        self.is_running = False
        print("=" * 50)
        print("✓ 모든 에이전트 종료 완료\n")

    def stop(self) -> None:
        """모니터링 중지"""
        self.is_running = False

    async def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        status = {
            'is_running': self.is_running,
            'agents': {}
        }
        
        for name, agent in self.agents.items():
            agent_status = await agent.get_status()
            status['agents'][name] = {
                'state': agent_status.state.value,
                'execution_count': agent_status.execution_count,
                'success_rate': f"{agent_status.success_rate:.1%}",
                'avg_time_ms': f"{agent_status.average_execution_time:.2f}",
                'last_error': agent_status.last_error
            }
        
        return status
