"""
BaseAgent: 모든 에이전트의 기본 클래스

생명주기: initialize() → run() → cleanup()
메트릭 자동 수집: 실행 시간, 성공률, 메모리 사용
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class AgentState(Enum):
    """에이전트 상태"""
    IDLE = "IDLE"                          # 유휴 상태
    INITIALIZING = "INITIALIZING"          # 초기화 중
    RUNNING = "RUNNING"                    # 실행 중
    PAUSED = "PAUSED"                      # 일시 정지
    ERROR = "ERROR"                        # 에러 상태
    SHUTDOWN = "SHUTDOWN"                  # 종료 상태


@dataclass
class AgentStatus:
    """에이전트 상태 정보"""
    agent_id: str
    agent_type: str
    state: AgentState
    execution_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    last_execution_time: Optional[datetime] = None
    last_error: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """성공률"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count

    @property
    def average_execution_time(self) -> float:
        """평균 실행 시간"""
        if self.success_count == 0:
            return 0.0
        return self.total_execution_time / self.success_count


class BaseAgent(ABC):
    """모든 에이전트의 기본 클래스"""

    def __init__(self, agent_id: str, agent_type: str):
        """
        에이전트 초기화

        Args:
            agent_id: 에이전트 고유 ID
            agent_type: 에이전트 타입 (예: "DataLoader", "Inference")
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.logger = logging.getLogger(f"{agent_type}[{agent_id}]")
        
        self.status = AgentStatus(
            agent_id=agent_id,
            agent_type=agent_type,
            state=AgentState.IDLE
        )

    async def initialize(self) -> bool:
        """
        에이전트 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        try:
            self.status.state = AgentState.INITIALIZING
            self.logger.info(f"Initializing {self.agent_type} agent...")
            
            await self._initialize()
            
            self.status.state = AgentState.IDLE
            self.logger.info(f"{self.agent_type} agent initialized successfully")
            return True
        
        except Exception as e:
            self.status.state = AgentState.ERROR
            self.status.last_error = str(e)
            self.logger.error(f"Failed to initialize {self.agent_type}: {e}")
            return False

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        에이전트 실행 (메트릭 자동 수집)

        Args:
            input_data: 입력 데이터

        Returns:
            Dict: {
                'success': bool,
                'status': AgentState,
                'data': Any,
                'error': str (있을 경우),
                'execution_time': float (ms)
            }
        """
        start_time = time.time()
        
        try:
            self.status.state = AgentState.RUNNING
            self.status.execution_count += 1
            
            # 에이전트별 실행 로직
            result = await self._run(input_data)
            
            # 메트릭 업데이트
            execution_time = (time.time() - start_time) * 1000  # ms
            self.status.success_count += 1
            self.status.total_execution_time += execution_time
            self.status.min_execution_time = min(
                self.status.min_execution_time, 
                execution_time
            )
            self.status.max_execution_time = max(
                self.status.max_execution_time, 
                execution_time
            )
            self.status.last_execution_time = datetime.now()
            self.status.state = AgentState.IDLE
            
            self.logger.debug(
                f"{self.agent_type} executed in {execution_time:.2f}ms"
            )
            
            return {
                'success': True,
                'status': self.status.state.value,
                'data': result,
                'error': None,
                'execution_time': execution_time
            }
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.status.state = AgentState.ERROR
            self.status.error_count += 1
            self.status.last_error = str(e)
            self.status.last_execution_time = datetime.now()
            
            self.logger.error(
                f"{self.agent_type} execution failed: {e} "
                f"(after {execution_time:.2f}ms)"
            )
            
            return {
                'success': False,
                'status': self.status.state.value,
                'data': None,
                'error': str(e),
                'execution_time': execution_time
            }

    async def cleanup(self) -> bool:
        """
        에이전트 정리 및 리소스 해제

        Returns:
            bool: 정리 성공 여부
        """
        try:
            self.status.state = AgentState.SHUTDOWN
            self.logger.info(f"Cleaning up {self.agent_type} agent...")
            
            await self._cleanup()
            
            self.logger.info(f"{self.agent_type} agent cleaned up")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup {self.agent_type}: {e}")
            return False

    async def get_status(self) -> AgentStatus:
        """에이전트 상태 조회"""
        return self.status

    # 서브클래스에서 구현해야 할 추상 메서드
    @abstractmethod
    async def _initialize(self):
        """에이전트 초기화 로직 (서브클래스 구현)"""
        pass

    @abstractmethod
    async def _run(self, input_data: Dict[str, Any]) -> Any:
        """에이전트 실행 로직 (서브클래스 구현)"""
        pass

    @abstractmethod
    async def _cleanup(self):
        """에이전트 정리 로직 (서브클래스 구현)"""
        pass
