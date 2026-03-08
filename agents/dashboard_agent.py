"""
DashboardAgent: Streamlit UI 업데이트

기능:
- 실시간 데이터 수집
- 히스토리 관리
- 메트릭 계산
- 시각화 데이터 준비
"""

from typing import Any, Dict, List, Deque
from collections import deque
from datetime import datetime, timedelta
import numpy as np
from .base_agent import BaseAgent


class DashboardAgent(BaseAgent):
    """대시보드 에이전트 (Streamlit 통합)"""

    def __init__(self, max_history: int = 32):
        """
        Args:
            max_history: 최대 히스토리 길이 (기본 32초 = 최대 32개 포인트)
        """
        super().__init__(
            agent_id="Dashboard",
            agent_type="Dashboard"
        )
        
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.alerts_history = deque(maxlen=100)

    async def _initialize(self):
        """초기화"""
        self.logger.info(f"대시보드 준비 완료 (히스토리: {self.max_history}초)")

    async def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        대시보드 데이터 준비

        Returns:
            {
                'current': {...},
                'history': [...],
                'alerts': [...],
                'metrics': {...},
                'charts_data': {...}
            }
        """
        
        # 현재 데이터
        current = {
            'timestamp': input_data.get('timestamp', datetime.now().isoformat()),
            'level': input_data.get('severity_level', 0),
            'level_name': input_data.get('level_name', 'Unknown'),
            'confidence': input_data.get('confidence', 0),
            'predicted_class': input_data.get('predicted_class', 0),
            'probabilities': input_data.get('probabilities', [0, 0, 0, 0]),
            'summary': input_data.get('summary', ''),
            # 시계열 데이터 추가
            'window_data': input_data.get('window_data', []),  # (window_size, num_sensors)
            'feature_names': input_data.get('feature_names', []),
            'raw_sensors': input_data.get('raw_sensors', []),
            'normalized_sensors': input_data.get('normalized_sensors', []),
            # XAI 데이터 추가
            'top_features': input_data.get('top_features', []),
            'shap_values': input_data.get('shap_values', []),
            'anomaly_features': input_data.get('anomaly_features', []),
            'explanation_text': input_data.get('explanation_text', '')
        }
        
        # 히스토리에 추가
        self.history.append(current)
        
        # 알람이 있으면 알람 히스토리에 추가
        if current['level'] > 0:
            self.alerts_history.append(current)
        
        # 메트릭 계산
        metrics = self._calculate_metrics()
        
        # 차트 데이터 준비
        charts_data = self._prepare_charts_data()
        
        return {
            'current': current,
            'history': list(self.history),
            'alerts': list(self.alerts_history),
            'metrics': metrics,
            'charts_data': charts_data,
            'timestamp': current['timestamp']
        }

    def _calculate_metrics(self) -> Dict[str, Any]:
        """메트릭 계산"""
        
        if not self.history:
            return {
                'normal_rate': 0.0,
                'grey_zone_rate': 0.0,
                'warning_rate': 0.0,
                'danger_rate': 0.0,
                'avg_confidence': 0.0,
                'total_alerts': 0,
                'recent_alert_time': None
            }
        
        history_list = list(self.history)
        levels = [h['level'] for h in history_list]
        confidences = [h['confidence'] for h in history_list]
        
        total = len(history_list)
        
        metrics = {
            'normal_rate': sum(1 for l in levels if l == 0) / total,
            'grey_zone_rate': sum(1 for l in levels if l == 1) / total,
            'warning_rate': sum(1 for l in levels if l == 2) / total,
            'danger_rate': sum(1 for l in levels if l == 3) / total,
            'avg_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences),
            'total_alerts': len(self.alerts_history),
            'recent_alert_time': (
                self.alerts_history[-1]['timestamp'] 
                if self.alerts_history else None
            )
        }
        
        return metrics

    def _prepare_charts_data(self) -> Dict[str, List]:
        """차트 데이터 준비"""
        
        if not self.history:
            return {
                'timestamps': [],
                'levels': [],
                'confidences': [],
                'probabilities_0': [],
                'probabilities_1': [],
                'probabilities_2': [],
                'probabilities_3': []
            }
        
        history_list = list(self.history)
        
        data = {
            'timestamps': [h['timestamp'].split('T')[1] if 'T' in h['timestamp'] else h['timestamp'] 
                          for h in history_list],
            'levels': [h['level'] for h in history_list],
            'confidences': [h['confidence'] for h in history_list],
            'probabilities_0': [h['probabilities'][0] for h in history_list],
            'probabilities_1': [h['probabilities'][1] for h in history_list],
            'probabilities_2': [h['probabilities'][2] for h in history_list],
            'probabilities_3': [h['probabilities'][3] for h in history_list],
        }
        
        return data

    async def _cleanup(self):
        """리소스 정리"""
        self.history.clear()
        self.alerts_history.clear()

    # Streamlit에서 사용할 헬퍼 메서드
    def get_current_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        if self.history:
            return dict(list(self.history)[-1])
        return {}

    def get_history(self, last_n: int = 60) -> List[Dict]:
        """히스토리 조회 (최근 N개)"""
        return list(self.history)[-last_n:]

    def get_alerts(self, last_n: int = 10) -> List[Dict]:
        """알람 히스토리 조회"""
        return list(self.alerts_history)[-last_n:]

    def get_level_distribution(self) -> Dict[str, int]:
        """레벨별 분포"""
        if not self.history:
            return {'Normal': 0, 'Grey Zone': 0, 'Warning': 0, 'Danger': 0}
        
        levels = [h['level'] for h in self.history]
        return {
            'Normal': sum(1 for l in levels if l == 0),
            'Grey Zone': sum(1 for l in levels if l == 1),
            'Warning': sum(1 for l in levels if l == 2),
            'Danger': sum(1 for l in levels if l == 3)
        }
