"""
XAIAgent: SHAP 기반 해석

기능:
- SHAP KernelExplainer 사용
- Top-K 특성 추출
- 특성별 기여도 계산
- 이상 특성 자동 식별
"""

import numpy as np
import shap
from typing import Any, Dict, List
from .base_agent import BaseAgent


class XAIAgent(BaseAgent):
    """XAI (SHAP) 해석 에이전트"""

    def __init__(self, model, feature_names: List[str], num_top_features: int = 5):
        """
        Args:
            model: 추론 함수 (입력: np.ndarray, 출력: np.ndarray)
            feature_names: 특성 이름 리스트
            num_top_features: 상위 N개 특성 표시
        """
        super().__init__(
            agent_id="XAI-SHAP",
            agent_type="XAI"
        )
        
        self.model = model
        self.feature_names = feature_names
        self.num_top_features = num_top_features
        self.explainer = None

    async def _initialize(self):
        """SHAP Explainer 생성"""
        # KernelExplainer 사용 (모든 모델 타입 지원)
        # background_data: 설명의 기준이 되는 배경 데이터
        # 간단한 배경 데이터 생성 (정규분포)
        background_data = np.random.normal(0.5, 0.2, (100, len(self.feature_names)))
        background_data = np.clip(background_data, 0, 1)
        
        self.explainer = shap.KernelExplainer(
            self.model,
            background_data
        )
        
        self.logger.info(
            f"SHAP Explainer 생성 완료 "
            f"(특성: {len(self.feature_names)}개)"
        )

    async def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        SHAP 해석 계산

        Args:
            input_data: {
                'normalized_sensors': [float, ...],
                'raw_sensors': [float, ...],
                'predicted_class': int,
                'probabilities': [float, ...],
                'feature_names': [str, ...],
                'timestamp': str
            }

        Returns:
            {
                'shap_values': [float, ...],
                'top_features': [
                    {
                        'name': str,
                        'shap_value': float,
                        'sensor_value': float,
                        'base_value': float,
                        'contribution': float (0-1)
                    },
                    ...
                ],
                'anomaly_features': [
                    {'name': str, 'reason': str},
                    ...
                ]
            }
        """
        
        # 입력 데이터 준비
        sensor_data = np.array(input_data['normalized_sensors']).reshape(1, -1)
        
        # SHAP 값 계산
        shap_values = self.explainer.shap_values(sensor_data)
        
        # 예측 클래스에 해당하는 SHAP 값 추출
        predicted_class = int(input_data['predicted_class'])  # numpy int64 → Python int
        
        # SHAP 값 형태 분석
        # - list of arrays: shap_values = [array(1, 14), array(1, 14), ...]
        # - 3D array: shap_values = array(1, 14, 4)
        if isinstance(shap_values, list):
            # 리스트 형태: 각 클래스별 SHAP 값
            class_shap_values = shap_values[predicted_class][0]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                # 3D array: (batch, features, classes)
                class_shap_values = shap_values[0, :, predicted_class]
            elif shap_values.ndim == 2:
                # 2D array: (batch, features)
                class_shap_values = shap_values[0]
            else:
                # 1D array: (features,)
                class_shap_values = shap_values
        else:
            raise ValueError(f"Unsupported SHAP values type: {type(shap_values)}")
        
        # 절댓값 기반 중요도
        importance = np.abs(class_shap_values)
        
        # Top-K 특성 추출
        top_indices = np.argsort(importance)[-self.num_top_features:][::-1]
        
        top_features = []
        for rank, idx in enumerate(top_indices, 1):
            shap_val = class_shap_values[idx]
            sensor_val = input_data['normalized_sensors'][idx]
            contribution = float(importance[idx] / np.sum(importance))
            
            top_features.append({
                'rank': rank,
                'index': int(idx),
                'name': self.feature_names[idx],
                'shap_value': float(shap_val),
                'sensor_value': float(sensor_val),
                'contribution': contribution,
                'direction': 'increase' if shap_val > 0 else 'decrease'
            })
        
        # 이상 특성 식별 (극값 + 높은 기여도)
        anomaly_features = self._identify_anomalies(
            input_data['normalized_sensors'],
            top_features,
            input_data['raw_sensors']
        )
        
        return {
            'shap_values': class_shap_values.tolist(),
            'top_features': top_features,
            'anomaly_features': anomaly_features,
            'predicted_class': predicted_class,
            'timestamp': input_data.get('timestamp', ''),
            'explanation_text': self._generate_explanation(
                top_features,
                anomaly_features,
                input_data['probabilities']
            )
        }

    def _identify_anomalies(
        self,
        normalized_sensors: List[float],
        top_features: List[Dict],
        raw_sensors: List[float]
    ) -> List[Dict]:
        """이상 특성 식별"""
        anomalies = []
        
        # Top-3 특성 중 극값인 특성 찾기
        for feature in top_features[:3]:
            sensor_val = feature['sensor_value']
            
            # 극값 판정 (0.1 이하 또는 0.9 이상)
            if sensor_val < 0.15 or sensor_val > 0.85:
                anomalies.append({
                    'index': feature['index'],
                    'name': feature['name'],
                    'raw_value': raw_sensors[feature['index']],
                    'normalized_value': sensor_val,
                    'severity': 'HIGH' if (sensor_val < 0.1 or sensor_val > 0.95) else 'MEDIUM',
                    'reason': 'Extreme value detected'
                })
        
        return anomalies

    def _generate_explanation(
        self,
        top_features: List[Dict],
        anomalies: List[Dict],
        probabilities: List[float]
    ) -> str:
        """자연어 해석 생성"""
        
        # 클래스별 설명
        class_descriptions = {
            0: "정상 상태입니다",
            1: "주의 상태입니다",
            2: "경고 상태입니다",
            3: "긴급 상태입니다"
        }
        
        predicted_class = np.argmax(probabilities)
        description = class_descriptions.get(predicted_class, "알 수 없음")
        
        # Top 특성 설명
        top_feature_names = [f['name'] for f in top_features[:3]]
        features_str = ", ".join(top_feature_names)
        
        # 이상 특성 설명
        if anomalies:
            anomaly_names = [a['name'] for a in anomalies]
            anomaly_str = ", ".join(anomaly_names)
            explanation = (
                f"{description}. "
                f"주요 특성: {features_str}. "
                f"이상 특성: {anomaly_str}."
            )
        else:
            explanation = (
                f"{description}. "
                f"주요 특성: {features_str}."
            )
        
        return explanation

    async def _cleanup(self):
        """리소스 정리"""
        self.explainer = None
