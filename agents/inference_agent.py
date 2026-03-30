"""
InferenceAgent: TimesNet 모델 기반 예측

기능:
- 체크포인트에서 모델 로드
- 배치 추론 (GPU/CPU)
- 확률값 및 신뢰도 계산
"""

import pdb
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
from .base_agent import BaseAgent


class InferenceAgent(BaseAgent):
    """TimesNet 추론 에이전트"""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Args:
            checkpoint_path: 체크포인트 디렉토리 경로
            device: 'cpu' 또는 'cuda'
        """
        super().__init__(
            agent_id=f"Inference-{Path(checkpoint_path).name}",
            agent_type="Inference"
        )
        
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.model = None
        self.config = None
        self.class_names = {
            0: "Normal",
            1: "Grey Zone", 
            2: "Warning",
            3: "Danger"
        }

    async def _initialize(self):
        """모델 로드"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"체크포인트 없음: {self.checkpoint_path}"
            )
        
        # 설정 로드
        config_path = self.checkpoint_path / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.logger.info(f"설정 로드: {config_path}")
        
        # 모델 파일 찾기 (best_model.pth 또는 checkpoint.pth)
        model_path = self.checkpoint_path / 'best_model.pth'
        if not model_path.exists():
            model_path = self.checkpoint_path / 'checkpoint.pth'
        
        if not model_path.exists():
            self.logger.warning(
                f"모델 파일 없음: {model_path}. "
                "Dummy 모델로 동작합니다."
            )
            self.model = None
            return
        
        # TimesNet 모델 임포트 및 로드
        try:
            from models import TimesNet
            from argparse import Namespace
            
            # 모델 생성
            if self.config is None:
                # 기본 설정 (TimesNet 표준 구성)
                self.config = {
                    'task_name': 'classification',
                    'seq_len': 16,
                    'pred_len': 0,
                    'enc_in': 11,
                    'num_class': 4,
                    'd_model': 64,
                    'd_ff': 128,
                    'top_k': 3,
                    'num_kernels': 6,
                    'e_layers': 2,
                    'embed': 'timeF',
                    'freq': 'h',
                    'dropout': 0.1
                }
            
            # Convert config dict to namespace-like object for Model compatibility
            config_obj = Namespace(**self.config) if isinstance(self.config, dict) else self.config
            
            # TimesNet module 내의 Model 클래스 호출
            self.model = TimesNet.Model(config_obj)
            
            # 가중치 로드
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(
                f"모델 로드 완료: {model_path} (device: {self.device})"
            )
        
        except Exception as e:
            self.logger.warning(
                f"TimesNet 모델 로드 실패: {e}. "
                "Dummy 모델로 동작합니다."
            )
            self.model = None

    async def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        모델 추론

        Args:
            input_data: {
                'window_data': (window_size, num_sensors),
                'timestamp': str,
                'feature_names': [str, ...]
            }

        Returns:
            {
                'predicted_class': int (0-3),
                'class_name': str,
                'probabilities': [float, float, float, float],
                'confidence': float,
                'top_class': int,
                'margin': float,
                'timestamp': str
            }
        """
        window_data = np.array(input_data['window_data'])
        
        print(window_data.shape)
        # 입력 데이터 정형화 및 모델 입력 크기에 맞추기
        # (window_size, num_sensors) → (1, enc_in, window_size)
        num_input_sensors = window_data.shape[1]
        expected_sensors = self.config.get('enc_in', num_input_sensors) if isinstance(self.config, dict) else num_input_sensors
        
        # 센서 수가 맞지 않으면 조정
        if num_input_sensors > expected_sensors:
            # 초과 센서 제거 (처음 enc_in개만 사용)
            window_data = window_data[:, :expected_sensors]
            self.logger.warning(
                f"입력 센서 수({num_input_sensors}) > 모델 기대값({expected_sensors}), "
                f"처음 {expected_sensors}개만 사용"
            )
        elif num_input_sensors < expected_sensors:
            # 부족한 센서 패딩 (0으로 채우기)
            padding = np.zeros((window_data.shape[0], expected_sensors - num_input_sensors))
            window_data = np.hstack([window_data, padding])
            self.logger.warning(
                f"입력 센서 수({num_input_sensors}) < 모델 기대값({expected_sensors}), "
                f"부족한 {expected_sensors - num_input_sensors}개는 0으로 패딩"
            )
        
        # (window_size, enc_in) → (1, enc_in, window_size)
        x = torch.from_numpy(window_data[np.newaxis, :, :]).float()
        x = x.to(self.device)
        
        with torch.no_grad():
            # 모델 추론
            if self.model is not None:
                output = self.model(x)
                logits = output.cpu().numpy()[0]
            else:
                # Dummy 예측 (모델 없을 때)
                logits = np.random.randn(4)
        
        # 확률 계산 (softmax)
        probabilities = self._softmax(logits)
        predicted_class = np.argmax(probabilities)
        
        # 신뢰도 계산
        sorted_prob = np.sort(probabilities)[::-1]
        confidence = sorted_prob[0]
        margin = sorted_prob[0] - sorted_prob[1]  # 상위 2개 확률 차이
        
        return {
            'predicted_class': int(predicted_class),
            'class_name': self.class_names.get(int(predicted_class), "Unknown"),
            'probabilities': probabilities.tolist(),
            'confidence': float(confidence),
            'margin': float(margin),
            'timestamp': input_data.get('timestamp', ''),
            'logits': logits.tolist(),  # XAI에서 사용
            'raw_input': window_data.tolist()
        }

    def get_predict_function(self):
        """SHAP 등에서 사용할 수 있는 예측 함수 반환"""
        def predict_fn(x):
            """
            입력: np.ndarray (batch_size, num_features)
            출력: np.ndarray (batch_size, num_classes)
            """
            if x.ndim == 2:
                # (batch, features) 형태
                batch_size = x.shape[0]
                num_features = x.shape[1]
            else:
                # (features,) 형태
                x = x.reshape(1, -1)
                batch_size = 1
                num_features = x.shape[1]
            
            # 센서 수 조정
            expected_sensors = self.config.get('enc_in', num_features) if isinstance(self.config, dict) else num_features
            
            if num_features > expected_sensors:
                x = x[:, :expected_sensors]
            elif num_features < expected_sensors:
                padding = np.zeros((x.shape[0], expected_sensors - num_features))
                x = np.hstack([x, padding])
            
            # (batch, enc_in) → (batch, enc_in, seq_len)
            # SHAP 입력은 (batch, features)이므로 seq_len=1로 취급
            x_reshaped = x[:, np.newaxis, :]  # (batch, 1, enc_in)
            x_reshaped = np.transpose(x_reshaped, (0, 2, 1))  # (batch, enc_in, 1)
            
            x_tensor = torch.from_numpy(x_reshaped).float()
            x_tensor = x_tensor.to(self.device)
            
            with torch.no_grad():
                if self.model is not None:
                    output = self.model(x_tensor)
                    logits = output.cpu().numpy()
                else:
                    # Dummy 예측
                    logits = np.random.randn(batch_size, 4)
            
            # Softmax 확률 반환
            probabilities = np.array([self._softmax(logit) for logit in logits])
            return probabilities
        
        return predict_fn

    async def _cleanup(self):
        """리소스 정리"""
        if self.model is not None:
            self.model = None
        torch.cuda.empty_cache()

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Softmax 함수"""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
