"""
DataLoaderAgent: 실시간 CSV 데이터 로딩 및 정규화

기능:
- CSV 파일을 1초 단위로 행 단위로 읽기
- 센서 데이터 정규화 (OHT_fire 원본 scaler 사용)
- 슬라이딩 윈도우 (시계열 맥락 유지)
- 결측치 검증
"""

import asyncio
import pandas as pd
import numpy as np
import pdb
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from .base_agent import BaseAgent


class DataLoaderAgent(BaseAgent):
    """실시간 데이터 로딩 에이전트"""

    def __init__(self, csv_path: str, window_size: int = 16, delay: float = 1.0, 
                 norm_method: str = 'std', scaler=None, scaler_path: Optional[str] = None):
        """
        Args:
            csv_path: CSV 파일 경로
            window_size: 슬라이딩 윈도우 크기 (TimesNet input)
            delay: 각 행 읽기 사이의 지연 시간 (초)
            norm_method: 정규화 방법 ('std', 'minmax', 'robust')
            scaler: 사전에 fit된 sklearn scaler (shared scaler)
            scaler_path: 저장된 scaler.pkl 파일 경로 (학습 시 사용한 scaler)
        """
        super().__init__(
            agent_id=f"DataLoader-{Path(csv_path).stem}",
            agent_type="DataLoader"
        )
        
        self.csv_path = Path(csv_path)
        self.window_size = window_size
        self.delay = delay
        self.norm_method = norm_method
        self.scaler_path = scaler_path
        
        self.df = None
        self.current_index = 0
        self.data_buffer = []  # 슬라이딩 윈도우 버퍼
        
        # Scaler: 우선순위 1) 외부에서 전달받은 scaler
        #        2) scaler.pkl 파일에서 로드
        #        3) 새로 생성
        if scaler is not None:
            self.scaler = scaler  # 이미 fit된 scaler 사용
            self.scaler_source = "passed_argument"
        elif scaler_path is not None:
            # scaler.pkl에서 로드 시도
            scaler_pkl = Path(scaler_path)
            if scaler_pkl.exists():
                self.scaler = joblib.load(scaler_pkl)
                self.scaler_source = f"loaded_from_{scaler_pkl}"
            else:
                self.logger.warning(f"Scaler 파일 없음: {scaler_pkl}. 새로 생성합니다.")
                self._initialize_new_scaler(norm_method)
                self.scaler_source = "new_from_csv"
        else:
            # 정규화 방법에 따라 scaler 초기화
            self._initialize_new_scaler(norm_method)
            self.scaler_source = "new_from_csv"
        
        self.feature_names = None
        self.scaler_fitted = (scaler is not None or scaler_path is not None)  # 전달받거나 로드한 scaler는 이미 fit됨
    
    def _initialize_new_scaler(self, norm_method: str):
        """새로운 scaler 초기화"""
        if norm_method == 'std':
            self.scaler = StandardScaler()
        elif norm_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif norm_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = None

    async def _initialize(self):
        """CSV 로드 및 scaler fit (필요시)"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV 파일 없음: {self.csv_path}")
        
        # CSV 로드
        self.df = pd.read_csv(self.csv_path)
        self.logger.info(f"CSV 로드됨: {self.csv_path} ({len(self.df)} rows)")
        
        # 센서 컬럼 추출 (OHT_fire 원본과 동일)
        # 이 컬럼들은 OHT_fire 프로젝트의 ts_target_col과 동일
        self.feature_names = [
            'NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4',
            'ex_temperature', 'ex_humidity', 'ex_illuminance'
        ]
        
        # 실제 CSV에 있는 센서 컬럼만 추출
        available_features = [f for f in self.feature_names if f in self.df.columns]
        if not available_features:
            # 대체 컬럼 명 확인
            exclude_cols = {'Timestamp', 'collection_datetime', 'tagging_state', 
                          'DateTime', 'Time', 'Date'}
            available_features = [col for col in self.df.columns if col not in exclude_cols]
        
        self.feature_names = available_features
        self.logger.info(f"센서 수: {len(self.feature_names)}, 센서명: {self.feature_names}")
        
        # Scaler fit (전달받지 않은 경우만)
        if not self.scaler_fitted and self.scaler is not None:
            sensor_data = self.df[self.feature_names].values
            self.scaler.fit(sensor_data)
            self.scaler_fitted = True
            self.logger.info(f"Scaler fit 완료 ({self.norm_method})")
        else:
            self.logger.info(f"기존 scaler 사용 (출처: {self.scaler_source})")

    async def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """다음 데이터 행 읽기 및 정규화"""
        # 모든 데이터를 읽었으면 순환
        if self.current_index >= len(self.df):
            self.current_index = 0
        
        # 현재 행 데이터
        row = self.df.iloc[self.current_index]
        
        # Timestamp 추출 (collection_datetime 포맷: "MM-DD HH:MM:SS")
        timestamp = None
        current_time = None
        
        try:
            # 먼저 collection_datetime 시도
            if 'collection_datetime' in row and pd.notna(row['collection_datetime']):
                datetime_str = str(row['collection_datetime']).strip()
                if datetime_str and datetime_str != '':
                    # "08-26 14:54:16" 형식을 "2026-08-26 14:54:16"으로 변환
                    current_time = pd.to_datetime(
                        f"2026-{datetime_str}", 
                        format="%Y-%m-%d %H:%M:%S"
                    )
                    timestamp = current_time.isoformat()
        except Exception as e:
            self.logger.warning(f"collection_datetime 파싱 실패: {e}")
        
        # collection_datetime이 없거나 파싱 실패한 경우
        if not timestamp or timestamp == '':
            try:
                # base_time에서 시작해서 오프셋 추가
                base_time_str = input_data.get('base_time', '')
                if base_time_str and base_time_str != '':
                    base_time = datetime.fromisoformat(base_time_str)
                else:
                    # 첫 행의 datetime을 기준시간으로 사용
                    if current_time is None:
                        current_time = datetime.now()
                    base_time = current_time
                    
                current_time = base_time + pd.Timedelta(seconds=self.current_index)
                timestamp = current_time.isoformat()
            except Exception as e:
                self.logger.warning(f"timestamp 생성 실패: {e}")
                # 최후의 수단: 현재 시간 사용
                current_time = datetime.now()
                timestamp = current_time.isoformat()
        
        # 센서 데이터 추출
        raw_sensors = row[self.feature_names].values.astype(float)
        
        # 정규화 (OHT_fire 원본 scaler 사용)
        if self.scaler is not None:
            normalized_sensors = self.scaler.transform(raw_sensors.reshape(1, -1))[0]
        else:
            # Scaler 없으면 그대로 반환
            normalized_sensors = raw_sensors
        
        # 슬라이딩 윈도우 업데이트
        self.data_buffer.append(normalized_sensors)
        if len(self.data_buffer) > self.window_size:
            self.data_buffer.pop(0)
        
        # 윈도우 크기에 도달할 때까지 버퍼 채우기
        window_data = np.array(self.data_buffer)
        if len(window_data) < self.window_size:
            # 부족한 부분은 0으로 패딩
            padding = np.zeros((self.window_size - len(window_data), 
                              len(self.feature_names)))
            window_data = np.vstack([padding, window_data])
        
        # 다음 인덱스로 이동
        self.current_index += 1
        
        # 지연 시간 (실시간 시뮬레이션)
        await asyncio.sleep(self.delay)
        
        return {
            'timestamp': timestamp,
            'index': self.current_index - 1,  # 현재 행 인덱스
            'raw_sensors': raw_sensors.tolist(),
            'normalized_sensors': normalized_sensors.tolist(),
            'window_data': window_data.tolist(),  # (window_size, num_sensors)
            'feature_names': self.feature_names,
            'total_samples': len(self.df)
        }

    async def _cleanup(self):
        """리소스 정리"""
        self.df = None
        self.data_buffer = []

    async def reset(self):
        """데이터 인덱스 초기화"""
        self.current_index = 0
        self.data_buffer = []
        self.logger.info("DataLoader 리셋됨")
    
    def get_scaler(self):
        """Scaler 객체 반환 (다른 agent와 공유용)"""
        return self.scaler
    
    def inverse_transform(self, normalized_data):
        """정규화된 데이터를 원본 스케일로 역변환"""
        if self.scaler is not None:
            return self.scaler.inverse_transform(normalized_data)
        return normalized_data
    
    def save_scaler(self, save_path: str):
        """
        현재 scaler를 파일로 저장
        
        Args:
            save_path: scaler.pkl 저장 경로
        """
        if self.scaler is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, save_path)
            self.logger.info(f"Scaler 저장됨: {save_path}")
        else:
            self.logger.warning("저장할 scaler가 없습니다.")
    
    @staticmethod
    def load_scaler(scaler_path: str):
        """
        scaler.pkl 파일에서 scaler 로드
        
        Args:
            scaler_path: scaler.pkl 파일 경로
            
        Returns:
            로드된 scaler 객체
        """
        scaler_path = Path(scaler_path)
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            return scaler
        else:
            raise FileNotFoundError(f"Scaler 파일 없음: {scaler_path}")
