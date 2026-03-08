import numpy as np


def test_junha():
    """Junha의 테스트"""
    print("=" * 60)
    print("Junha의 테스트")
    print("=" * 60)
    
    # Test 1: Normal 상태 - report 생성 안함
    print("\n[Test 1] Normal 상태 - report 생성 안함")
    normal_input = {
        'predicted_class': 0,
        'top_features': [],
        'timestamp': "2024-06-01T12:00:00",
        'feature_names': ['NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4']
    }
    
    # Test 2: Warning 상태 - report 생성
    print("\n[Test 2] Warning 상태 - report 생성")
    warning_input = {
        'predicted_class': 2,
        'top_features': [
            {'name': 'PM10', 'contribution': 0.35},
            {'name': 'NTC', 'contribution': 0.25},
            {'name': 'CT1', 'contribution': 0.15}
        ],
        'timestamp': "2024-06-01T12:05:00",
        'feature_names': ['NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4']
    }
    
    # Test 3: 같은 결과 - report 생성 안함
    print("\n[Test 3] 같은 결과 - report 생성 안함")
    warning_input2 = {
        'predicted_class': 2,
        'top_features': [
            {'name': 'PM10', 'contribution': 0.35},
            {'name': 'NTC', 'contribution': 0.25},
            {'name': 'CT1', 'contribution': 0.15}
        ],
        'timestamp': "2024-06-01T12:10:00",
        'feature_names': ['NTC', 'PM10', 'PM2.5', 'PM1.0', 'CT1', 'CT2', 'CT3', 'CT4']
    }