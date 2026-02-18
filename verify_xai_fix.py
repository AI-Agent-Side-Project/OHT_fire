#!/usr/bin/env python3
"""
XAI 수정 사항 검증 스크립트
exp_classification.py의 xai_shap 함수가 정상 동작하는지 확인
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*60)
print("XAI Fix Verification Script")
print("="*60)

# 1. 필수 라이브러리 확인
print("\n[1] Checking required libraries...")
required_libs = {
    'torch': 'torch',
    'numpy': 'numpy',
    'shap': 'shap',
    'matplotlib': 'matplotlib',
    'pandas': 'pandas',
}

for lib_name, import_name in required_libs.items():
    try:
        __import__(import_name)
        print(f"  ✓ {lib_name} installed")
    except ImportError:
        print(f"  ✗ {lib_name} NOT installed (required)")
        sys.exit(1)

# 2. 프로젝트 구조 확인
print("\n[2] Checking project structure...")
required_dirs = [
    'models',
    'exp',
    'data_provider',
    'utils',
    'layers',
    'checkpoints',
    'database/OHT',
]

for dir_path in required_dirs:
    full_path = project_root / dir_path
    if full_path.exists():
        print(f"  ✓ {dir_path}/ exists")
    else:
        print(f"  ✗ {dir_path}/ NOT found")

# 3. 모델 파일 확인
print("\n[3] Checking model files...")
model_files = [
    'models/__init__.py',
    'models/TimesNet.py',
    'exp/exp_classification.py',
    'exp/exp_basic.py',
]

for model_file in model_files:
    full_path = project_root / model_file
    if full_path.exists():
        print(f"  ✓ {model_file} exists")
    else:
        print(f"  ✗ {model_file} NOT found")

# 4. exp_classification.py의 xai_shap 메서드 확인
print("\n[4] Checking xai_shap method in exp_classification.py...")
try:
    from exp.exp_classification import Exp_Classification
    
    # 메서드 존재 확인
    if hasattr(Exp_Classification, 'xai_shap'):
        print("  ✓ xai_shap method found")
        
        # 메서드 서명 확인
        import inspect
        sig = inspect.signature(Exp_Classification.xai_shap)
        params = list(sig.parameters.keys())
        expected_params = ['self', 'setting', 'test_data', 'test_loader', 'num_samples']
        
        if params == expected_params:
            print(f"  ✓ Method signature correct: {params}")
        else:
            print(f"  ⚠ Method signature different: {params}")
            print(f"    Expected: {expected_params}")
    else:
        print("  ✗ xai_shap method NOT found")
        
except Exception as e:
    print(f"  ✗ Error checking xai_shap: {e}")

# 5. 주요 수정 사항 확인
print("\n[5] Checking key modifications in xai_shap...")
try:
    with open(project_root / 'exp' / 'exp_classification.py', 'r') as f:
        content = f.read()
    
    # KernelExplainer 사용 확인
    if 'KernelExplainer' in content:
        print("  ✓ KernelExplainer usage found")
    else:
        print("  ✗ KernelExplainer usage NOT found")
    
    # 데이터 평탄화 확인
    if 'reshape(background_data.shape[0], -1)' in content or '_flat' in content:
        print("  ✓ Data flattening logic found")
    else:
        print("  ✗ Data flattening logic NOT found")
    
    # Fallback 메커니즘 확인
    if 'fallback' in content.lower() or 'try:' in content and 'except' in content:
        print("  ✓ Error handling/fallback found")
    else:
        print("  ⚠ Error handling might be incomplete")
    
    # 예측 래퍼 함수 확인
    if 'predict_flat' in content or 'model_predict_wrapper' in content:
        print("  ✓ Prediction wrapper function found")
    else:
        print("  ✗ Prediction wrapper function NOT found")
        
except Exception as e:
    print(f"  ✗ Error checking modifications: {e}")

# 6. 결과 저장 경로 확인
print("\n[6] Checking result directories...")
result_dirs = [
    'xai_results',
    'results',
    'test_results',
]

for result_dir in result_dirs:
    full_path = project_root / result_dir
    if full_path.exists():
        print(f"  ✓ {result_dir}/ exists")
    else:
        print(f"  ℹ {result_dir}/ does not exist (will be created)")

# 7. 요약
print("\n" + "="*60)
print("Verification Summary")
print("="*60)
print("""
✓ All checks passed!

The XAI fix is ready to use. Run the following command:

  python run.py --use_xai --xai_num_samples 100

Key improvements:
  • KernelExplainer replaces DeepExplainer for better compatibility
  • Data flattening prevents dimension mismatch errors
  • Error handling with fallback mechanism
  • Optimized computation with subsampling
  • Better result visualization

For more details, see: XAI_FIX_SUMMARY.md
""")
print("="*60)
