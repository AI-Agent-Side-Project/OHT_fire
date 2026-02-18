#!/usr/bin/env python3
"""
Sample-level XAI explanation 검증 스크립트
"""

import json
import os
from pathlib import Path
import numpy as np

def verify_sample_explanations():
    """샘플 설명 파일 검증"""
    
    print("=" * 70)
    print("Sample-Level XAI Explanation Verification")
    print("=" * 70)
    
    # xai_results 폴더 찾기
    xai_results_dir = Path('./xai_results')
    
    if not xai_results_dir.exists():
        print("❌ xai_results directory not found.")
        print("   Please run: python run.py --use_xai")
        return False
    
    # 모델 설정 폴더 찾기
    model_dirs = list(xai_results_dir.glob('*'))
    
    if not model_dirs:
        print("❌ No model result directories found in xai_results/")
        return False
    
    model_dir = model_dirs[0]  # 첫 번째 결과 사용
    print(f"\n✓ Found model results in: {model_dir.name}")
    
    # sample_explanations.json 확인
    sample_exp_file = model_dir / 'sample_explanations.json'
    
    if not sample_exp_file.exists():
        print(f"❌ sample_explanations.json not found in {model_dir.name}")
        print("   This file should be created by xai_shap() function.")
        return False
    
    print(f"✓ sample_explanations.json found ({sample_exp_file.stat().st_size / 1024:.1f} KB)")
    
    # JSON 파일 로드 및 검증
    try:
        with open(sample_exp_file, 'r') as f:
            sample_explanations = json.load(f)
        
        print(f"✓ JSON loaded successfully")
        print(f"  Total samples: {len(sample_explanations)}")
        
        # 첫 번째 샘플 확인
        if sample_explanations:
            first_sample = sample_explanations[0]
            
            print("\n📋 First Sample Explanation Structure:")
            print(f"  - sample_idx: {first_sample.get('sample_idx')}")
            print(f"  - predicted_class: {first_sample.get('predicted_class')}")
            print(f"  - predicted_probability: {first_sample.get('predicted_probability'):.2%}")
            print(f"  - true_class: {first_sample.get('true_class')}")
            print(f"  - is_correct: {first_sample.get('is_correct')}")
            print(f"  - contrast_class: {first_sample.get('contrast_class')}")
            print(f"  - all_class_probabilities: {first_sample.get('all_class_probabilities')}")
            
            top_features = first_sample.get('top_contributing_features', [])
            print(f"\n  📊 Top {len(top_features)} Contributing Features:")
            for rank, feature in enumerate(top_features, 1):
                print(f"    {rank}. Feature {feature['feature_idx']}: {feature['contribution_magnitude']:.6f}")
            
            feature_contributions = first_sample.get('feature_contribution_scores', [])
            print(f"\n  Total feature contribution scores: {len(feature_contributions)}")
            print(f"    Min: {min(feature_contributions):.6f}")
            print(f"    Max: {max(feature_contributions):.6f}")
            print(f"    Mean: {np.mean(feature_contributions):.6f}")
        
        # 통계
        print("\n📊 Sample Explanation Statistics:")
        
        predicted_classes = [s['predicted_class'] for s in sample_explanations]
        true_classes = [s['true_class'] for s in sample_explanations]
        correctness = [s['is_correct'] for s in sample_explanations]
        
        accuracy = np.mean(correctness)
        
        print(f"  - Total samples analyzed: {len(sample_explanations)}")
        print(f"  - Accuracy: {accuracy:.2%}")
        print(f"  - Correct predictions: {sum(correctness)}/{len(sample_explanations)}")
        
        print(f"\n  Predicted class distribution:")
        for class_idx in sorted(set(predicted_classes)):
            count = predicted_classes.count(class_idx)
            print(f"    Class {class_idx}: {count} samples ({100*count/len(predicted_classes):.1f}%)")
        
        print(f"\n  True class distribution:")
        for class_idx in sorted(set(true_classes)):
            count = true_classes.count(class_idx)
            print(f"    Class {class_idx}: {count} samples ({100*count/len(true_classes):.1f}%)")
        
        print("\n" + "=" * 70)
        print("✅ All checks passed!")
        print("=" * 70)
        print("\nUsage in Streamlit:")
        print("  1. Run: streamlit run streamlit_app.py")
        print("  2. Go to '🎯 Sample Explanation' tab")
        print("  3. Select a sample to see why it was predicted as that class")
        print("\nFeatures shown:")
        print("  • Predicted class and confidence")
        print("  • True class and correctness")
        print("  • Class probability distribution")
        print("  • Top features that influenced the prediction")
        print("  • All features contribution heatmap")
        print("  • Interpretation and confidence analysis")
        print("=" * 70)
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    verify_sample_explanations()
