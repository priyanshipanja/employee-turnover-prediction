"""
Test Runner for Employee Turnover Prediction System.
Executes data integrity, pipeline preprocessing, and inference test suites.
"""

import sys
from tests.test_data import test_dataset_generation
from tests.test_pipeline import test_preprocessing_pipeline
from tests.test_predictor import test_single_prediction, test_batch_prediction

def main():
    print("=" * 60)
    print("🧪 RUNNING EMPLOYEE TURNOVER PREDICTION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Dataset Synthesis & Schema Integrity", test_dataset_generation),
        ("Preprocessing & Feature Engineering Pipeline", test_preprocessing_pipeline),
        ("Single Employee Inference & Retention Strategy", test_single_prediction),
        ("Batch Roster Inference & Workforce Aggregations", test_batch_prediction)
    ]
    
    passed = 0
    for name, test_fn in tests:
        print(f"Running: {name}...", end=" ", flush=True)
        try:
            test_fn()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            
    print("=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed cleanly!")
    print("=" * 60)
    
    if passed != len(tests):
        sys.exit(1)

if __name__ == '__main__':
    main()
