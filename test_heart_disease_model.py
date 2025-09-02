"""
Test script for Heart Disease Detection Model

This script validates the functionality of the heart disease detection model
to ensure it works correctly and produces expected results.
"""

import sys
import pandas as pd
import numpy as np
from heart_disease_model import HeartDiseaseDetector

def test_model_initialization():
    """Test model initialization."""
    print("Testing model initialization...")
    detector = HeartDiseaseDetector()
    
    assert detector.models == {}
    assert detector.is_trained == False
    assert detector.feature_columns is None
    assert detector.best_model is None
    
    print("✓ Model initialization test passed")

def test_sample_data_generation():
    """Test sample data generation."""
    print("Testing sample data generation...")
    detector = HeartDiseaseDetector()
    
    # Test with default samples
    data = detector.create_sample_data()
    assert len(data) == 1000
    assert 'heart_disease' in data.columns
    assert data['heart_disease'].nunique() == 2  # Binary classification
    
    # Test with custom samples
    data_small = detector.create_sample_data(n_samples=100)
    assert len(data_small) == 100
    
    print("✓ Sample data generation test passed")

def test_data_loading_and_preprocessing():
    """Test data loading and preprocessing."""
    print("Testing data loading and preprocessing...")
    detector = HeartDiseaseDetector()
    
    # Load data
    data = detector.load_data()
    
    # Preprocess data
    X, y = detector.preprocess_data(data)
    
    assert len(X) == len(y)
    assert len(X.columns) == 11  # All features except target
    assert detector.feature_columns is not None
    
    print("✓ Data loading and preprocessing test passed")

def test_model_training():
    """Test model training."""
    print("Testing model training...")
    detector = HeartDiseaseDetector()
    
    # Load and preprocess data
    data = detector.create_sample_data(n_samples=200)  # Smaller dataset for faster testing
    X, y = detector.preprocess_data(data)
    
    # Train models
    detector.train_models(X, y)
    
    assert detector.is_trained == True
    assert len(detector.models) == 3  # Three models trained
    assert detector.best_model is not None
    assert detector.best_model_name is not None
    
    print("✓ Model training test passed")

def test_predictions():
    """Test model predictions."""
    print("Testing model predictions...")
    detector = HeartDiseaseDetector()
    
    # Load, preprocess and train
    data = detector.create_sample_data(n_samples=200)
    X, y = detector.preprocess_data(data)
    detector.train_models(X, y)
    
    # Test prediction with dictionary
    sample_patient = {
        'age': 50,
        'sex': 1,
        'chest_pain_type': 1,
        'resting_bp': 120,
        'cholesterol': 200,
        'fasting_blood_sugar': 0,
        'resting_ecg': 0,
        'max_heart_rate': 160,
        'exercise_angina': 0,
        'st_depression': 0.5,
        'st_slope': 1
    }
    
    result = detector.predict(sample_patient)
    
    assert 'prediction' in result
    assert 'risk_probability' in result
    assert 'risk_level' in result
    assert 'model_used' in result
    assert 'recommendation' in result
    
    assert result['prediction'] in [0, 1]
    assert 0 <= result['risk_probability'] <= 1
    assert result['risk_level'] in ['Low', 'Medium', 'High']
    
    print("✓ Model predictions test passed")

def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")
    detector = HeartDiseaseDetector()
    
    # Test prediction without training
    try:
        detector.predict({'age': 50})
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    # Test evaluation without training
    try:
        detector.evaluate_models()
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    print("✓ Error handling test passed")

def run_all_tests():
    """Run all tests."""
    print("Heart Disease Detection Model - Test Suite")
    print("=" * 50)
    
    try:
        test_model_initialization()
        test_sample_data_generation()
        test_data_loading_and_preprocessing()
        test_model_training()
        test_predictions()
        test_error_handling()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("The heart disease detection model is working correctly.")
        
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()