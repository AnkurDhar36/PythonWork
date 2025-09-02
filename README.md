# PythonWork
College Work - Heart Disease Detection ML Project

## Heart Disease Detection Machine Learning Model

This project implements a comprehensive machine learning solution for detecting heart disease using patient data. The system includes multiple ML algorithms, data preprocessing, model evaluation, and prediction capabilities.

### Features

- **Multiple ML Models**: Logistic Regression, Random Forest, and SVM
- **Data Preprocessing**: Automatic data cleaning and feature scaling
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Risk Assessment**: Probability-based risk scoring with recommendations
- **Sample Data Generation**: Creates realistic synthetic patient data for testing

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Basic Usage
```python
from heart_disease_model import HeartDiseaseDetector

# Initialize the detector
detector = HeartDiseaseDetector()

# Load data (uses sample data if none provided)
data = detector.load_data()
X, y = detector.preprocess_data(data)

# Train models
detector.train_models(X, y)

# Evaluate models
detector.evaluate_models()

# Make prediction for a new patient
patient = {
    'age': 55,
    'sex': 1,  # 1 = Male, 0 = Female
    'chest_pain_type': 2,
    'resting_bp': 140,
    'cholesterol': 260,
    'fasting_blood_sugar': 0,
    'resting_ecg': 0,
    'max_heart_rate': 150,
    'exercise_angina': 1,
    'st_depression': 1.5,
    'st_slope': 1
}

result = detector.predict(patient)
print(f"Heart Disease Risk: {result['risk_level']}")
print(f"Probability: {result['risk_probability']:.1%}")
```

#### Running the Complete Demo
```bash
python heart_disease_model.py
```

#### Running Tests
```bash
python test_heart_disease_model.py
```

### Model Features

The model uses the following patient features:
- **age**: Age in years
- **sex**: Gender (1 = male, 0 = female)
- **chest_pain_type**: Type of chest pain (0-3)
- **resting_bp**: Resting blood pressure (mm Hg)
- **cholesterol**: Serum cholesterol (mg/dl)
- **fasting_blood_sugar**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **resting_ecg**: Resting ECG results (0-2)
- **max_heart_rate**: Maximum heart rate achieved
- **exercise_angina**: Exercise induced angina (1 = yes, 0 = no)
- **st_depression**: ST depression induced by exercise
- **st_slope**: Slope of the peak exercise ST segment (0-2)

### Model Performance

The system trains and compares three different algorithms:
- **Logistic Regression**: Linear model with good interpretability
- **Random Forest**: Ensemble method with feature importance analysis
- **SVM**: Support Vector Machine with probability estimates

Performance metrics include:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Output

The model provides:
- **Binary prediction**: Heart disease risk (Yes/No)
- **Risk probability**: Numerical probability (0-1)
- **Risk level**: Categorical assessment (Low/Medium/High)
- **Recommendations**: Clinical guidance based on risk level

### Files

- `heart_disease_model.py`: Main ML model implementation
- `test_heart_disease_model.py`: Test suite for validation
- `requirements.txt`: Python dependencies
- `model_evaluation.png`: Generated visualization of model performance
- `heart_disease_model_summary.txt`: Generated model performance summary

### Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
