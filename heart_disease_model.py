"""
Heart Disease Detection Machine Learning Model

This module provides a comprehensive machine learning solution for detecting
heart disease using patient data. It includes data preprocessing, multiple
model implementations, and evaluation metrics.

Author: ML Implementation for Heart Disease Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

class HeartDiseaseDetector:
    """
    A comprehensive heart disease detection system using machine learning.
    
    This class provides functionality for:
    - Data preprocessing and feature engineering
    - Training multiple ML models
    - Model evaluation and comparison
    - Making predictions on new patient data
    """
    
    def __init__(self):
        """Initialize the HeartDiseaseDetector."""
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = None
        self.best_model = None
        self.best_model_name = None
        
    def create_sample_data(self, n_samples=1000):
        """
        Create a sample heart disease dataset for demonstration.
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Generated sample dataset
        """
        np.random.seed(42)
        
        # Generate realistic patient data
        age = np.random.randint(30, 80, n_samples)
        sex = np.random.choice([0, 1], n_samples)  # 0: Female, 1: Male
        chest_pain = np.random.choice([0, 1, 2, 3], n_samples)  # Chest pain types
        resting_bp = np.random.normal(130, 20, n_samples)  # Resting blood pressure
        cholesterol = np.random.normal(240, 50, n_samples)  # Cholesterol level
        fasting_bs = np.random.choice([0, 1], n_samples)  # Fasting blood sugar
        resting_ecg = np.random.choice([0, 1, 2], n_samples)  # Resting ECG results
        max_hr = np.random.normal(150, 25, n_samples)  # Maximum heart rate
        exercise_angina = np.random.choice([0, 1], n_samples)  # Exercise induced angina
        oldpeak = np.random.exponential(1, n_samples)  # ST depression
        st_slope = np.random.choice([0, 1, 2], n_samples)  # ST slope
        
        # Create target variable with realistic relationships
        # Higher risk factors increase probability of heart disease
        risk_score = (
            (age - 40) * 0.02 +
            sex * 0.3 +
            chest_pain * 0.2 +
            (resting_bp - 120) * 0.01 +
            (cholesterol - 200) * 0.002 +
            fasting_bs * 0.3 +
            resting_ecg * 0.1 +
            (160 - max_hr) * 0.01 +
            exercise_angina * 0.4 +
            oldpeak * 0.3 +
            st_slope * 0.1
        )
        
        # Convert to probability and create binary target
        prob = 1 / (1 + np.exp(-risk_score + 2))
        target = np.random.binomial(1, prob, n_samples)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'sex': sex,
            'chest_pain_type': chest_pain,
            'resting_bp': np.clip(resting_bp, 80, 200),
            'cholesterol': np.clip(cholesterol, 150, 400),
            'fasting_blood_sugar': fasting_bs,
            'resting_ecg': resting_ecg,
            'max_heart_rate': np.clip(max_hr, 80, 200),
            'exercise_angina': exercise_angina,
            'st_depression': np.clip(oldpeak, 0, 6),
            'st_slope': st_slope,
            'heart_disease': target
        })
        
        return data
    
    def load_data(self, data=None):
        """
        Load and preprocess the heart disease dataset.
        
        Args:
            data (pd.DataFrame, optional): Custom dataset. If None, creates sample data.
            
        Returns:
            pd.DataFrame: Processed dataset
        """
        if data is None:
            print("No data provided. Creating sample dataset...")
            data = self.create_sample_data()
        
        print(f"Dataset loaded with {len(data)} samples and {len(data.columns)} features")
        print(f"Heart disease cases: {data['heart_disease'].sum()} ({data['heart_disease'].mean()*100:.1f}%)")
        
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess the data for machine learning.
        
        Args:
            data (pd.DataFrame): Raw dataset
            
        Returns:
            tuple: (X, y) - Features and target variables
        """
        # Separate features and target
        X = data.drop('heart_disease', axis=1)
        y = data['heart_disease']
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        print(f"Features: {self.feature_columns}")
        
        return X, y
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """
        Train multiple machine learning models.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of dataset for testing
            random_state (int): Random state for reproducibility
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # Define models
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
            'SVM': SVC(random_state=random_state, probability=True)
        }
        
        print("Training models...")
        
        # Train models and store results
        model_scores = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train_scaled, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            # Test prediction
            y_pred = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # Store model and metrics
            self.models[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_accuracy
            }
            
            model_scores[name] = test_accuracy
            
            print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"Test Accuracy: {test_accuracy:.3f}")
        
        # Select best model
        self.best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.models[self.best_model_name]['model']
        
        print(f"\nBest model: {self.best_model_name} (Accuracy: {model_scores[self.best_model_name]:.3f})")
        
        self.is_trained = True
    
    def evaluate_models(self):
        """Evaluate all trained models and display comprehensive metrics."""
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation. Call train_models() first.")
        
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        for name, model_info in self.models.items():
            model = model_info['model']
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            print(f"\n{name}:")
            print("-" * len(name))
            print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.3f}")
            print(f"Precision: {precision_score(self.y_test, y_pred):.3f}")
            print(f"Recall: {recall_score(self.y_test, y_pred):.3f}")
            print(f"F1-Score: {f1_score(self.y_test, y_pred):.3f}")
            print(f"ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba):.3f}")
        
        # Create comparison visualization
        self.plot_model_comparison()
    
    def plot_model_comparison(self):
        """Create visualizations comparing model performance."""
        if not self.is_trained:
            return
        
        # Prepare data for plotting
        model_names = list(self.models.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        metric_values = {metric: [] for metric in metrics}
        
        for name in model_names:
            model = self.models[name]['model']
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            metric_values['Accuracy'].append(accuracy_score(self.y_test, y_pred))
            metric_values['Precision'].append(precision_score(self.y_test, y_pred))
            metric_values['Recall'].append(recall_score(self.y_test, y_pred))
            metric_values['F1-Score'].append(f1_score(self.y_test, y_pred))
            metric_values['ROC-AUC'].append(roc_auc_score(self.y_test, y_pred_proba))
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Model comparison
        plt.subplot(2, 2, 1)
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i * width, metric_values[metric], width, label=metric, alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width * 2, model_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion Matrix for best model
        plt.subplot(2, 2, 2)
        y_pred = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # ROC Curves
        plt.subplot(2, 2, 3)
        for name in model_names:
            model = self.models[name]['model']
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Feature Importance (for Random Forest)
        if 'Random Forest' in self.models:
            plt.subplot(2, 2, 4)
            rf_model = self.models['Random Forest']['model']
            feature_importance = rf_model.feature_importances_
            feature_names = self.feature_columns
            
            # Sort features by importance
            indices = np.argsort(feature_importance)[::-1]
            
            plt.bar(range(len(feature_importance)), feature_importance[indices], alpha=0.8)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Feature Importance (Random Forest)')
            plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/runner/work/PythonWork/PythonWork/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nVisualization saved as 'model_evaluation.png'")
    
    def predict(self, patient_data):
        """
        Make predictions for new patient data.
        
        Args:
            patient_data (dict or pd.DataFrame): Patient features
            
        Returns:
            dict: Prediction results including probability and risk assessment
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call train_models() first.")
        
        # Convert to DataFrame if dict
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])
        
        # Ensure all required features are present
        for feature in self.feature_columns:
            if feature not in patient_data.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Scale the data
        patient_data_scaled = self.scaler.transform(patient_data[self.feature_columns])
        
        # Make prediction with best model
        prediction = self.best_model.predict(patient_data_scaled)[0]
        prediction_proba = self.best_model.predict_proba(patient_data_scaled)[0]
        
        # Risk assessment
        risk_probability = prediction_proba[1]
        if risk_probability < 0.3:
            risk_level = "Low"
        elif risk_probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        results = {
            'prediction': int(prediction),
            'risk_probability': float(risk_probability),
            'risk_level': risk_level,
            'model_used': self.best_model_name,
            'recommendation': self._get_recommendation(risk_level, risk_probability)
        }
        
        return results
    
    def _get_recommendation(self, risk_level, risk_probability):
        """Generate recommendations based on risk level."""
        if risk_level == "Low":
            return "Continue regular health checkups and maintain a healthy lifestyle."
        elif risk_level == "Medium":
            return "Consider consulting a cardiologist and monitoring heart health more closely."
        else:
            return "Strongly recommend immediate consultation with a cardiologist for comprehensive evaluation."
    
    def save_model_summary(self, filename='heart_disease_model_summary.txt'):
        """Save a summary of the trained models."""
        if not self.is_trained:
            return
        
        filepath = f"/home/runner/work/PythonWork/PythonWork/{filename}"
        
        with open(filepath, 'w') as f:
            f.write("HEART DISEASE DETECTION MODEL SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Best Model: {self.best_model_name}\n")
            f.write(f"Features Used: {', '.join(self.feature_columns)}\n\n")
            
            f.write("Model Performance:\n")
            f.write("-" * 20 + "\n")
            
            for name, model_info in self.models.items():
                f.write(f"\n{name}:\n")
                f.write(f"  Cross-validation accuracy: {model_info['cv_mean']:.3f} ± {model_info['cv_std']:.3f}\n")
                f.write(f"  Test accuracy: {model_info['test_accuracy']:.3f}\n")
        
        print(f"Model summary saved to: {filepath}")


def main():
    """
    Demonstrate the heart disease detection system.
    """
    print("Heart Disease Detection System")
    print("=" * 40)
    
    # Initialize the detector
    detector = HeartDiseaseDetector()
    
    # Load and preprocess data
    data = detector.load_data()
    X, y = detector.preprocess_data(data)
    
    # Train models
    detector.train_models(X, y)
    
    # Evaluate models
    detector.evaluate_models()
    
    # Save model summary
    detector.save_model_summary()
    
    # Example prediction
    print("\n" + "="*40)
    print("EXAMPLE PREDICTION")
    print("="*40)
    
    sample_patient = {
        'age': 55,
        'sex': 1,  # Male
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
    
    result = detector.predict(sample_patient)
    
    print(f"Patient Profile: {sample_patient}")
    print(f"\nPrediction Results:")
    print(f"Heart Disease Risk: {'Yes' if result['prediction'] == 1 else 'No'}")
    print(f"Risk Probability: {result['risk_probability']:.1%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Model Used: {result['model_used']}")
    print(f"Recommendation: {result['recommendation']}")


if __name__ == "__main__":
    main()