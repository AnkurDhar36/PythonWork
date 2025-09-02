#!/usr/bin/env python3
"""
Interactive Heart Disease Detection CLI

A simple command-line interface for the heart disease detection model.
Users can input patient data and get immediate risk predictions.
"""

import sys
from heart_disease_model import HeartDiseaseDetector

def get_patient_input():
    """Get patient data from user input."""
    print("\nHeart Disease Risk Assessment")
    print("=" * 40)
    print("Please enter the following patient information:")
    
    try:
        patient_data = {}
        
        # Age
        while True:
            try:
                age = int(input("Age (years): "))
                if 1 <= age <= 120:
                    patient_data['age'] = age
                    break
                else:
                    print("Please enter a valid age (1-120)")
            except ValueError:
                print("Please enter a valid number")
        
        # Sex
        while True:
            sex_input = input("Sex (M/F): ").upper().strip()
            if sex_input in ['M', 'F']:
                patient_data['sex'] = 1 if sex_input == 'M' else 0
                break
            else:
                print("Please enter 'M' for Male or 'F' for Female")
        
        # Chest pain type
        print("\nChest Pain Type:")
        print("0: No chest pain")
        print("1: Typical angina")
        print("2: Atypical angina") 
        print("3: Non-anginal pain")
        while True:
            try:
                cp_type = int(input("Chest pain type (0-3): "))
                if 0 <= cp_type <= 3:
                    patient_data['chest_pain_type'] = cp_type
                    break
                else:
                    print("Please enter a number between 0 and 3")
            except ValueError:
                print("Please enter a valid number")
        
        # Resting blood pressure
        while True:
            try:
                bp = float(input("Resting blood pressure (mm Hg, e.g., 120): "))
                if 50 <= bp <= 250:
                    patient_data['resting_bp'] = bp
                    break
                else:
                    print("Please enter a valid blood pressure (50-250)")
            except ValueError:
                print("Please enter a valid number")
        
        # Cholesterol
        while True:
            try:
                chol = float(input("Cholesterol level (mg/dl, e.g., 200): "))
                if 100 <= chol <= 600:
                    patient_data['cholesterol'] = chol
                    break
                else:
                    print("Please enter a valid cholesterol level (100-600)")
            except ValueError:
                print("Please enter a valid number")
        
        # Fasting blood sugar
        while True:
            fbs_input = input("Fasting blood sugar > 120 mg/dl? (Y/N): ").upper().strip()
            if fbs_input in ['Y', 'N']:
                patient_data['fasting_blood_sugar'] = 1 if fbs_input == 'Y' else 0
                break
            else:
                print("Please enter 'Y' for Yes or 'N' for No")
        
        # Resting ECG
        print("\nResting ECG Results:")
        print("0: Normal")
        print("1: ST-T wave abnormality")
        print("2: Left ventricular hypertrophy")
        while True:
            try:
                ecg = int(input("Resting ECG result (0-2): "))
                if 0 <= ecg <= 2:
                    patient_data['resting_ecg'] = ecg
                    break
                else:
                    print("Please enter a number between 0 and 2")
            except ValueError:
                print("Please enter a valid number")
        
        # Maximum heart rate
        while True:
            try:
                max_hr = float(input("Maximum heart rate achieved (bpm, e.g., 150): "))
                if 60 <= max_hr <= 220:
                    patient_data['max_heart_rate'] = max_hr
                    break
                else:
                    print("Please enter a valid heart rate (60-220)")
            except ValueError:
                print("Please enter a valid number")
        
        # Exercise induced angina
        while True:
            angina_input = input("Exercise induced angina? (Y/N): ").upper().strip()
            if angina_input in ['Y', 'N']:
                patient_data['exercise_angina'] = 1 if angina_input == 'Y' else 0
                break
            else:
                print("Please enter 'Y' for Yes or 'N' for No")
        
        # ST depression
        while True:
            try:
                st_dep = float(input("ST depression induced by exercise (e.g., 1.0): "))
                if 0 <= st_dep <= 10:
                    patient_data['st_depression'] = st_dep
                    break
                else:
                    print("Please enter a valid ST depression value (0-10)")
            except ValueError:
                print("Please enter a valid number")
        
        # ST slope
        print("\nST Slope:")
        print("0: Downsloping")
        print("1: Flat")
        print("2: Upsloping")
        while True:
            try:
                slope = int(input("ST slope (0-2): "))
                if 0 <= slope <= 2:
                    patient_data['st_slope'] = slope
                    break
                else:
                    print("Please enter a number between 0 and 2")
            except ValueError:
                print("Please enter a valid number")
        
        return patient_data
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)

def display_results(patient_data, prediction_results):
    """Display the prediction results in a formatted way."""
    print("\n" + "=" * 50)
    print("HEART DISEASE RISK ASSESSMENT RESULTS")
    print("=" * 50)
    
    # Patient summary
    print("\nPatient Summary:")
    print("-" * 20)
    print(f"Age: {patient_data['age']} years")
    print(f"Sex: {'Male' if patient_data['sex'] == 1 else 'Female'}")
    print(f"Resting BP: {patient_data['resting_bp']} mm Hg")
    print(f"Cholesterol: {patient_data['cholesterol']} mg/dl")
    print(f"Max Heart Rate: {patient_data['max_heart_rate']} bpm")
    
    # Results
    print("\nRisk Assessment:")
    print("-" * 20)
    print(f"Heart Disease Risk: {'POSITIVE' if prediction_results['prediction'] == 1 else 'NEGATIVE'}")
    print(f"Risk Probability: {prediction_results['risk_probability']:.1%}")
    print(f"Risk Level: {prediction_results['risk_level']}")
    print(f"Model Used: {prediction_results['model_used']}")
    
    # Recommendation
    print("\nRecommendation:")
    print("-" * 20)
    print(f"{prediction_results['recommendation']}")
    
    # Disclaimer
    print("\n" + "!" * 50)
    print("IMPORTANT MEDICAL DISCLAIMER")
    print("!" * 50)
    print("This is a machine learning prediction tool for educational")
    print("purposes only. It should NOT be used as a substitute for")
    print("professional medical diagnosis or treatment. Always consult")
    print("with qualified healthcare professionals for medical advice.")
    print("!" * 50)

def main():
    """Main CLI application."""
    print("Heart Disease Detection System - Interactive CLI")
    print("=" * 55)
    
    try:
        # Initialize and train the model
        print("Initializing the heart disease detection model...")
        detector = HeartDiseaseDetector()
        
        # Use sample data to train the model
        data = detector.load_data()
        X, y = detector.preprocess_data(data)
        detector.train_models(X, y)
        
        print("Model trained and ready for predictions!")
        
        while True:
            # Get patient input
            patient_data = get_patient_input()
            
            # Make prediction
            try:
                results = detector.predict(patient_data)
                display_results(patient_data, results)
            except Exception as e:
                print(f"Error making prediction: {str(e)}")
                continue
            
            # Ask if user wants to make another prediction
            print("\n" + "-" * 50)
            while True:
                another = input("Would you like to assess another patient? (Y/N): ").upper().strip()
                if another in ['Y', 'N']:
                    break
                else:
                    print("Please enter 'Y' for Yes or 'N' for No")
            
            if another == 'N':
                break
        
        print("\nThank you for using the Heart Disease Detection System!")
        
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()