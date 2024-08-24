import numpy as np
import pandas as pd
import joblib

#loading the model
loaded_model = joblib.load('studyhours_vs_performance_model/student_performance_model.pkl')

#asking user input
study_hours = float(input("Enter weekly study hours: "))

#predicting percentage for study hours inputted 
new_study_hours = pd.DataFrame({'study_hours': [study_hours]})
predicted_percentage = loaded_model.predict(new_study_hours)[0]
predicted_percentage = min(100, predicted_percentage)
study_hours_value = new_study_hours['study_hours'].iloc[0]

print(f"Predicted Percentage for {study_hours_value} study hours: {predicted_percentage}")