import joblib
import pandas as pd

#loading model and scaler
loaded_model = joblib.load('employee_attrition_model/employee_attrition_classification.pkl')
scaler = joblib.load('employee_attrition_model/scaler.pkl')

#asking user to input data
env_satisfaction = int(input("Enter Environment Satisfaction (1-4): "))
job_involvement = int(input("Enter Job Involvement (1-4): "))
job_satisfaction = int(input("Enter Job Satisfaction (1-4): "))
work_life_balance = int(input("Enter Work-Life Balance (1-4): "))
years_at_company = int(input("Enter Years at Company: "))
monthly_income = int(input("Enter Monthly Income: "))
age = int(input("Enter Age: "))

new_data = pd.DataFrame({
    'EnvironmentSatisfaction': [env_satisfaction],
    'JobInvolvement': [job_involvement],
    'JobSatisfaction': [job_satisfaction],
    'WorkLifeBalance': [work_life_balance],
    'YearsAtCompany': [years_at_company],
    'MonthlyIncome': [monthly_income],
    'Age': [age]
})

#scaling using scaler
new_data_scaled = scaler.transform(new_data)

#prediction
prediction = loaded_model.predict(new_data_scaled)

if prediction[0] == 1:
    print("Employee Attrition Prediction: Yes")
else:
    print("Employee Attrition Prediction: No")