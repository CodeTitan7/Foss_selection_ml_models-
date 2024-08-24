import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

#loading model
model = joblib.load('iris_classifier_model/iris_classification.pkl')

#asking user input
sepal_length = float(input("Enter the sepal length (in cm): "))
sepal_width = float(input("Enter the sepal width (in cm): "))
petal_length = float(input("Enter the petal length (in cm): "))
petal_width = float(input("Enter the petal width (in cm): "))

#new_sample = np.array([[4.3, 2.0, 4.1, 2.2]]) 
new_sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

#standarizing sample
scaler = StandardScaler()
new_sample_scaled = scaler.fit_transform(new_sample)

#predicting
predicted_species = model.predict(new_sample_scaled)
print(f"Predicted Species: {predicted_species[0]}")