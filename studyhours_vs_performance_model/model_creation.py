import pandas as pd  
import numpy as np  
from sklearn.datasets import make_regression  
from sklearn.model_selection import train_test_split, RandomizedSearchCV   
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline  
from sklearn.metrics import mean_squared_error, r2_score
import joblib

#read csv file and modifying data by changing GPA to percentage
file_location = 'studyhours_vs_performance_model/gpa_study_hours.csv'
df = pd.read_csv(file_location)
df = df[df['gpa'] <= 4]
df['percentage'] = df['gpa'] * 25

#to generate synthetic data, to make more data to train the model   
X_synthetic, y_synthetic = make_regression(
    n_samples=150, 
    n_features=1, 
    noise=10, 
    random_state=42
)
X_synthetic = np.clip(X_synthetic, df['study_hours'].min(), df['study_hours'].max())
y_synthetic = np.clip(y_synthetic, df['percentage'].min(), df['percentage'].max())
df_synthetic = pd.DataFrame({
    'study_hours': X_synthetic.flatten(),
    'percentage': y_synthetic
})
df_combined = pd.concat([df, df_synthetic], ignore_index=True)

#data is splitted
X = df_combined[['study_hours']]
y = df_combined['percentage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#defining a pipeline with RandomForest
pipeline = Pipeline([
    ('preprocessor', Pipeline([
        ('poly', PolynomialFeatures()),
        ('scaler', StandardScaler())
    ])),
    ('model', RandomForestRegressor())
])

#setting parameter grid for RandomizedSearchCV
param_grid = {
    'preprocessor__poly__degree': [1, 2, 3, 4],
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20, 30]
}

#randomized searching for find the best model
grid_search = RandomizedSearchCV(pipeline, param_grid, n_iter=20, cv=5, scoring='neg_mean_squared_error', random_state=85)
grid_search.fit(X_train, y_train)

#obtaining the best model
best_model = grid_search.best_estimator_

#model testing and evaluation
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

#saving the model 
joblib.dump(best_model, 'studyhours_vs_performance_model/student_performance_model.pkl')