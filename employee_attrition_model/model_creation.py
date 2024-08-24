import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

#read csv file and selecting relevant feautures 
file_path = 'employee_attrition_model/employee_attrition.csv'
df = pd.read_csv(file_path)
features = [
    'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 
    'WorkLifeBalance', 'YearsAtCompany', 'MonthlyIncome', 'Age'
]
X = df[features].copy()
y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)  
X.fillna(X.mean(), inplace=True)

#Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
classification_reports = []
conf_matrices = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Handling class imbalance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Scaling features
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    # Building Random Forest model
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)

    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    classification_reports.append(classification_rep)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrices.append(conf_matrix)

#model is evaluated
mean_accuracy = sum(accuracies) / len(accuracies)
print(f'Mean Accuracy: {mean_accuracy:.2f}')
print("Classification Report for the last fold:")
print(classification_report(y_test, y_pred))

#saving the model and scaler
joblib.dump(model, "employee_attrition_model/employee_attrition_classification.pkl")
joblib.dump(scaler, "employee_attrition_model/scaler.pkl")