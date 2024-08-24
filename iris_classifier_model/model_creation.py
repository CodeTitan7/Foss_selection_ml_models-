import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

#read csv file and dropping useless columns
file_path = 'iris_classifier_model/Iris.csv'  
iris_data = pd.read_csv(file_path)
iris_data = iris_data.drop(columns=['Id'])

#data is splitted
X = iris_data.drop(columns=['Species'])
y = iris_data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#standardizing Szfeatures
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#building the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

#model is evaluated
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_rep)

#saving the model
joblib.dump(knn, 'iris_classifier_model/iris_classification.pkl')