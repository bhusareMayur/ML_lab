import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flask import Flask, request, jsonify
import joblib
# Example dataset: student-mat.csv (UCI Student Performance dataset)
data = pd.read_csv("student-mat.csv", sep=";")

features = ["studytime", "failures", "absences", "G1", "G2"]
X = data[features]
y = np.where(data["G3"] >= 10, 1, 0)  # 1 = Pass, 0 = Fail

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump((scaler, svm_model), 'student_performance_svm.pkl')

app = Flask(__name__)
scaler, svm_model = joblib.load('student_performance_svm.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = svm_model.predict(scaled_features)[0]
    result = "Pass" if prediction == 1 else "Fail"
    return jsonify({'prediction': result})

if __name__ == "__main__":
    app.run(port=5000, debug=False)
