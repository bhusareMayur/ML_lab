import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flask import Flask, request, jsonify
import joblib

data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

def clean_text(text):
text = re.sub(r'http\S+', '', text)
text = re.sub(r'[^A-Za-z ]+', '', text)
return text.lower()

data['clean_message'] = data['message'].apply(clean_text)
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(

4

data['clean_message'], data['label_num'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=2500)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Model Training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump((vectorizer, model), 'spam_detector.pkl')

app = Flask(__name__)
vectorizer, model = joblib.load('spam_detector.pkl')

@app.route('/predict', methods=['POST'])
def predict():
data = request.get_json(force=True)
text = data['text']
text_clean = clean_text(text)
vector = vectorizer.transform([text_clean])
prediction = model.predict(vector)[0]
label = 'Spam' if prediction == 1 else 'Ham'

5

return jsonify({'prediction': label})
if __name__ == '__main__':
app.run(port=5000, debug=False)
