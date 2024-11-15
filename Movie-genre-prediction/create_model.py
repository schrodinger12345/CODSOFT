import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

train_data['text'] = train_data['Title'] + " " + train_data['Description']
test_data['text'] = test_data['Title'] + " " + test_data['Description']

X_train, X_val, y_train, y_val = train_test_split(train_data['text'], train_data['Genre'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)  # You can tune max_features for performance
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(test_data['text'])

model = LogisticRegression(multi_class='multinomial', max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_val_tfidf)
# print(classification_report(y_val, y_pred, zero_division=1))

test_data['Predicted_Genre'] = model.predict(X_test_tfidf)

if 'index' not in test_data.columns:
    test_data.reset_index(inplace=True)


test_data[['index', 'Predicted_Genre']].to_csv("test_predictions.csv", index=False)

joblib.dump(model, "genre_prediction_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and predictions saved successfully.")
