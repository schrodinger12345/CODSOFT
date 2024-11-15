import joblib
model = joblib.load(r"SMS-spam\sms_spam_classifier.pkl")
vectorizer = joblib.load(r"SMS-spam\tfidf_vectorizer.pkl")

#Put the message you want to test here
new_data = ["Replace this with your message"]
new_data_tfidf = vectorizer.transform(new_data)

predicted_res = model.predict(new_data_tfidf)
predicted_res = "Spam" if predicted_res[0] == 1 else "Not Spam"
print(predicted_res)