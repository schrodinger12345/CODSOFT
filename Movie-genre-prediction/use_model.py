import joblib
model = joblib.load(r"Movie-genre-prediction\genre_prediction_model.pkl")
vectorizer = joblib.load(r"Movie-genre-prediction\tfidf_vectorizer.pkl")

#Put the movie description here
new_data = ["description of the movie"]
new_data_tfidf = vectorizer.transform(new_data)

predicted_genre = model.predict(new_data_tfidf)
print(predicted_genre)
