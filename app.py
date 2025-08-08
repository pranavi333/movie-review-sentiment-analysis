import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('wordnet')

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    filtered = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(filtered)

st.title("🎬 Movie Review Sentiment Analyzer")

user_input = st.text_area("Enter a movie review:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        sentiment = "Positive 😄" if prediction == 1 else "Negative 😞"
        st.success(f"Predicted Sentiment: {sentiment}")
