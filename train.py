import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

train_df = pd.read_csv('train_clean.csv')
test_df = pd.read_csv('test_clean.csv')

vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_df['clean_review'])
X_test = vectorizer.transform(test_df['clean_review'])

y_train = train_df['sentiment']
y_test = test_df['sentiment']

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
