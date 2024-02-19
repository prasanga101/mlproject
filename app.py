from flask import Flask, render_template, request
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create Flask app
app = Flask(__name__)

# Load CSV file and preprocess data
df = pd.read_csv("untitled.csv")
post_mapping = {"Positive": 1, "Negative": 2, "Neutral": 0, "पॉजिटिभ": 1, "नकारात्मक": 2, "मध्यम": 3}
df["Sentiment"] = df["Sentiment"].map(post_mapping)
text_data = df['Post']
tfidf_vectorizer = TfidfVectorizer()
X_t = tfidf_vectorizer.fit_transform(text_data)
X_train, X_test, Y_train, Y_test = train_test_split(X_t, df['Sentiment'], test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        preprocessed_input = preprocess_text(text)
        numerical_features = tfidf_vectorizer.transform([preprocessed_input])
        prediction = model.predict(numerical_features)
        if prediction == 1:
            sentiment = "Positive"
        elif prediction == 2:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        return render_template('index.html', text=text, sentiment=sentiment)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
