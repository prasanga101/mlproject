
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv("untitled.csv")
df.head()


post_mappint = {"Positive":1 , "Negative":2 , "Neutral":0 , "पॉजिटिभ":1 ,"नकारात्मक":2 ,"मध्यम":3 }
df["Sentiment"] = df["Sentiment"].map(post_mappint)

df.head()



X = df.drop("Sentiment",axis =1)
Y = df["Sentiment"]
X.shape



from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming you have already loaded your data from the CSV file into a DataFrame and stored it in the variable df

# Extract the text data from the DataFrame
text_data = df['Post']  # Replace 'Text_Column' with the name of the column containing text data

# Instantiate the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the text data
X_t = tfidf_vectorizer.fit_transform(text_data)

# Check the shape of the transformed data
print(X_t.shape)


print(X_t)


df.head()


from sklearn.model_selection import train_test_split
X_train  , X_test, Y_train , Y_test = train_test_split(X_t , Y , test_size = 0.2 , random_state = 42)


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

print(Y_train)



from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train , Y_train)


y_pred = model.predict(X_test)


print(y_pred)

print(Y_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score( Y_test , y_pred)

print(accuracy)

import re
import string

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove extra white spaces
    text = ' '.join(text.split())
    return text

x = input("Enter the post string : ")
preprocessed_input = preprocess_text(x)

# Convert the preprocessed input into numerical features
numerical_features = tfidf_vectorizer.transform([preprocessed_input])

# Make predictions using the trained model
prediction = model.predict(numerical_features)

if prediction==1:
    print("Positive")
elif prediction==2:
    print("Negative")
else:
    print("Neutral")



