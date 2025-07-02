import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
texts = [
    "book a flight to London",
    "play music",
    "what's the weather like today",
    "turn on the lights",
    "schedule a meeting at 3 PM"
]
labels = ["travel", "entertainment", "weather", "home", "productivity"]

# Text vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train classifier
clf = MultinomialNB()
clf.fit(X, labels)

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump((vectorizer, clf), f)

print("Model trained and saved to model.pkl")
