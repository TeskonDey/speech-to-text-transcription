import pickle

def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        vectorizer, model = pickle.load(f)
    return vectorizer, model

def classify_text(text):
    vectorizer, model = load_model()
    X = vectorizer.transform([text])
    prediction = model.predict(X)
    return prediction[0]

if __name__ == "__main__":
    example_text = "play some music"
    label = classify_text(example_text)
    print(f"Predicted label: {label}")
