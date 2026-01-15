from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


@app.route("/")
def home():
    return "Fake News Detection API is running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["news"]

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "Real News"
    else:
        result = "Fake News"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
