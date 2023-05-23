import pickle
import numpy as np
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import torch

app = Flask(__name__)


def tokenise(text):
    # Set random seed to make reproducible
    rng = torch.default_generator
    rng.manual_seed(42)
    # Load model and tokenizer from hugging face
    model = AutoModelForSequenceClassification.from_pretrained("muhtasham/bert-small-finetuned-wnut17-ner")
    tokenizer = AutoTokenizer.from_pretrained("muhtasham/bert-small-finetuned-wnut17-ner")
    # Load pretrained model
    filename = 'model'
    loaded_model = pickle.load(open(filename, 'rb'))
    texts = ''.join(text)
    # Encode text and put through bert model
    texts = tokenizer.encode(texts, return_tensors="pt")
    texts = model(texts)
    texts = texts.logits.detach().numpy()
    # Return prediction
    probs = loaded_model.predict(texts)
    return probs


def get_array(text):
    # Set random seed to make reproducible
    rng = torch.default_generator
    rng.manual_seed(42)
    # Load model and tokenizer from hugging face
    model = AutoModelForSequenceClassification.from_pretrained("muhtasham/bert-small-finetuned-wnut17-ner")
    tokenizer = AutoTokenizer.from_pretrained("muhtasham/bert-small-finetuned-wnut17-ner")
    # Load pretrained model
    filename = 'model'
    loaded_model = pickle.load(open(filename, 'rb'))
    value = []
    for i in range(len(text)):
        text[i] = ''.join(text[i])
        # Encode text and put through bert model
        text[i] = tokenizer.encode(text[i], return_tensors="pt")
        text[i] = model(text[i])
        text[i] = text[i].logits.detach().numpy()
        # Predict probabilities for encoded text through classifier
        proba = loaded_model.predict_proba(text[i])
        # Creates an array of probabilities
        value.append(proba.tolist()[0])
    return np.array(value)


# Route for main page
@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        class_names = ["false", "half-true", "mostly-true", "true", "barely-true", "Pants On Fire"]
        # Initialises lime explainer
        explainer = LimeTextExplainer(class_names=class_names)
        # Gets users text input
        statement = request.form.get("statement")
        # Get prediction for given statement
        pred = tokenise(statement)
        if pred == [0]:
            # Generates LIME explanation and renders the webpage
            exp = explainer.explain_instance(statement, get_array,
                                             num_features=5, num_samples=10, labels=[0])
            exp = exp.as_html(predict_proba=False)
            return render_template("exp.html", pred=pred, term='False', color_change='#FF3333', exp=exp)
        if pred == [1]:
            # Generates LIME explanation and renders the webpage
            exp = explainer.explain_instance(statement, get_array,
                                             num_features=5, num_samples=10, labels=[1])
            exp = exp.as_html(predict_proba=False)
            return render_template("exp.html", pred=pred, term='Half-true', color_change='#FF7C00', exp=exp)
        if pred == [2]:
            # Generates LIME explanation and renders the webpage
            exp = explainer.explain_instance(statement, get_array,
                                             num_features=5, num_samples=10, labels=[2])
            exp = exp.as_html(predict_proba=False)
            return render_template("exp.html", pred=pred, term='Mostly-true', color_change='#D4FF00', exp=exp)
        if pred == [3]:
            # Generates LIME explanation and renders the webpage
            exp = explainer.explain_instance(statement, get_array,
                                             num_features=5, num_samples=10, labels=[3])
            exp = exp.as_html(predict_proba=False)
            return render_template("exp.html", pred=pred, term='True', color_change='#66CC00', exp=exp)
        if pred == [4]:
            # Generates LIME explanation and renders the webpage
            exp = explainer.explain_instance(statement, get_array,
                                             num_features=5, num_samples=10, labels=[4])
            exp = exp.as_html(predict_proba=False)
            return render_template("exp.html", pred=pred, term='Barely-true', color_change='#FF2D00', exp=exp)
        if pred == [5]:
            # Generates LIME explanation and renders the webpage
            exp = explainer.explain_instance(statement, get_array,
                                             num_features=5, num_samples=10, labels=[5])
            exp = exp.as_html(predict_proba=False)
            return render_template("exp.html", pred=pred, term='Pants on Fire', color_change='#5A0000', exp=exp)
        return render_template('exp.html')
    return render_template("home.html")


if __name__ == '__main__':
    app.run()
