import pickle

import numpy as np
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

import torch

app = Flask(__name__)
# Set random seed to make reproducible
rng = torch.default_generator
rng.manual_seed(42)
# Load model and tokenizer from hugging face
model = AutoModelForSequenceClassification.from_pretrained("muhtasham/bert-small-finetuned-wnut17-ner")
tokenizer = AutoTokenizer.from_pretrained("muhtasham/bert-small-finetuned-wnut17-ner")
# Load pretrained model
filename = 'model'
loaded_model = pickle.load(open(filename, 'rb'))


# Computes predicted probabilities for all classes for lime module
def get_array(text):
    value = []
    for i in range(len(text)):
        text[i] = ''.join(text[i])
        # Encode text and put through bert model
        text[i] = tokenizer.encode(text[i], return_tensors="pt")
        text[i] = model(text[i])
        text[i] = text[i].logits.detach().numpy()
        # Predict probabilities for encoded text through classifier
        proba = loaded_model.predict_proba(text[i])
        probs = loaded_model.predict(text[i])
        value.append(proba.tolist()[0])
    return np.array(value)


# Route for main page
@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form.get("statement")
        texts = ''.join(text)
        # Encode text and put through bert model
        texts = tokenizer.encode(texts, return_tensors="pt")
        texts = model(texts)
        texts = texts.logits.detach().numpy()
        probs = loaded_model.predict(texts)
        # Gets probabilities for all classes from exp
        # Generates html page to display probabilities
        # Labels for each class for lime explanations
        class_names = ["false", "half-true", "mostly-true", "true", "barely-true", "Pants On Fire"]
        # Initialises lime explainer
        explainer = LimeTextExplainer(class_names=class_names)
        # Gets text entered by user from form
        # Explain prediction with the top 5 words , 10 samples and display explanation for the top label
        exp = explainer.explain_instance(text, get_array,
                                         num_features=5, num_samples=10, top_labels=1)
        exp = exp.as_html(predict_proba=False)
        return render_template('exp.html', exp=exp)
    return render_template("home.html")


if __name__ == '__main__':
    app.run()
