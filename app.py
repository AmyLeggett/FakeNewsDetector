import numpy as np
from flask import Flask, render_template
from flask import request
import pickle

from torch import argmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import torch

app = Flask(__name__)
# Set random seed to make reproducible


# Different numpy version
# 1.21.6
# 1.24.2
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
        value.append(proba.tolist()[0])
    return np.array(value)


# Route for main page
@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Labels for each class for lime explanations
        class_names = ["false", "half-true", "mostly-true", "true", "barely-true", "Pants On Fire"]
        # Initialises lime explainer
        explainer = LimeTextExplainer(class_names=class_names)
        # Gets users text input
        statement = request.form.get("statement")
        text = ''.join(statement)
        # Encode text and put through bert model
        text = tokenizer.encode(text, return_tensors="pt")
        text = model(text)
        text = text.logits.detach().numpy()
        pred = loaded_model.predict(text)
        print(pred)
        if pred == [0]:
            exp = explainer.explain_instance(statement, get_array,
                                             num_features=5, num_samples=50, labels=[0])
            exp = exp.as_html(predict_proba=False)
            return render_template("exp.html", pred=pred, term='False', color_change='#FF3333', exp=exp)
        if pred == [1]:
            exp = explainer.explain_instance(statement, get_array,
                                             num_features=5, num_samples=50, labels=[1])
            exp = exp.as_html(predict_proba=False)
            return render_template("exp.html", pred=pred, term='Half-true', color_change='#FF7C00', exp=exp)
        if pred == [2]:
            exp = explainer.explain_instance(statement, get_array,
                                             num_features=5, num_samples=50, labels=[2])
            exp = exp.as_html(predict_proba=False)
            return render_template("exp.html", pred=pred, term='Mostly-true', color_change='#D4FF00', exp=exp)
        if pred == [3]:
            exp = explainer.explain_instance(statement, get_array,
                                             num_features=5, num_samples=50, labels=[3])
            exp = exp.as_html(predict_proba=False)
            return render_template("exp.html", pred=pred, term='True', color_change='#66CC00', exp=exp)
        if pred == [4]:
            exp = explainer.explain_instance(statement, get_array,
                                             num_features=5, num_samples=50, labels=[4])
            exp = exp.as_html(predict_proba=False)
            return render_template("exp.html", pred=pred, term='Barely-true', color_change='#FF2D00', exp=exp)
        if pred == [5]:
            exp = explainer.explain_instance(statement, get_array,
                                             num_features=5, num_samples=50, labels=[5])
            exp = exp.as_html(predict_proba=False)
            return render_template("exp.html", pred=pred, term='Pants on Fire', color_change='#5A0000', exp=exp)
        # Gets text entered by user from form
        # Explain prediction with the top 5 words , 10 samples and display explanation for the top label
        # exp = explainer.explain_instance(text, get_array,
        # num_features=5, num_samples=10000, top_labels=6)
        # Gets probabilities for all classes from exp
        # prob = exp.predict_proba
        # y = list(prob)
        # Generates html page to display probabilities
    return render_template("home.html")


if __name__ == '__main__':
    app.run()


@app.route('/', methods=["GET", "POST"])
def about():
    return render_template("home.html")
