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
    probs = loaded_model.predict(texts)
    return probs
# Route for main page
@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form.get("statement")
        # Gets probabilities for all classes from exp
        # Generates html page to display probabilities
        # Labels for each class for lime explanations
        print(tokenise(text))
        return render_template('exp.html')
    return render_template("home.html")


if __name__ == '__main__':
    app.run()
