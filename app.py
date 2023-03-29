import io

import numpy as np
from flask import Flask, render_template, session
from flask import request
import pickle
from sklearn import preprocessing
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import gunicorn
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'key'

# 1.21.6
# 1.24.2


def get_array(text):
    value = []
    model = AutoModelForSequenceClassification.from_pretrained("muhtasham/bert-small-finetuned-wnut17-ner")
    tokenizer = AutoTokenizer.from_pretrained("muhtasham/bert-small-finetuned-wnut17-ner")
    filename = 'bert-small-finetuned-wnut17-ner-6label'
    loaded_model = pickle.load(open(filename, 'rb'))
    for i in range(len(text)):
        text[i] = ''.join(text[i])
        text[i] = tokenizer.encode(text[i], return_tensors="pt")
        text[i] = model(text[i])
        text[i] = text[i].logits.detach().numpy()
        proba = loaded_model.predict_proba(text[i])
        value.append(proba.tolist()[0])
        print(value)
        np.random.seed(16)
    return np.array(value)


@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        class_names = ["false", "half-true", "mostly-true", "true", "barely-true", "Liar"]
        explainer = LimeTextExplainer(class_names=class_names)
        text = request.form.get("statement")
        exp = explainer.explain_instance(text, get_array,
                                         num_features=5, num_samples=10, labels=(1,))
        prob = exp.predict_proba
        x = class_names
        y = list(prob)
        print(x)
        exp = exp.as_html()
        return render_template('exp.html', exp=exp, y=y)
    return render_template("home.html")


if __name__ == '__main__':
    app.run()


@app.route('/', methods=["GET", "POST"])
def about():
    return render_template("home.html")
