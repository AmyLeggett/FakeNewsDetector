import numpy as np
from flask import Flask, render_template, session
from flask import request
import pickle
from sklearn import preprocessing
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import gunicorn

app = Flask(__name__)
# 1.21.6
# 1.24.2
class_names = ["false", "half-true", "mostly-true", "true", "barely-true", "pants-on-fire"]
explainer = LimeTextExplainer(class_names=class_names)
app.secret_key = "key"


def get_array(text):
    value = []
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    filename = 'bert-base-multilingual-uncased-sentiment-6label'
    loaded_model = pickle.load(open(filename, 'rb'))
    for i in range(len(text)):
        text[i] = ''.join(text[i])
        text[i] = tokenizer.encode(text[i], return_tensors="pt", max_length=512, truncation=True)
        text[i] = model(text[i])
        text[i] = text[i].logits.detach().numpy()
        proba = loaded_model.predict_proba(text[i])
        value.append(proba.tolist()[0])
    print(value)
    return np.array(value)


@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        session.pop("text", None)
        model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        filename = 'bert-base-multilingual-uncased-sentiment-6label'
        loaded_model = pickle.load(open(filename, 'rb'))
        text = request.form.get("statement")
        session['text'] = text
        state = tokenizer.encode(text.lower(), return_tensors="pt")
        state = model(state)
        state = preprocessing.normalize(state.logits.detach().numpy())
        pred = loaded_model.predict(state)
        print(pred)
        if pred == [0]:
            return render_template("result.html", pred=pred, term='False', color_change='#FF3333')
        if pred == [1]:
            return render_template("result.html", pred=pred, term='Half-true', color_change='#F1A008')
        if pred == [2]:
            return render_template("result.html", pred=pred, term='Mostly-true', color_change='#92DC19')
        if pred == [3]:
            return render_template("result.html", pred=pred, term='True', color_change='#66CC00')
        if pred == [4]:
            return render_template("result.html", pred=pred, term='Barely-true', color_change='#F15208')
        if pred == [5]:
            return render_template("result.html", pred=pred, term='Liar', color_change='#000000')
    return render_template("home.html")


if __name__ == '__main__':
    app.run()


@app.route('/', methods=["GET", "POST"])
def about():
    return render_template("home.html")


@app.route('/exp', methods=["GET", "POST"])
def exp():
    text = session.get('text', None)
    print(text)
    exp = explainer.explain_instance(text, get_array,
                                     num_features=5, num_samples=50)
    exp = exp.as_html()
    session.pop("text", None)
    return render_template("exp.html", exp=exp)
